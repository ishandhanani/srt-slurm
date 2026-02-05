# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend stage mixin for SweepOrchestrator.

Handles frontend/router and nginx startup.
"""

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends import get_frontend

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@dataclass
class FrontendTopology:
    """Describes where nginx and frontends should run.

    Topology rules:
    - Single node OR multiple_frontends disabled: 1 frontend on head, no nginx
    - 2+ nodes AND multiple_frontends enabled: nginx on head, frontends on other nodes
    """

    nginx_node: str | None  # Node running nginx, or None if no nginx
    frontend_nodes: list[str]  # Nodes running frontends
    frontend_port: int  # Port frontends listen on
    public_port: int  # Public-facing port (nginx or direct frontend)

    @property
    def uses_nginx(self) -> bool:
        """Whether this topology uses nginx."""
        return self.nginx_node is not None


class FrontendStageMixin:
    """Mixin for frontend/nginx startup stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend: BackendProtocol
        self.backend_processes: list[Process]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def backend(self) -> Any:
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @property
    def backend_processes(self) -> list["Process"]:  # type: ignore[empty-body]
        """Compute physical process topology from endpoints (provided by main class)."""
        ...

    def _compute_frontend_topology(self) -> FrontendTopology:
        """Determine where nginx and frontends should run.

        Topology rules:
        - Single node OR multiple_frontends disabled: 1 frontend on head, no nginx
        - 2+ nodes AND multiple_frontends enabled: nginx on head, frontends on other nodes

        Returns:
            FrontendTopology describing where to run nginx and frontends.
        """
        nodes = self.runtime.nodes.worker
        head = self.runtime.nodes.head
        fe_config = self.config.frontend

        # Single node or multiple frontends disabled: single frontend, no nginx
        if len(nodes) == 1 or not fe_config.enable_multiple_frontends:
            return FrontendTopology(
                nginx_node=None,
                frontend_nodes=[head],
                frontend_port=8000,
                public_port=8000,
            )

        # Multiple nodes with multiple frontends enabled:
        # nginx on head, frontends on other nodes
        other_nodes = [n for n in nodes if n != head]

        # Limit number of frontends based on config (num_additional_frontends is extra beyond first)
        max_frontends = min(
            fe_config.num_additional_frontends + 1,
            len(other_nodes),
        )
        frontend_nodes = other_nodes[:max_frontends]

        logger.info(
            "Frontend topology: nginx on %s, %d frontends on %s",
            head,
            len(frontend_nodes),
            frontend_nodes,
        )

        return FrontendTopology(
            nginx_node=head,
            frontend_nodes=frontend_nodes,
            frontend_port=8080,  # Internal port behind nginx
            public_port=8000,  # Public port exposed by nginx
        )

    def _start_nginx(self, topology: FrontendTopology) -> ManagedProcess:
        """Start nginx load balancer on the designated node."""
        assert topology.nginx_node is not None
        logger.info("Starting nginx on %s", topology.nginx_node)

        nginx_log = self.runtime.log_dir / f"{topology.nginx_node}_nginx.out"

        # Generate nginx config from template
        nginx_config = self._generate_nginx_config(topology)
        nginx_config_path = self.runtime.log_dir / "nginx.conf"
        nginx_config_path.write_text(nginx_config)
        logger.debug("Nginx config written to %s", nginx_config_path)

        # Install nginx and run it (daemon off keeps nginx in foreground so srun can manage it)
        # Use container path (/logs) since log_dir is mounted there
        container_config_path = "/logs/nginx.conf"
        cmd = [
            "bash",
            "-c",
            f"nginx -c {container_config_path} -g 'daemon off;'",
        ]

        proc = start_srun_process(
            command=cmd,
            nodelist=[topology.nginx_node],
            output=str(nginx_log),
            container_image=self.config.frontend.nginx_container,
            container_mounts=self.runtime.container_mounts,
            use_bash_wrapper=False,  # Already wrapped in bash -c
            srun_options={
                "container-remap-root": "",
            },
        )

        return ManagedProcess(
            name="nginx",
            popen=proc,
            log_file=nginx_log,
            node=topology.nginx_node,
            critical=True,
        )

    def _generate_nginx_config(self, topology: FrontendTopology) -> str:
        """Generate nginx configuration from template."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent.parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("nginx.conf.j2")

        # Get IPs for frontend nodes
        frontend_hosts = [get_hostname_ip(node) for node in topology.frontend_nodes]

        return template.render(
            frontend_hosts=frontend_hosts,
            backend_port=topology.frontend_port,
            listen_port=topology.public_port,
        )

    def start_frontend(self, registry: "ProcessRegistry") -> list[ManagedProcess]:
        """Start the frontend layer (nginx + frontends if applicable).

        Returns:
            List of ManagedProcess instances for all frontend processes.
        """
        logger.info("Starting frontend layer")
        topology = self._compute_frontend_topology()
        processes: list[ManagedProcess] = []

        # Start nginx if topology requires it
        if topology.uses_nginx:
            nginx_proc = self._start_nginx(topology)
            processes.append(nginx_proc)

        # Handle custom frontend type
        if self.config.frontend.type == "custom":
            custom_proc = self._start_custom_frontend(topology)
            processes.append(custom_proc)
            return processes

        # Get frontend implementation based on config type (dynamo, sglang)
        frontend_impl = get_frontend(self.config.frontend.type)
        frontend_procs = frontend_impl.start_frontends(
            topology=topology,
            runtime=self.runtime,
            config=self.config,
            backend=self.backend,
            backend_processes=self.backend_processes,
        )

        processes.extend(frontend_procs)
        return processes

    def _start_custom_frontend(self, topology: "FrontendTopology") -> ManagedProcess:
        """Start a custom frontend using user-provided command.

        Custom frontends run on the head node and use a user-specified command.
        They must listen on the topology's frontend_port for health checks.

        Args:
            topology: FrontendTopology describing port configuration

        Returns:
            ManagedProcess for the custom frontend
        """
        head_node = self.runtime.nodes.head
        fe_config = self.config.frontend

        logger.info("Starting custom frontend on %s", head_node)
        logger.info("Custom frontend command: %s", fe_config.command)

        frontend_log = self.runtime.log_dir / f"{head_node}_frontend_custom.out"

        # Determine container image (frontend-specific > model.container)
        container_image = fe_config.container or str(self.runtime.container_image)

        # Command is required for custom frontend (validated in schema)
        assert fe_config.command is not None, "Custom frontend requires 'command' field"

        # Build environment variables
        env_to_set = {
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.infra}:4222",
            "DYN_REQUEST_PLANE": "nats",
        }

        # Add frontend env from config
        if fe_config.env:
            env_to_set.update(fe_config.env)

        # Build bash preamble (setup script + dynamo install) - same pattern as workers
        bash_preamble = self._build_custom_frontend_preamble()

        # Parse user command into list
        cmd = shlex.split(fe_config.command)

        logger.info("Custom frontend container: %s", container_image)
        logger.info("Custom frontend log: %s", frontend_log)

        proc = start_srun_process(
            command=cmd,
            nodelist=[head_node],
            output=str(frontend_log),
            container_image=container_image,
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
        )

        return ManagedProcess(
            name="frontend_custom",
            popen=proc,
            log_file=frontend_log,
            node=head_node,
            critical=True,
        )

    def _build_custom_frontend_preamble(self) -> str | None:
        """Build bash preamble for custom frontend (same pattern as workers).

        Runs (in order):
        1. Custom setup script from /configs/ (if config.setup_script set)
        2. Dynamo installation (if dynamo.install is True)

        Returns:
            Preamble string to run before main command, or None if no preamble needed.
        """
        parts = []

        # 1. Run setup script if configured
        if self.config.setup_script:
            script_path = f"/configs/{self.config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )

        # 2. Dynamo installation (custom frontend typically needs dynamo for NATS/etcd communication)
        if self.config.dynamo.install:
            parts.append(self.config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)
