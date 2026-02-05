# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sidecar stage mixin for SweepOrchestrator.

Handles starting auxiliary sidecar processes (e.g., custom Dynamo router, processor).
Sidecars run on the head node after infrastructure (ETCD/NATS) and before the frontend.
"""

import logging
import shlex
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SidecarConfig, SrtConfig

logger = logging.getLogger(__name__)


class SidecarStageMixin:
    """Mixin for sidecar process startup stage.

    Sidecars are auxiliary long-running processes that run on the head node.
    They start after infrastructure (ETCD/NATS) and before the frontend.

    Use cases:
        - Custom Dynamo router (Thompson sampling)
        - Custom Dynamo processor
        - Metrics collectors

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    def start_sidecars(self) -> dict[str, ManagedProcess]:
        """Start all configured sidecar processes.

        Returns:
            Dict mapping sidecar names to ManagedProcess instances.
        """
        if not self.config.sidecars:
            logger.debug("No sidecars configured")
            return {}

        logger.info("Starting %d sidecar(s)", len(self.config.sidecars))
        processes: dict[str, ManagedProcess] = {}

        for name, sidecar_config in self.config.sidecars.items():
            proc = self._start_sidecar(name, sidecar_config)
            processes[proc.name] = proc

        return processes

    def _start_sidecar(self, name: str, config: "SidecarConfig") -> ManagedProcess:
        """Start a single sidecar process.

        Args:
            name: Sidecar name (used for logging and process identification)
            config: Sidecar configuration

        Returns:
            ManagedProcess for the sidecar
        """
        head_node = self.runtime.nodes.head
        logger.info("Starting sidecar '%s' on %s", name, head_node)

        sidecar_log = self.runtime.log_dir / f"sidecar_{name}.out"

        # Determine container image (sidecar-specific > model.container)
        container_image = config.container or str(self.runtime.container_image)

        # Build environment variables
        env_to_set = {
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.infra}:4222",
            "DYN_REQUEST_PLANE": "nats",
        }
        env_to_set.update(config.env)

        # Build bash preamble (setup script + dynamo install) - same pattern as workers
        bash_preamble = self._build_sidecar_preamble()

        # Parse user command into list
        cmd = shlex.split(config.command)

        logger.info("Sidecar '%s' command: %s", name, config.command)
        logger.info("Sidecar '%s' container: %s", name, container_image)
        logger.info("Sidecar '%s' log: %s", name, sidecar_log)

        proc = start_srun_process(
            command=cmd,
            nodelist=[head_node],
            output=str(sidecar_log),
            container_image=container_image,
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
        )

        return ManagedProcess(
            name=f"sidecar_{name}",
            popen=proc,
            log_file=sidecar_log,
            node=head_node,
            critical=True,
        )

    def _build_sidecar_preamble(self) -> str | None:
        """Build bash preamble for sidecar processes (same pattern as workers).

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

        # 2. Dynamo installation (sidecars typically need dynamo for NATS/etcd communication)
        if self.config.dynamo.install:
            parts.append(self.config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)
