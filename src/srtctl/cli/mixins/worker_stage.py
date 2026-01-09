# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker stage mixin for SweepOrchestrator.

Handles starting backend worker processes (prefill/decode/agg).
"""

import logging
import shlex
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess, NamedProcesses
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)


class WorkerStageMixin:
    """Mixin for worker process startup stage.

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
    def backend_processes(self) -> list["Process"]:
        """Compute physical process topology from endpoints (cached)."""
        ...

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        ...

    def _build_worker_preamble(self) -> str | None:
        """Build bash preamble for worker processes.

        Runs (in order):
        1. Custom setup script from /configs/ (if config.setup_script set)
        2. Dynamo installation (if frontend type is dynamo)
        """
        parts = []

        # 1. Custom setup script (runs first)
        if self.config.setup_script:
            script_path = f"/configs/{self.config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )

        # 2. Dynamo installation (required for dynamo frontend)
        if self.config.frontend.type == "dynamo":
            parts.append(self.config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)

    def start_worker(self, process: "Process", endpoint_processes: list["Process"]) -> ManagedProcess:
        """Start a single worker process."""
        mode = process.endpoint_mode
        index = process.endpoint_index

        logger.info("Starting %s worker %d on %s", mode, index, process.node)

        # Log and config files
        worker_log = self.runtime.log_dir / f"{process.node}_{mode}_w{index}.out"
        config_dump = self.runtime.log_dir / f"{process.node}_config.json"

        # Profiling setup
        profiling = self.config.profiling
        nsys_prefix = None
        if profiling.is_nsys:
            # nsys runs inside the container; /logs is the mounted host log directory.
            nsys_output = f"/logs/profiles/{process.node}_{mode}_w{index}_profile"
            nsys_prefix = profiling.get_nsys_prefix(nsys_output, frontend_type=self.config.frontend.type)

        # Build command using backend's method
        cmd = self.backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            frontend_type=self.config.frontend.type,
            nsys_prefix=nsys_prefix,
            dump_config_path=config_dump,
        )

        # Environment variables
        env_to_set = {
            "HEAD_NODE_IP": self.runtime.head_node_ip,
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.head}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.head}:4222",
            "DYN_SYSTEM_PORT": str(process.sys_port),
        }

        # Keep request-plane consistent across frontend/workers. Dynamo frontend defaults to NATS request plane.
        if self.config.frontend.type == "dynamo" and "DYN_REQUEST_PLANE" not in env_to_set:
            frontend_plane = None
            if self.config.frontend.env:
                frontend_plane = self.config.frontend.env.get("DYN_REQUEST_PLANE")
            env_to_set["DYN_REQUEST_PLANE"] = frontend_plane if frontend_plane else "nats"

        # Add mode-specific environment variables from backend
        env_to_set.update(self.backend.get_environment_for_mode(mode))

        # Add config environment variables
        env_to_set.update(self.runtime.environment)

        # Add profiling environment variables
        if profiling.enabled:
            profile_dir = str(self.runtime.log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs
        if len(process.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install)
        bash_preamble = self._build_worker_preamble()

        proc = start_srun_process(
            command=cmd,
            nodelist=[process.node],
            output=str(worker_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
        )

        return ManagedProcess(
            name=f"{mode}_{index}_{process.node}",
            popen=proc,
            log_file=worker_log,
            node=process.node,
            critical=True,
        )

    def start_all_workers(self) -> NamedProcesses:
        """Start all backend workers."""
        logger.info("Starting backend workers")

        grouped: dict[tuple, list[Process]] = defaultdict(list)
        for process in self.backend_processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}
        for _endpoint_key, endpoint_processes in grouped.items():
            for process in endpoint_processes:
                managed = self.start_worker(process, endpoint_processes)
                result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result
