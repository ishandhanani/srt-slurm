# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM native disaggregated serving frontend implementation.

Uses trtllm-serve disaggregated with static worker URLs (no NATS/etcd needed).
"""

import logging
import shlex
from typing import TYPE_CHECKING, Any

import yaml

from srtctl.core.health import WorkerHealthResult, check_trtllm_serve_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class TRTLLMServeFrontend:
    """TRT-LLM native disaggregated serving frontend.

    Uses trtllm-serve disaggregated with a generated disagg_config.yaml
    containing static worker URLs. No NATS/etcd infrastructure required.

    Health checks via /health endpoint (HTTP 200 = all workers ready).
    """

    @property
    def type(self) -> str:
        return "trtllm-serve"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse trtllm-serve /health endpoint response."""
        return check_trtllm_serve_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend args dict to CLI arguments."""
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
    ) -> list["ManagedProcess"]:
        """Start trtllm-serve disaggregated orchestrators on designated nodes.

        Generates a disagg_config.yaml with static worker URLs grouped by
        context (prefill) and generation (decode) servers, then launches
        trtllm-serve disaggregated on each frontend node.
        """
        from srtctl.core.processes import ManagedProcess

        # Collect leader IPs/ports grouped by mode
        context_urls: list[str] = []
        generation_urls: list[str] = []

        for process in backend_processes:
            if not process.is_leader:
                continue
            leader_ip = get_hostname_ip(process.node)
            url = f"{leader_ip}:{process.http_port}"
            if process.endpoint_mode == "prefill":
                context_urls.append(url)
            elif process.endpoint_mode == "decode":
                generation_urls.append(url)

        # trtllm-serve disaggregated is its own orchestrator/router — only one instance
        node = topology.frontend_nodes[0]
        logger.info("Starting trtllm-serve disaggregated on %s", node)

        frontend_log = runtime.log_dir / f"{node}_trtllm_serve.out"

        # Generate disagg_config.yaml
        disagg_config = {
            "hostname": "0.0.0.0",
            "port": topology.frontend_port,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": len(context_urls),
                "urls": list(context_urls),
            },
            "generation_servers": {
                "num_instances": len(generation_urls),
                "urls": list(generation_urls),
            },
        }

        host_config_path = runtime.log_dir / "disagg_config.yaml"
        host_config_path.write_text(yaml.safe_dump(disagg_config))

        cmd = [
            "trtllm-serve",
            "disaggregated",
            "-c",
            "/logs/disagg_config.yaml",
            "-t",
            "7200"
        ]
        cmd.extend(self.get_frontend_args_list(config.frontend.args))

        logger.info("trtllm-serve command: %s", shlex.join(cmd))

        # Build env vars
        env_to_set: dict[str, str] = {}
        if config.frontend.env:
            env_to_set.update(config.frontend.env)

        proc = start_srun_process(
            command=cmd,
            nodelist=[node],
            output=str(frontend_log),
            container_image=str(runtime.container_image),
            container_mounts=runtime.container_mounts,
            env_to_set=env_to_set if env_to_set else None,
            mpi="pmix",
        )

        return [
            ManagedProcess(
                name="trtllm_serve",
                popen=proc,
                log_file=frontend_log,
                node=node,
                critical=True,
            )
        ]
