# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Mocker backend configuration.

Implements BackendProtocol for Dynamo mocker engines that simulate GPU workers
without requiring actual GPUs. Useful for testing router logic and benchmarking.
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, Process, WorkerMode


@dataclass(frozen=True)
class MockerServerConfig:
    """Mocker server CLI configuration per mode.

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the mocker worker.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class MockerProtocol:
    """Mocker protocol - implements BackendProtocol.

    Launches Dynamo mocker engines that simulate GPU inference without
    requiring actual hardware. Useful for testing router and queue logic.

    Example YAML:
        backend:
          type: mocker
          mocker_config:
            aggregated:
              model-path: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
              block-size: 64
    """

    type: Literal["mocker"] = "mocker"

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # Mocker server CLI config per mode
    mocker_config: MockerServerConfig | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> SrunConfig:
        """Mocker uses per-process launching (one srun per node)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get config dict for a worker mode."""
        if not self.mocker_config:
            return {}

        if mode == "prefill":
            return dict(self.mocker_config.prefill or {})
        elif mode == "decode":
            return dict(self.mocker_config.decode or {})
        elif mode == "agg":
            return dict(self.mocker_config.aggregated or {})
        return {}

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            return dict(self.prefill_environment)
        elif mode == "decode":
            return dict(self.decode_environment)
        elif mode == "agg":
            return dict(self.aggregated_environment)
        return {}

    def get_process_environment(self, process: Process) -> dict[str, str]:
        """Get process-specific environment variables. Mocker needs none."""
        return {}

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from mocker config, or return default."""
        if self.mocker_config:
            for cfg in [self.mocker_config.aggregated, self.mocker_config.prefill, self.mocker_config.decode]:
                if cfg:
                    name = cfg.get("model-path") or cfg.get("model_path")
                    if name:
                        return name
        return default

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> list[Endpoint]:
        """Allocate endpoints to nodes."""
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = 8081,
    ) -> list[Process]:
        """Convert endpoints to processes. One process per node."""
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port)

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "dynamo",
        profiling_enabled: bool = False,
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
    ) -> list[str]:
        """Build the command to start a mocker worker process."""
        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Base command - use dynamo.mocker module
        cmd: list[str] = [
            "python3",
            "-m",
            "dynamo.mocker",
        ]

        # Add all config flags from mocker_config
        cmd.extend(_config_to_cli_args(config))

        return cmd


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args
