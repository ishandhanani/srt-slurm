# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
vLLM backend configuration.

Implements BackendProtocol for vLLM inference serving with prefill/decode disaggregation.
Uses dynamo.vllm integration module.
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
    from srtctl.core.topology import Endpoint, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM server CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker. These are passed directly to
    vLLM's AsyncEngineArgs.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class VLLMProtocol:
    """vLLM protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    Example YAML:
        backend:
          type: vllm
          connector: nixl  # default connector (can be overridden per mode)
          prefill_environment:
            PYTHONUNBUFFERED: "1"
          vllm_config:
            prefill:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.9
              connector: kvbm  # override connector for prefill
            decode:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.85
              # uses default connector (nixl)
    """

    type: Literal["vllm"] = "vllm"

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # vLLM server CLI config per mode
    vllm_config: VLLMServerConfig | None = None

    # Default KV connector(s): "nixl", "kvbm", ["kvbm", "nixl"], etc.
    # Can be overridden per mode by setting "connector" in vllm_config.prefill/decode/aggregated
    # Passed as --connector flag to dynamo.vllm
    connector: str | list[str] | None = "nixl"

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> SrunConfig:
        """vLLM uses per-process launching (one srun per node)."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.vllm_config:
            return {}

        if mode == "prefill":
            return dict(self.vllm_config.prefill or {})
        elif mode == "decode":
            return dict(self.vllm_config.decode or {})
        elif mode == "agg":
            return dict(self.vllm_config.aggregated or {})
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
        """Get process-specific environment variables for vLLM workers.

        vLLM with dynamo requires unique ports for each worker:
        - DYN_VLLM_KV_EVENT_PORT: ZMQ port for KV events publishing
        - VLLM_NIXL_SIDE_CHANNEL_PORT: Port for NIXL side channel transfers
        """
        env: dict[str, str] = {}
        if process.kv_events_port is not None:
            env["DYN_VLLM_KV_EVENT_PORT"] = str(process.kv_events_port)
        if process.nixl_port is not None:
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(process.nixl_port)
        return env

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from vLLM config, or return default."""
        if self.vllm_config:
            for cfg in [self.vllm_config.prefill, self.vllm_config.aggregated, self.vllm_config.decode]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
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

    def _is_dp_mode(self, mode: WorkerMode) -> bool:
        """Check if this mode uses Data Parallel + Expert Parallel pattern.

        DP+EP mode is detected when data-parallel-size is set in the mode's config.
        In this mode, each GPU runs its own process (rather than TP across GPUs).
        """
        config = self.get_config_for_mode(mode)
        return config.get("data-parallel-size") is not None or config.get("data_parallel_size") is not None

    def _get_dp_size(self, mode: WorkerMode) -> int | None:
        """Get the data-parallel-size for a mode, or None if not in DP mode."""
        config = self.get_config_for_mode(mode)
        return config.get("data-parallel-size") or config.get("data_parallel_size")

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = 8081,
    ) -> list[Process]:
        """Convert endpoints to processes.

        For DP+EP mode (data-parallel-size set), creates one process per GPU.
        For standard TP mode, creates one process per node.
        """
        from srtctl.core.topology import NodePortAllocator, Process, endpoints_to_processes

        # Check if any endpoint uses DP mode
        has_dp_mode = any(self._is_dp_mode(ep.mode) for ep in endpoints)

        if not has_dp_mode:
            # Standard TP mode: one process per node
            return endpoints_to_processes(endpoints, base_sys_port=base_sys_port)

        # DP+EP mode: one process per GPU
        processes: list[Process] = []
        current_sys_port = base_sys_port
        port_allocator = NodePortAllocator()

        for endpoint in endpoints:
            if not self._is_dp_mode(endpoint.mode):
                # Non-DP endpoints get standard processing
                # (This shouldn't happen in practice since all modes should be consistent)
                for node_rank, node in enumerate(endpoint.nodes):
                    is_leader = node_rank == 0
                    http_port = port_allocator.next_http_port(node) if is_leader else 0
                    bootstrap_port = (
                        port_allocator.next_bootstrap_port(node) if endpoint.mode == "prefill" and is_leader else None
                    )
                    kv_events_port = port_allocator.next_kv_events_port()
                    nixl_port = port_allocator.next_nixl_port()

                    processes.append(
                        Process(
                            node=node,
                            gpu_indices=endpoint.gpu_indices,
                            sys_port=current_sys_port,
                            http_port=http_port,
                            endpoint_mode=endpoint.mode,
                            endpoint_index=endpoint.index,
                            node_rank=node_rank,
                            bootstrap_port=bootstrap_port,
                            kv_events_port=kv_events_port,
                            nixl_port=nixl_port,
                        )
                    )
                    current_sys_port += 1
            else:
                # DP+EP mode: one process per GPU
                # Each process gets a single GPU and a unique dp_rank
                dp_rank = 0
                for _node_rank, node in enumerate(endpoint.nodes):
                    for gpu_idx in sorted(endpoint.gpu_indices):
                        is_leader = dp_rank == 0
                        http_port = port_allocator.next_http_port(node) if is_leader else 0
                        bootstrap_port = (
                            port_allocator.next_bootstrap_port(node)
                            if endpoint.mode == "prefill" and is_leader
                            else None
                        )
                        kv_events_port = port_allocator.next_kv_events_port()
                        nixl_port = port_allocator.next_nixl_port()

                        processes.append(
                            Process(
                                node=node,
                                gpu_indices=frozenset([gpu_idx]),  # Single GPU per process
                                sys_port=current_sys_port,
                                http_port=http_port,
                                endpoint_mode=endpoint.mode,
                                endpoint_index=endpoint.index,
                                node_rank=dp_rank,  # dp_rank stored in node_rank for now
                                bootstrap_port=bootstrap_port,
                                kv_events_port=kv_events_port,
                                nixl_port=nixl_port,
                            )
                        )
                        current_sys_port += 1
                        dp_rank += 1

        return processes

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
        """Build the command to start a vLLM worker process.

        Args:
            process: The process to start
            endpoint_processes: All processes for this endpoint (for multi-node)
            runtime: Runtime context with paths and settings
            frontend_type: Frontend type (currently only "dynamo" supported for vLLM)
            profiling_enabled: Whether profiling is enabled
            nsys_prefix: Optional nsys profiling command prefix
            dump_config_path: Path to dump config JSON
        """
        from srtctl.core.slurm import get_hostname_ip

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Get leader IP for distributed init
        leader_ip = get_hostname_ip(endpoint_nodes[0])

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Get served model name from config or use model path name
        served_model_name = self.get_served_model_name(runtime.model_path.name)

        # Start with nsys prefix if provided
        cmd: list[str] = list(nsys_prefix) if nsys_prefix else []

        # Base command - use dynamo.vllm module
        cmd.extend(
            [
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                model_arg,
                "--served-model-name",
                served_model_name,
            ]
        )

        # Disaggregation flags (different from SGLang's --disaggregation-mode)
        if mode == "prefill":
            cmd.append("--is-prefill-worker")
        elif mode == "decode":
            cmd.append("--is-decode-worker")
        # agg mode: no flag (default behavior)

        # KV connector - check for mode-specific override first, then fall back to default
        # Pop from config so it doesn't get added again by _config_to_cli_args
        mode_connector = config.pop("connector", None)
        connector = mode_connector if mode_connector is not None else self.connector

        if isinstance(connector, list):
            cmd.extend(["--connector"] + connector)
        elif connector and connector not in ("null", "none", None):
            cmd.extend(["--connector", connector])

        # Check if this is DP+EP mode (data-parallel-size set)
        is_dp_mode = self._is_dp_mode(mode)

        if is_dp_mode:
            # DP+EP mode: each GPU runs its own process
            # process.node_rank is the dp_rank (set in endpoints_to_processes)
            dp_rank = process.node_rank
            dp_rpc_port = config.pop("data-parallel-rpc-port", None) or config.pop("data_parallel_rpc_port", 13345)

            cmd.extend(
                [
                    "--data-parallel-rank",
                    str(dp_rank),
                    "--data-parallel-address",
                    leader_ip,
                    "--data-parallel-rpc-port",
                    str(dp_rpc_port),
                ]
            )
            # Note: --data-parallel-size is added via _config_to_cli_args from vllm_config
        elif is_multi_node:
            # Standard TP+PP multi-node coordination flags
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--master-addr",
                    leader_ip,
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

            # Non-leader nodes run headless
            if node_rank > 0:
                cmd.append("--headless")

        # Add config dump path
        if dump_config_path:
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        # Add all config flags from vllm_config
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
