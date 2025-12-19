#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic schema definitions for job configuration validation.
"""

from enum import Enum
import re
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Cluster Configuration (srtslurm.yaml)
# ============================================================================


class ClusterConfig(BaseModel):
    """Cluster configuration from srtslurm.yaml.

    Optional configuration file that provides cluster-specific defaults
    and aliases for model paths and containers.
    """

    model_config = {"extra": "allow"}  # Allow additional fields

    # Default SLURM settings
    default_account: Optional[str] = Field(None, description="Default SLURM account")
    default_partition: Optional[str] = Field(None, description="Default SLURM partition")
    default_time_limit: Optional[str] = Field(None, description="Default job time limit")

    # Resource defaults
    gpus_per_node: Optional[int] = Field(None, description="Default GPUs per node")
    network_interface: Optional[str] = Field(None, description="Network interface (e.g., enP6p9s0np0)")

    # SLURM directive compatibility
    use_gpus_per_node_directive: Optional[bool] = Field(
        True, description="Include #SBATCH --gpus-per-node directive (set False for incompatible clusters)"
    )
    use_segment_sbatch_directive: Optional[bool] = Field(
        True, description="Include #SBATCH --segment directive for segment-based scheduling"
    )

    # Path settings
    srtctl_root: Optional[str] = Field(
        None,
        description="Path to srtctl repo root (where scripts/templates/ lives)",
    )

    # Model path aliases (optional convenience)
    model_paths: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of short names to full model paths",
        examples=[{"deepseek-r1": "/models/deepseek-r1"}],
    )

    # Container aliases (optional convenience)
    containers: Optional[Dict[str, str]] = Field(
        None,
        description="Mapping of short names to container paths",
        examples=[{"latest": "/containers/sglang-latest.sqsh"}],
    )

    # Cloud sync settings (optional)
    cloud: Optional[Dict[str, str]] = Field(
        None,
        description="S3-compatible cloud storage settings for result syncing",
    )


# ============================================================================
# Job Configuration
# ============================================================================


class GpuType(str, Enum):
    """Supported GPU types."""

    GB200 = "gb200"
    GB300 = "gb300"
    H100 = "h100"


class Precision(str, Enum):
    """Model precision/quantization formats."""

    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class BenchmarkType(str, Enum):
    """Benchmark types."""

    MANUAL = "manual"
    SA_BENCH = "sa-bench"
    MMLU = "mmlu"
    GPQA = "gpqa"
    LONGBENCHV2 = "longbenchv2"


class ModelConfig(BaseModel):
    """Model configuration."""

    model_config = {"use_enum_values": True}

    path: str = Field(..., description="Path or alias to model directory")
    container: str = Field(..., description="Path or alias to container image")
    precision: Precision = Field(..., description="Model precision (fp4, fp8, fp16, bf16)")


class ResourceConfig(BaseModel):
    """Resource allocation configuration."""

    model_config = {"use_enum_values": True}

    gpu_type: GpuType = Field(..., description="GPU type (gb200, gb300, h100)")
    gpus_per_node: int = Field(4, description="Number of GPUs per node")

    # Disaggregated mode
    prefill_nodes: Optional[int] = Field(None, description="Number of prefill nodes")
    decode_nodes: Optional[int] = Field(None, description="Number of decode nodes")
    prefill_workers: Optional[int] = Field(None, description="Number of prefill workers")
    decode_workers: Optional[int] = Field(None, description="Number of decode workers")

    # Aggregated mode
    agg_nodes: Optional[int] = Field(None, description="Number of aggregated nodes")
    agg_workers: Optional[int] = Field(None, description="Number of aggregated workers")

    @field_validator("prefill_nodes", "decode_nodes", "agg_nodes")
    @classmethod
    def validate_mode(cls, v, info):
        """Validate that either disagg or agg mode is specified."""
        data = info.data
        has_disagg = any(k in data for k in ["prefill_nodes", "decode_nodes"])
        has_agg = "agg_nodes" in data

        if has_disagg and has_agg:
            raise ValueError("Cannot specify both disaggregated and aggregated mode")

        return v


class SlurmConfig(BaseModel):
    """SLURM job settings."""

    account: Optional[str] = Field(None, description="SLURM account")
    partition: Optional[str] = Field(None, description="SLURM partition")
    time_limit: Optional[str] = Field(None, description="Job time limit (HH:MM:SS)")


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    type: BenchmarkType = Field(BenchmarkType.MANUAL, description="Benchmark type")

    # SA-bench specific
    isl: Optional[int] = Field(None, description="Input sequence length")
    osl: Optional[int] = Field(None, description="Output sequence length")
    concurrencies: Optional[list[int] | str] = Field(
        None, description="Concurrency levels to test (list of ints or x-delimited string like '1x4x8')"
    )
    req_rate: Optional[str] = Field("inf", description="Request rate")

    # Accuracy benchmark arguments
    num_examples: Optional[int] = Field(None, description="Number of examples")
    max_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    repeat: Optional[int] = Field(None, description="Number of times to repeat the benchmark")
    num_threads: Optional[int] = Field(None, description="Number of running threads for accuracy benchmark")
    max_context_length: Optional[int] = Field(None, description="Maximum context length for LongBench-v2 accuracy benchmark")
    categories: Optional[list[str]] = Field(None, description="Comma-separated list of categories to evaluate for LongBench-v2 (None for all)")


class ProfilingType(str, Enum):
    """Supported profiling types."""

    NSYS = "nsys"
    TORCH = "torch"
    NONE = "none"


class ProfilingConfig(BaseModel):
    """Profiling configuration."""

    type: ProfilingType = Field(ProfilingType.NONE, description="Profiling type")
    # Unified profiling spec (used for both prefill and decode in PD
    # disaggregation mode, or for aggregated mode).
    isl: Optional[int] = Field(None, description="Input sequence length")
    osl: Optional[int] = Field(None, description="Output sequence length")
    concurrency: Optional[int] = Field(None, description="Batch size / concurrency")
    start_step: Optional[int] = Field(None, description="Profiling start step")
    stop_step: Optional[int] = Field(None, description="Profiling stop step")


class SGLangPrefillConfig(BaseModel):
    """SGLang prefill worker configuration.

    Accepts any SGLang flags - no required fields.
    """

    model_config = {"extra": "allow"}


class SGLangDecodeConfig(BaseModel):
    """SGLang decode worker configuration.

    Accepts any SGLang flags - no required fields.
    """

    model_config = {"extra": "allow"}


class SGLangAggregatedConfig(BaseModel):
    """SGLang aggregated worker configuration.

    Accepts any SGLang flags - no required fields.
    """

    model_config = {"extra": "allow"}


class SGLangConfig(BaseModel):
    """SGLang backend configuration."""

    prefill: Optional[SGLangPrefillConfig] = None
    decode: Optional[SGLangDecodeConfig] = None
    aggregated: Optional[SGLangAggregatedConfig] = None


class BackendConfig(BaseModel):
    """Backend configuration (auto-populated, not user-facing)."""

    type: Literal["sglang"] = "sglang"  # Only SGLang supported for now

    # Auto-populated from resources.gpu_type + model.precision
    gpu_type: Optional[str] = None

    # Environment variables
    prefill_environment: Optional[dict[str, str]] = None
    decode_environment: Optional[dict[str, str]] = None
    aggregated_environment: Optional[dict[str, str]] = None

    # SGLang-specific config
    sglang_config: Optional[SGLangConfig] = None

    # Frontend / router settings
    enable_multiple_frontends: bool = True
    num_additional_frontends: int = 9
    # Whether to launch sglang_router alongside the workers (PD disaggregation).
    # This is user-configurable via backend.use_sglang_router in the recipe.
    use_sglang_router: bool = False


class JobConfig(BaseModel):
    """Complete job configuration."""

    model_config = {"use_enum_values": True}

    name: str = Field(..., description="Job name")
    model: ModelConfig
    resources: ResourceConfig
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    backend: Optional[BackendConfig] = None  # Auto-populated
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

    # Additional optional settings
    enable_config_dump: bool = True
    extra_mount: Optional[list[str]] = Field(
        default=None,
        description="Additional host-to-container mounts in 'host:container' format.",
    )

    @field_validator("extra_mount")
    @classmethod
    def validate_extra_mount(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("extra_mount must be a list of 'host:container' strings")
        pattern = re.compile(r"^[^:]+:[^:]+$")
        for item in v:
            if not isinstance(item, str):
                raise ValueError("extra_mount entries must be strings")
            if not pattern.match(item):
                raise ValueError("extra_mount entries must be in 'host:container' format")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Auto-populate backend config if not provided."""
        if self.backend is None:
            self.backend = BackendConfig()

        # Auto-populate gpu_type from resources (values are already strings due to use_enum_values)
        if self.backend.gpu_type is None:
            self.backend.gpu_type = f"{self.resources.gpu_type}-{self.model.precision}"

        # Validate profiling mode constraints
        self._validate_profiling_mode()

        # Validate resource allocation
        self._validate_resources()

    def _validate_profiling_mode(self) -> None:
        """Validate profiling mode constraints."""
        prof = getattr(self, "profiling", None)
        if not prof or prof.type in (ProfilingType.NONE, None):
            return

        # Auto-disable config dump when profiling (already handled in backend, but validate here too)
        if self.enable_config_dump:
            # This is fine - backend will handle disabling it
            pass

        # Profiling mode is mutually exclusive with benchmarking
        if self.benchmark and self.benchmark.type != BenchmarkType.MANUAL:
            raise ValueError(
                f"Cannot enable profiling with benchmark type '{self.benchmark.type}'. "
                "Profiling mode is mutually exclusive with benchmarking."
            )

        # Profiling mode requires single worker only
        is_disaggregated = self.resources.prefill_nodes is not None

        if is_disaggregated:
            if self.resources.prefill_workers and self.resources.prefill_workers > 1:
                raise ValueError(
                    f"Profiling mode requires single worker only. "
                    f"Got prefill_workers={self.resources.prefill_workers}"
                )
            if self.resources.decode_workers and self.resources.decode_workers > 1:
                raise ValueError(
                    f"Profiling mode requires single worker only. "
                    f"Got decode_workers={self.resources.decode_workers}"
                )
        else:
            # Aggregated mode
            if self.resources.agg_workers and self.resources.agg_workers > 1:
                raise ValueError(
                    f"Profiling mode requires single worker only. " f"Got agg_workers={self.resources.agg_workers}"
                )

    def _validate_resources(self) -> None:
        """Validate resource allocation and TP size constraints."""
        if not self.backend or not self.backend.sglang_config:
            return

        is_disaggregated = self.resources.prefill_nodes is not None
        gpus_per_node = self.resources.gpus_per_node

        # Validate that sglang_config sections match resource allocation mode
        sglang_cfg = self.backend.sglang_config
        has_prefill_cfg = sglang_cfg.prefill is not None
        has_decode_cfg = sglang_cfg.decode is not None
        has_agg_cfg = hasattr(sglang_cfg, "aggregated") and sglang_cfg.aggregated is not None

        if is_disaggregated:
            # Disaggregated resources but no prefill/decode config
            if not has_prefill_cfg and not has_decode_cfg:
                raise ValueError(
                    "Disaggregated mode (prefill_nodes/decode_nodes) requires "
                    "prefill and decode sections in sglang_config. "
                    f"Found: prefill={has_prefill_cfg}, decode={has_decode_cfg}"
                )
            # Has aggregated config but using disaggregated resources
            if has_agg_cfg:
                raise ValueError(
                    "Cannot use aggregated sglang_config section with disaggregated resources. "
                    "Use prefill and decode sections instead, or switch to agg_nodes/agg_workers."
                )
        else:
            # Aggregated resources but has decode config
            if has_decode_cfg:
                raise ValueError(
                    "Cannot use decode sglang_config section with aggregated resources "
                    "(agg_nodes/agg_workers). Use disaggregated resources "
                    "(prefill_nodes/decode_nodes) or use aggregated section in sglang_config."
                )

        # Validate disaggregated mode
        if is_disaggregated:
            # Validate prefill resources
            if self.backend.sglang_config.prefill:
                prefill_config = self.backend.sglang_config.prefill
                self._validate_worker_resources(
                    mode="prefill",
                    config=prefill_config,
                    nodes=self.resources.prefill_nodes,
                    workers=self.resources.prefill_workers,
                    gpus_per_node=gpus_per_node,
                )

            # Validate decode resources
            if self.backend.sglang_config.decode:
                decode_config = self.backend.sglang_config.decode
                self._validate_worker_resources(
                    mode="decode",
                    config=decode_config,
                    nodes=self.resources.decode_nodes,
                    workers=self.resources.decode_workers,
                    gpus_per_node=gpus_per_node,
                )
        else:
            # Validate aggregated mode
            if hasattr(self.backend.sglang_config, "aggregated") and self.backend.sglang_config.aggregated:
                agg_config = self.backend.sglang_config.aggregated
                self._validate_worker_resources(
                    mode="aggregated",
                    config=agg_config,
                    nodes=self.resources.agg_nodes,
                    workers=self.resources.agg_workers,
                    gpus_per_node=gpus_per_node,
                )
            # Aggregated can also use "prefill" section
            elif self.backend.sglang_config.prefill:
                prefill_config = self.backend.sglang_config.prefill
                self._validate_worker_resources(
                    mode="aggregated",
                    config=prefill_config,
                    nodes=self.resources.agg_nodes,
                    workers=self.resources.agg_workers,
                    gpus_per_node=gpus_per_node,
                )

    def _validate_worker_resources(self, mode: str, config: Any, nodes: int, workers: int, gpus_per_node: int) -> None:
        """Validate that worker configuration fits available resources.

        Args:
            mode: Worker mode (prefill, decode, aggregated)
            config: SGLang config dict for this mode
            nodes: Number of nodes allocated
            workers: Number of workers
            gpus_per_node: GPUs per node

        Raises:
            ValueError: If resource constraints are violated
        """
        if not config or not nodes or not workers:
            return

        # Get TP size from config (we only care about tensor parallelism, not DP/EP)
        tp_size = None

        # Config can be a Pydantic model or dict - handle both
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = config

        if isinstance(config_dict, dict):
            tp_size = (
                config_dict.get("tensor-parallel-size")
                or config_dict.get("tensor_parallel_size")
                or config_dict.get("tp-size")
                or config_dict.get("tp_size")
            )

        if not tp_size:
            # No TP size specified, can't validate
            return

        # Convert to int if it's a string (can happen during sweep template expansion)
        try:
            tp_size = int(tp_size)
        except (ValueError, TypeError):
            # Template placeholder like "{tp_size}" - skip validation
            return

        # Calculate resources needed
        # Each worker needs tp_size GPUs (DP/EP don't affect GPU requirements per worker)
        total_gpus_available = nodes * gpus_per_node
        gpus_per_worker = tp_size
        total_gpus_needed = gpus_per_worker * workers

        # Validate: Total GPUs needed <= Total GPUs available
        if total_gpus_needed > total_gpus_available:
            raise ValueError(
                f"{mode.capitalize()} resource mismatch:\n"
                f"  Workers: {workers}\n"
                f"  TP size: {tp_size}\n"
                f"  GPUs per worker: {gpus_per_worker}\n"
                f"  Total GPUs needed: {total_gpus_needed}\n"
                f"  Total GPUs available: {total_gpus_available} ({nodes} nodes × {gpus_per_node} GPUs/node)\n"
                f"  → Need {total_gpus_needed - total_gpus_available} more GPUs!"
            )

        # Validate: Each worker's GPUs fit on the allocated nodes
        # For multi-node workers, TP size should span across nodes per worker
        nodes_per_worker = nodes // workers if workers > 0 else nodes

        if nodes_per_worker == 0:
            raise ValueError(
                f"{mode.capitalize()} resource mismatch:\n"
                f"  Workers: {workers}\n"
                f"  Nodes: {nodes}\n"
                f"  → Each worker needs at least 1 node, but {nodes} nodes / {workers} workers = {nodes_per_worker} nodes/worker"
            )

        gpus_per_worker_from_nodes = nodes_per_worker * gpus_per_node

        if gpus_per_worker > gpus_per_worker_from_nodes:
            raise ValueError(
                f"{mode.capitalize()} resource mismatch:\n"
                f"  Workers: {workers}\n"
                f"  Nodes per worker: {nodes_per_worker}\n"
                f"  GPUs per worker (from TP): {gpus_per_worker}\n"
                f"  GPUs available per worker: {gpus_per_worker_from_nodes} ({nodes_per_worker} nodes × {gpus_per_node} GPUs/node)\n"
                f"  → Each worker needs {gpus_per_worker - gpus_per_worker_from_nodes} more GPUs!"
            )
