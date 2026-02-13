"""
Pydantic schema for rate-matching sweep configuration.

Defines a YAML-friendly config that drives the full sweep pipeline:
  CTX-only SOL -> GEN-only SOL -> rate-matching -> Pareto -> E2E validation.

The gen_sweep section supports named groups with zip/grid expansion,
matching srt-slurm's native sweep semantics.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Model identity."""
    path: str = Field(..., description="Model name or path alias (resolved by srtslurm.yaml)")
    container: str = Field(
        default="nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1.post1",
        description="Container image URI",
    )
    precision: str = Field(default="fp8", description="Model precision")


class WorkloadConfig(BaseModel):
    """ISL / OSL workload definition."""
    isl: int = Field(..., description="Input sequence length")
    osl: int = Field(..., description="Output sequence length")
    random_ratio: float = Field(
        default=1.0,
        description=(
            "Random ratio for output token sampling. 1.0 = fully random, "
            "0.0 = deterministic. Used in gen_req_rate calculation: "
            "avg_random_ratio = (random_ratio + 1) / 2"
        ),
    )
    mtp_accept_rates: Optional[dict[int, float]] = Field(
        default=None,
        description=(
            "MTP accept rates: effective tokens per decode step per user. "
            "Required when any gen_sweep item uses mtp_num > 0. "
            "Key = mtp_num (1, 2, 3), value = accept rate. "
            "MTP-0 (STP) is always 1.0 and does not need to be specified. "
            "Example: {1: 1.8, 2: 2.28, 3: 2.56}. "
            "These values are model- and ISL-dependent and must be measured "
            "or sourced from prior benchmarks."
        ),
    )


class ResourceConfig(BaseModel):
    """Hardware / GPU settings."""
    gpu_type: str = Field(default="h200", description="GPU type string")
    gpus_per_node: int = Field(default=8, description="GPUs per SLURM node")
    ctx_gpus_per_instance: int = Field(
        default=8,
        description="GPUs per prefill worker (TP size). Typically 8 for DSR1.",
    )
    gen_gpus_per_instance: int = Field(
        default=8,
        description="GPUs per decode worker (TP size). Typically 8 for DSR1.",
    )
    max_total_gpus: int = Field(
        default=64,
        description=(
            "Maximum total GPU budget for rate-matching allocation search. "
            "Constrains ctx_gpus*ctx_instances + gen_gpus*gen_instances."
        ),
    )


class CTXConfig(BaseModel):
    """CTX-only SOL benchmark settings."""
    benchmark_concurrency: int = Field(
        default=32,
        description="sa-bench concurrency for CTX-only run (high enough to saturate prefill).",
    )
    max_batch_size: Optional[int] = Field(default=None, description="Override prefill max_batch_size")
    max_num_tokens: Optional[int] = Field(default=None, description="Override prefill max_num_tokens")
    free_gpu_memory_fraction: Optional[float] = Field(default=None, description="Override prefill KV cache fraction")


# ---------------------------------------------------------------------------
# GEN sweep items and groups
# ---------------------------------------------------------------------------

class GenSweepItem(BaseModel):
    """Single GEN configuration to sweep.

    Each item becomes one GEN-only SOL job. Fields map to the decode worker
    config (batch_size, max_num_tokens, attention_dp, etc.) and the sa-bench
    concurrency.
    """
    mode: Literal["tep", "dep"] = Field(..., description="TEP or DEP parallelism")
    batch_size: int = Field(..., description="Decode max_batch_size")
    concurrency: Union[int, list[int]] = Field(
        ...,
        description=(
            "Per-worker concurrency for SOL benchmark. "
            "Single int or list (multiple concurrencies in one job, e.g. [8, 32, 64])."
        ),
    )
    tp_size: int = Field(default=8, description="Tensor parallelism size")
    mtp_num: int = Field(default=0, description="MTP layers (0 = STP, 1+ = MTP)")
    max_num_tokens: Optional[int] = Field(
        default=None,
        description="Decode max_num_tokens. Defaults to batch_size if not set.",
    )
    gpu_memory_fraction: Optional[float] = Field(
        default=None,
        description="Decode KV cache GPU memory fraction. Default depends on workload.",
    )
    eplb_num_slots: int = Field(default=0, description="Expert Load Balancer slots (0 = disabled)")

    @model_validator(mode="after")
    def _set_defaults(self):
        if self.max_num_tokens is None:
            # For MTP, tokens per step = batch_size * (1 + mtp_num)
            if self.mtp_num > 0:
                object.__setattr__(self, "max_num_tokens", self.batch_size * (1 + self.mtp_num))
            else:
                object.__setattr__(self, "max_num_tokens", self.batch_size)
        return self


class GenSweepGroup(BaseModel):
    """Named group of GEN configs with zip or grid expansion.

    Follows srt-slurm sweep semantics:
      - zip: pairs parameters by index (like Python zip)
      - grid: Cartesian product of all parameter values

    Example YAML:
      tep_mtp1:
        mode: zip
        parameters:
          concurrency: [8, 16, 32, 64]
          batch_size: [128, 128, 256, 256]
        defaults:
          mode: tep
          mtp_num: 1
    """
    expansion: Literal["zip", "grid"] = Field(
        default="zip", description="How to combine parameter lists",
    )
    parameters: dict[str, list[Any]] = Field(
        ..., description="Parameter lists to expand",
    )
    defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Default values applied to every expanded item",
    )

    def expand(self) -> list[GenSweepItem]:
        """Expand group into concrete GenSweepItem list."""
        param_names = list(self.parameters.keys())
        param_lists = list(self.parameters.values())

        if self.expansion == "zip":
            combos = list(zip(*param_lists, strict=False))
        else:  # grid
            combos = list(itertools.product(*param_lists))

        items = []
        for values in combos:
            merged = {**self.defaults, **dict(zip(param_names, values))}
            items.append(GenSweepItem(**merged))
        return items


# ---------------------------------------------------------------------------
# E2E validation settings
# ---------------------------------------------------------------------------

class E2EValidationSettings(BaseModel):
    """Controls E2E validation behaviour after Pareto extraction.

    For each Pareto point we run len(concurrency_multipliers) E2E jobs.
    System concurrency = int(per_worker_conc * gen_instances * multiplier).
    """
    concurrency_multipliers: list[float] = Field(
        default=[1.0, 1.05],
        description=(
            "Multipliers applied to per_worker_conc * gen_instances. "
            "1.0 = baseline, 1.05 = 5%% headroom to keep decode queues fed."
        ),
    )
    tpot_tolerance_pct: float = Field(
        default=15.0,
        description="TPOT diff %% threshold for pass/fail.",
    )
    throughput_tolerance_pct: float = Field(
        default=20.0,
        description="Throughput diff %% threshold for pass/fail.",
    )
    ttft_constraint_ms: Optional[float] = Field(
        default=5000.0,
        description=(
            "Soft TTFT constraint in ms. Pareto points exceeding this are flagged. "
            "Set to None to disable."
        ),
    )


# ---------------------------------------------------------------------------
# Sweep settings
# ---------------------------------------------------------------------------

class SweepSettings(BaseModel):
    """Automation knobs for the sweep pipeline."""
    poll_interval: int = Field(default=300, description="SLURM job poll interval in seconds")
    max_retries: int = Field(default=3, ge=0, description="Max retries per job on failure")
    run_e2e_validation: bool = Field(default=True, description="Run E2E after Pareto extraction")
    parallel_submissions: bool = Field(
        default=False,
        description=(
            "If True, submit all jobs in a phase at once then poll. "
            "If False, submit one-at-a-time (serialised). "
            "Parallel is faster on large clusters."
        ),
    )
    e2e_validation: E2EValidationSettings = Field(
        default_factory=E2EValidationSettings,
        description="E2E validation parameters",
    )


# ---------------------------------------------------------------------------
# Backend config template
# ---------------------------------------------------------------------------

class BackendConfig(BaseModel):
    """Template for the trtllm_config / environment sections.

    If not provided, generate_configs.py derives sensible defaults from
    the workload, mode, and existing H200 recipes.
    """
    prefill_environment: Optional[dict[str, str]] = None
    decode_environment: Optional[dict[str, str]] = None
    trtllm_prefill_overrides: Optional[dict[str, Any]] = None
    trtllm_decode_overrides: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Top-level sweep config
# ---------------------------------------------------------------------------

class RateMatchingSweepConfig(BaseModel):
    """Top-level rate-matching sweep configuration (loaded from YAML).

    The `gen_sweep` field accepts either:
      - A flat list of GenSweepItem dicts
      - A dict of named GenSweepGroup dicts (expanded automatically)
    """
    name: str = Field(..., description="Sweep name for identification and output paths")
    engine_type: Literal["trtllm"] = Field(
        default="trtllm",
        description="Inference engine. Only trtllm is currently supported.",
    )
    model: ModelConfig = Field(..., description="Model identity")
    workload: WorkloadConfig = Field(..., description="ISL / OSL workload")
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    ctx_config: CTXConfig = Field(default_factory=CTXConfig)
    gen_sweep: list[GenSweepItem] = Field(
        ..., description="GEN configurations to sweep (expanded from groups if needed)",
    )
    settings: SweepSettings = Field(default_factory=SweepSettings)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    @model_validator(mode="before")
    @classmethod
    def _expand_gen_sweep_groups(cls, values: dict[str, Any]) -> dict[str, Any]:
        """If gen_sweep is a dict of groups, expand them into a flat list."""
        gs = values.get("gen_sweep")
        if gs is None:
            return values

        if isinstance(gs, dict):
            expanded: list[dict] = []
            for _group_name, group_data in gs.items():
                if isinstance(group_data, dict) and "parameters" in group_data:
                    group = GenSweepGroup(**group_data)
                    expanded.extend(item.model_dump() for item in group.expand())
                else:
                    # Treat as a single GenSweepItem dict
                    expanded.append(group_data)
            values["gen_sweep"] = expanded

        return values

    @model_validator(mode="after")
    def _validate_mtp_accept_rates(self) -> "RateMatchingSweepConfig":
        """Require mtp_accept_rates when any GEN sweep item uses MTP."""
        mtp_levels_used: set[int] = set()
        for item in self.gen_sweep:
            if item.mtp_num > 0:
                mtp_levels_used.add(item.mtp_num)

        if not mtp_levels_used:
            return self

        if self.workload.mtp_accept_rates is None:
            # Known reference values to suggest (DSR1, random workloads)
            known = {
                1024: {1: 1.8, 2: 2.28, 3: 2.56},
                8192: {1: 1.84, 2: 2.38, 3: 2.76},
                32768: {1: 1.97, 2: 2.39, 3: 2.56},
            }
            isl = self.workload.isl
            suggestion = known.get(isl, known[1024])
            # Only include levels actually used
            filtered = {k: v for k, v in suggestion.items() if k in mtp_levels_used}

            raise ValueError(
                f"workload.mtp_accept_rates is required when using MTP "
                f"(gen_sweep uses mtp_num={sorted(mtp_levels_used)}).\n"
                f"\n"
                f"MTP accept rates are the effective tokens per decode step per user.\n"
                f"They are model- and ISL-dependent. Measure them or use known values.\n"
                f"\n"
                f"Reference values for DSR1 (ISL={isl}, random workload):\n"
                f"  mtp_accept_rates:\n"
                + "".join(f"    {k}: {v}\n" for k, v in sorted(filtered.items()))
                + f"\n"
                f"Add this to your sweep YAML under 'workload:'."
            )

        # Validate that all used MTP levels have rates defined
        missing = mtp_levels_used - set(self.workload.mtp_accept_rates.keys())
        if missing:
            raise ValueError(
                f"workload.mtp_accept_rates is missing entries for "
                f"mtp_num={sorted(missing)}. "
                f"Your gen_sweep uses these MTP levels but no accept rate "
                f"is defined for them."
            )

        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> RateMatchingSweepConfig:
    """Load and validate a sweep config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RateMatchingSweepConfig(**raw)
