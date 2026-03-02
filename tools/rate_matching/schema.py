"""
Pydantic schema for rate-matching sweep configuration.

Defines a YAML-friendly config that drives the full sweep pipeline:
  CTX-only SOL -> GEN-only SOL -> rate-matching -> Pareto -> E2E validation.

The gen_sweep section supports named groups with zip/grid expansion,
matching srt-slurm's native sweep semantics.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, Union

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
        default=0.8,
        description=(
            "Random ratio for output token sampling. Must match sa_bench default (0.8) "
            "for SA methodology. Used in gen_req_rate calculation: "
            "avg_random_ratio = (random_ratio + 1) / 2"
        ),
    )
    mtp_accept_rates: dict[int, float] | None = Field(
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
    max_batch_size: int | None = Field(default=None, description="Override prefill max_batch_size")
    max_num_tokens: int | None = Field(default=None, description="Override prefill max_num_tokens")
    free_gpu_memory_fraction: float | None = Field(default=None, description="Override prefill KV cache fraction")


# ---------------------------------------------------------------------------
# GEN sweep items and groups
# ---------------------------------------------------------------------------

class GenSweepItem(BaseModel):
    """Single GEN configuration to sweep.

    Each item becomes one GEN-only SOL job. Fields map to the decode worker
    config (batch_size, max_num_tokens, attention_dp, etc.) and the sa-bench
    concurrency.

    Per-item overrides (decode_overrides / prefill_overrides) are deep-merged
    ON TOP of the global backend overrides for the active engine:
      - TRT-LLM: merged into trtllm_config sections
      - SGLang:  merged into sglang_config CLI flag dicts
    Merge priority: per-item → global → tool defaults.

    This allows different MoE backends, GEMM configs, etc. per decode mode:
        TEP groups → moe_config.backend: TRTLLM  (lower latency at small batch)
        DEP groups → moe_config.backend: CUTEDSL  (higher throughput at large batch)
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
    max_num_tokens: int | None = Field(
        default=None,
        description="Decode max_num_tokens. Defaults to batch_size if not set.",
    )
    gpu_memory_fraction: float | None = Field(
        default=None,
        description="Decode KV cache GPU memory fraction. Default depends on workload.",
    )
    eplb_num_slots: int = Field(default=0, description="Expert Load Balancer slots (0 = disabled)")

    # vLLM DP+EP mode fields (optional, only used when engine_type == "vllm")
    dp_size: int | None = Field(
        default=None,
        description=(
            "Data-parallel size for vLLM EP mode. When set, tp_size is reinterpreted "
            "as dp_size (TP=1, EP enabled, DP=dp_size). If None, defaults to tp_size."
        ),
    )
    all2all_backend: str | None = Field(
        default=None,
        description=(
            "MoE all2all communication backend for vLLM. "
            "Common values: 'deepep_low_latency' (decode), 'deepep_high_throughput' (prefill), "
            "'allgather_reducescatter'. If None, config generator picks based on role."
        ),
    )

    # Per-item engine config overrides (merged on top of global overrides)
    decode_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-group TRT-LLM decode config overrides. Deep-merged on top of "
            "backend.trtllm_decode_overrides (global). Use this to set different "
            "MoE backends, GEMM configs, etc. per decode mode/group."
        ),
    )
    prefill_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-group TRT-LLM prefill config overrides. Deep-merged on top of "
            "backend.trtllm_prefill_overrides (global). Rarely needed — most "
            "prefill configs are the same across groups."
        ),
    )

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

    Per-group overrides (decode_overrides / prefill_overrides) are injected
    into every expanded GenSweepItem. This enables different TRT-LLM backend
    settings per decode mode — e.g. TRTLLM MoE backend for TEP groups and
    CUTEDSL for DEP groups.

    Example YAML:
      tep4:
        expansion: zip
        parameters:
          concurrency: [1, 4, 16, 32, 64]
          batch_size: [1, 4, 16, 32, 64]
        defaults:
          mode: tep
          tp_size: 4
        decode_overrides:            # ← per-group override
          moe_config:
            backend: TRTLLM          # optimal for TEP (small batch)
      dep8:
        expansion: zip
        parameters:
          concurrency: [512, 1024]
          batch_size: [64, 128]
        defaults:
          mode: dep
          tp_size: 8
        decode_overrides:            # ← per-group override
          moe_config:
            backend: CUTEDSL         # optimal for DEP (large batch)
            use_low_precision_moe_combine: true
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
    decode_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-group TRT-LLM decode config overrides. Applied to every item "
            "expanded from this group, merged on top of global "
            "backend.trtllm_decode_overrides."
        ),
    )
    prefill_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-group TRT-LLM prefill config overrides. Applied to every item "
            "expanded from this group, merged on top of global "
            "backend.trtllm_prefill_overrides."
        ),
    )

    def expand(self) -> list[GenSweepItem]:
        """Expand group into concrete GenSweepItem list.

        Group-level decode_overrides / prefill_overrides are injected into
        every expanded item (unless the item itself already defines overrides
        via the parameters dict, in which case the item-level value wins).
        """
        param_names = list(self.parameters.keys())
        param_lists = list(self.parameters.values())

        if self.expansion == "zip":
            combos = list(zip(*param_lists, strict=False))
        else:  # grid
            combos = list(itertools.product(*param_lists))

        items = []
        for values in combos:
            merged = {**self.defaults, **dict(zip(param_names, values))}
            # Inject group-level overrides only if the item doesn't already
            # have its own (item-level overrides take priority).
            if self.decode_overrides and "decode_overrides" not in merged:
                merged["decode_overrides"] = self.decode_overrides
            if self.prefill_overrides and "prefill_overrides" not in merged:
                merged["prefill_overrides"] = self.prefill_overrides
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
    ttft_constraint_ms: float | None = Field(
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
    max_poll_time: int = Field(
        default=14400,
        description="Maximum seconds to poll a single job before treating it as failed (default 4h)",
    )
    e2e_validation: E2EValidationSettings = Field(
        default_factory=E2EValidationSettings,
        description="E2E validation parameters",
    )


# ---------------------------------------------------------------------------
# Backend config template
# ---------------------------------------------------------------------------

class BackendConfig(BaseModel):
    """Template for engine-specific config and environment sections.

    If not provided, generate_configs.py derives sensible defaults from
    the workload, mode, and existing recipes.

    Environment dicts are passed as env vars to the respective SLURM job.
    Override dicts are deep-merged into the generated engine config:
      - TRT-LLM: merged into trtllm_config.prefill / trtllm_config.decode
      - SGLang:  merged into sglang_config.prefill / sglang_config.decode
                 (CLI flag dicts, e.g. {"max-running-requests": 32})

    Override merge priority (highest wins):
      per-item (GenSweepItem.decode_overrides) → global (this class) → tool defaults
    """
    prefill_environment: dict[str, str] | None = None
    decode_environment: dict[str, str] | None = None
    aggregated_environment: dict[str, str] | None = None

    # TRT-LLM engine overrides
    trtllm_prefill_overrides: dict[str, Any] | None = None
    trtllm_decode_overrides: dict[str, Any] | None = None

    # SGLang engine overrides (CLI flag dicts passed to sglang_config)
    sglang_prefill_overrides: dict[str, Any] | None = None
    sglang_decode_overrides: dict[str, Any] | None = None
    sglang_aggregated_overrides: dict[str, Any] | None = None

    # vLLM engine overrides (CLI flag dicts passed to vllm_config)
    vllm_prefill_overrides: dict[str, Any] | None = None
    vllm_decode_overrides: dict[str, Any] | None = None
    connector: str = Field(
        default="nixl",
        description="KV transfer connector for vLLM disaggregated mode (nixl, kvbm).",
    )
    vllm_aggregated_log: bool = Field(
        default=True,
        description=(
            "Whether vLLM logs use AggregatedLoggingStatLogger (Running: N is "
            "global across all DP replicas). If False, Running: N is per-replica "
            "and the GEN parser divides expected concurrency by dp_size. "
            "MUST be validated on first DP+EP run."
        ),
    )
    vllm_kimi_fp4_patch: bool = Field(
        default=False,
        description=(
            "Enable monkey-patch for Kimi-K2.5-nvfp4 + vLLM 0.15.1 flashinfer "
            "tile_tokens_dim API mismatch. Requires extra_mount for patches dir."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers for base/override mode
# ---------------------------------------------------------------------------

def _gen_sweep_from_overrides(gen_sol_base: dict[str, Any], values: dict[str, Any]) -> list[dict]:
    """Auto-generate gen_sweep items from gen_sol_base override_* keys.

    When gen_sol_base is in base/override format, each variant (base + overrides)
    becomes a GenSweepItem.  Metadata is extracted from the resolved config:
      - concurrency: from benchmark.concurrencies
      - tp_size: from resources.gen_gpus_per_instance
      - mode: "tep" (default, override via gen_sol_base.base metadata if needed)
      - mtp_num: 0 (default)
      - batch_size: defaults to concurrency
    """
    from generate_configs import _deep_merge

    base_cfg = gen_sol_base["base"]
    override_keys = sorted(k for k in gen_sol_base if k.startswith("override_"))

    # Defaults from sweep-level resources
    resources = values.get("resources", {})
    tp_size = resources.get("gen_gpus_per_instance", 8)

    items: list[dict] = []
    for variant_key in ["base", *override_keys]:
        if variant_key == "base":
            resolved = dict(base_cfg)
        else:
            resolved = _deep_merge(base_cfg, gen_sol_base[variant_key])

        # Extract concurrency from benchmark.concurrencies
        conc_str = resolved.get("benchmark", {}).get("concurrencies", "1")
        concurrency = int(conc_str.split("x")[0])  # handle "8x32" → 8

        items.append({
            "mode": "tep",
            "batch_size": concurrency,
            "concurrency": concurrency,
            "tp_size": tp_size,
        })

    return items


# ---------------------------------------------------------------------------
# Top-level sweep config
# ---------------------------------------------------------------------------

class RateMatchingSweepConfig(BaseModel):
    """Top-level rate-matching sweep configuration (loaded from YAML).

    The `gen_sweep` field accepts either:
      - A flat list of GenSweepItem dicts
      - A dict of named GenSweepGroup dicts (expanded automatically)
      - Omitted when gen_sol_base is in base/override format (auto-generated)

    Optional base/override mode (Phase 1 only):
      - `ctx_sol_base`: Complete srt-slurm config dict for CTX SOL (prefill).
        Written directly as a plain YAML file.
      - `gen_sol_base`: Complete srt-slurm config in base:/override_*: format
        (decode).  Written directly as one YAML file.  Each override_* variant
        becomes a separate GEN SOL job — no gen_sweep needed.
    """
    name: str = Field(..., description="Sweep name for identification and output paths")
    engine_type: Literal["trtllm", "sglang", "vllm"] = Field(
        default="trtllm",
        description="Inference engine. Determines which log parser and config generator to use.",
    )
    model: ModelConfig = Field(..., description="Model identity")
    workload: WorkloadConfig = Field(..., description="ISL / OSL workload")
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    ctx_config: CTXConfig = Field(default_factory=CTXConfig)
    gen_sweep: list[GenSweepItem] = Field(
        default_factory=list,
        description="GEN configurations to sweep. Optional when gen_sol_base has overrides.",
    )
    settings: SweepSettings = Field(default_factory=SweepSettings)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    ctx_sol_base: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Complete srt-slurm config dict for CTX SOL (prefill). "
            "Written directly as a plain YAML file."
        ),
    )
    gen_sol_base: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Complete srt-slurm config in base:/override_*: format (decode). "
            "Written directly as one YAML file. Each override_* key becomes "
            "a separate GEN SOL job. No gen_sweep needed."
        ),
    )

    # Preserved raw gen_sweep groups (before expansion) for legacy override mode.
    raw_gen_sweep_groups: dict[str, Any] | None = Field(
        default=None, exclude=True,
        description="Internal: raw gen_sweep groups before expansion. Do not set manually.",
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_gen_sweep_groups(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Expand gen_sweep groups OR auto-generate from gen_sol_base overrides.

        Three modes:
        1. gen_sweep is a dict of groups → expand into flat list (legacy)
        2. gen_sweep is a list → pass through (legacy)
        3. gen_sol_base has 'base:' key and gen_sweep is empty/missing →
           auto-generate gen_sweep from override_* keys
        """
        gs = values.get("gen_sweep")
        gen_sol_base = values.get("gen_sol_base")

        # Mode 3: gen_sol_base in base/override format, no gen_sweep
        if gen_sol_base is not None and isinstance(gen_sol_base, dict) and "base" in gen_sol_base:
            if gs is None or (isinstance(gs, list) and len(gs) == 0):
                values["gen_sweep"] = _gen_sweep_from_overrides(gen_sol_base, values)
                return values

        if gs is None:
            return values

        # Mode 1: dict of named groups → expand
        if isinstance(gs, dict):
            values["raw_gen_sweep_groups"] = dict(gs)
            expanded: list[dict] = []
            for _group_name, group_data in gs.items():
                if isinstance(group_data, dict) and "parameters" in group_data:
                    group = GenSweepGroup(**group_data)
                    expanded.extend(item.model_dump() for item in group.expand())
                else:
                    expanded.append(group_data)
            values["gen_sweep"] = expanded

        # Mode 2: flat list → pass through (handled by Pydantic)
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
