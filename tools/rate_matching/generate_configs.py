"""
Generate srt-slurm compatible YAML configurations for rate-matching sweeps.

Produces three types of config:
  1. CTX-only SOL  -- measures prefill throughput in isolation
  2. GEN-only SOL  -- measures decode throughput per concurrency/mode/MTP
  3. E2E validation -- full disaggregated serving with rate-matched allocation

The output YAMLs are directly consumable by `srtctl apply -f <config.yaml>`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from schema import GenSweepItem, RateMatchingSweepConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PREFILL_ENV = {
    "UCX_TLS": "rc,dc,ud,cuda_copy,cuda_ipc,gdr_copy,tcp",
    "TRTLLM_ENABLE_PDL": "1",
    "TRTLLM_SERVER_DISABLE_GC": "1",
    "TRTLLM_WORKER_DISABLE_GC": "1",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
}

_DEFAULT_DECODE_ENV = {
    "UCX_TLS": "rc,dc,ud,cuda_copy,cuda_ipc,gdr_copy,tcp",
    "TRTLLM_ENABLE_PDL": "1",
    "TRTLLM_SERVER_DISABLE_GC": "1",
    "TRTLLM_WORKER_DISABLE_GC": "1",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
}

# Extra env vars for GEN-only SOL isolation
_GEN_SOL_DECODE_ENV_EXTRAS = {
    "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1",
}


def _cuda_graph_batch_sizes(max_bs: int) -> list[int]:
    """Generate cuda_graph batch_sizes list for a given max_batch_size.

    Uses powers of 2 up to 16, then steps of 8 up to max_bs.
    """
    sizes = []
    # Powers of 2 up to min(16, max_bs)
    v = 1
    while v <= min(16, max_bs):
        sizes.append(v)
        v *= 2
    # Then multiples of 8 from 24 up to max_bs
    v = 24
    while v <= max_bs:
        if v not in sizes:
            sizes.append(v)
        v += 8
    # Ensure max_bs itself is included
    if max_bs not in sizes:
        sizes.append(max_bs)
    return sorted(sizes)


def _workload_params(cfg: RateMatchingSweepConfig) -> dict:
    """Derive workload-dependent parameters."""
    isl = cfg.workload.isl
    osl = cfg.workload.osl
    is_short_ctx = isl <= 2048  # 1k workload vs 8k+

    if is_short_ctx:
        prefill_max_batch_size = 8
        prefill_max_num_tokens = 8192
        prefill_max_seq_len = isl + 40
        prefill_gpu_mem_frac = 0.6
        cache_tokens_prefill = 8192
        cache_tokens_decode = 8192
        decode_max_seq_len = isl + osl + 40
        # 1k1k: prefill uses enable_attention_dp=true
        prefill_attention_dp = True
        prefill_stream_interval = 100
    else:
        prefill_max_batch_size = 2
        prefill_max_num_tokens = 16896
        prefill_max_seq_len = isl + 40
        prefill_gpu_mem_frac = 0.85
        cache_tokens_prefill = 32768
        cache_tokens_decode = 16384
        decode_max_seq_len = isl + osl + 40
        prefill_attention_dp = False
        prefill_stream_interval = 1

    return {
        "prefill_max_batch_size": prefill_max_batch_size,
        "prefill_max_num_tokens": prefill_max_num_tokens,
        "prefill_max_seq_len": prefill_max_seq_len,
        "prefill_gpu_mem_frac": prefill_gpu_mem_frac,
        "cache_tokens_prefill": cache_tokens_prefill,
        "cache_tokens_decode": cache_tokens_decode,
        "decode_max_seq_len": decode_max_seq_len,
        "prefill_attention_dp": prefill_attention_dp,
        "prefill_stream_interval": prefill_stream_interval,
    }


def _prefill_config(cfg: RateMatchingSweepConfig, wp: dict, mtp_num: int = 0) -> dict:
    """Build the prefill trtllm_config section."""
    tp = cfg.resources.ctx_gpus_per_instance
    config = {
        "backend": "pytorch",
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "moe_expert_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "enable_attention_dp": wp["prefill_attention_dp"],
        "enable_chunked_prefill": False,
        "max_batch_size": cfg.ctx_config.max_batch_size or wp["prefill_max_batch_size"],
        "max_num_tokens": cfg.ctx_config.max_num_tokens or wp["prefill_max_num_tokens"],
        "max_seq_len": wp["prefill_max_seq_len"],
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": cfg.ctx_config.free_gpu_memory_fraction or wp["prefill_gpu_mem_frac"],
            "dtype": "fp8",
        },
        "cache_transceiver_config": {
            "backend": "UCX",
            "max_tokens_in_buffer": wp["cache_tokens_prefill"],
        },
        "moe_config": {"backend": "CUTLASS"},
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "print_iter_log": True,
        "stream_interval": wp["prefill_stream_interval"],
        "num_postprocess_workers": 4,
    }
    if mtp_num > 0:
        config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp_num,
        }
    return config


def _decode_config(
    cfg: RateMatchingSweepConfig,
    wp: dict,
    gen_item: GenSweepItem,
    *,
    for_gen_sol: bool = False,
) -> dict:
    """Build the decode trtllm_config section."""
    tp = gen_item.tp_size
    batch_size = gen_item.batch_size
    max_num_tokens = gen_item.max_num_tokens or batch_size
    is_dep = gen_item.mode == "dep"
    gpu_mem_frac = gen_item.gpu_memory_fraction or (0.85 if cfg.workload.isl > 2048 else 0.9)

    config = {
        "backend": "pytorch",
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "moe_expert_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "enable_attention_dp": is_dep,
        "enable_chunked_prefill": False,
        "max_batch_size": batch_size,
        "max_num_tokens": max_num_tokens,
        "max_seq_len": wp["decode_max_seq_len"],
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": gpu_mem_frac,
            "dtype": "fp8",
        },
        "cache_transceiver_config": {
            "backend": "UCX",
            "max_tokens_in_buffer": wp["cache_tokens_decode"],
        },
        "moe_config": {
            "backend": "CUTLASS",
            "use_low_precision_moe_combine": True,
        },
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": _cuda_graph_batch_sizes(batch_size),
        },
        "disable_overlap_scheduler": False,
        "print_iter_log": True,
        "stream_interval": 100,
        "num_postprocess_workers": 4,
    }

    # DEP-specific fields
    if is_dep:
        config["enable_lm_head_tp_in_adp"] = True
    else:
        config["enable_lm_head_tp_in_adp"] = False

    # MTP
    if gen_item.mtp_num > 0:
        config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": gen_item.mtp_num,
        }

    return config


def _format_concurrency(conc) -> str:
    """Format concurrency for sa-bench (x-separated string)."""
    if isinstance(conc, list):
        return "x".join(str(c) for c in conc)
    return str(conc)


# ---------------------------------------------------------------------------
# CTX-only SOL config
# ---------------------------------------------------------------------------

def generate_ctx_sol_config(
    cfg: RateMatchingSweepConfig,
    output_path: str | None = None,
) -> dict:
    """Generate CTX-only SOL benchmark config.

    Runs with ISL/<osl=1> to measure pure prefill throughput.
    Single prefill node, single decode node (needed for disagg setup).
    """
    wp = _workload_params(cfg)
    prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_PREFILL_ENV)
    decode_env = cfg.backend.decode_environment or dict(_DEFAULT_DECODE_ENV)

    config = {
        "name": f"ctx_sol_{cfg.workload.isl // 1000}k{cfg.workload.osl // 1000}k",
        "model": {
            "path": cfg.model.path,
            "container": cfg.model.container,
            "precision": cfg.model.precision,
        },
        "sbatch_directives": {"cpus-per-gpu": "16"},
        "resources": {
            "gpu_type": cfg.resources.gpu_type,
            "prefill_nodes": 1,
            "prefill_workers": 1,
            "decode_nodes": 1,
            "decode_workers": 1,
            "gpus_per_node": cfg.resources.gpus_per_node,
        },
        "backend": {
            "type": cfg.engine_type,
            "prefill_environment": prefill_env,
            "decode_environment": decode_env,
            "trtllm_config": {
                "prefill": _prefill_config(cfg, wp),
                "decode": {
                    "backend": "pytorch",
                    "trust_remote_code": True,
                    "tensor_parallel_size": cfg.resources.gen_gpus_per_instance,
                    "moe_expert_parallel_size": cfg.resources.gen_gpus_per_instance,
                    "pipeline_parallel_size": 1,
                    "enable_attention_dp": False,
                    "enable_chunked_prefill": False,
                    "max_batch_size": 32,
                    "max_num_tokens": 32,
                    "max_seq_len": wp["decode_max_seq_len"],
                    "kv_cache_config": {
                        "enable_block_reuse": False,
                        "free_gpu_memory_fraction": 0.85,
                        "dtype": "fp8",
                    },
                    "cache_transceiver_config": {
                        "backend": "UCX",
                        "max_tokens_in_buffer": wp["cache_tokens_decode"],
                    },
                    "moe_config": {
                        "backend": "CUTLASS",
                        "use_low_precision_moe_combine": True,
                    },
                    "cuda_graph_config": {
                        "enable_padding": True,
                        "batch_sizes": _cuda_graph_batch_sizes(32),
                    },
                    "disable_overlap_scheduler": False,
                    "print_iter_log": True,
                    "stream_interval": 100,
                    "num_postprocess_workers": 4,
                },
            },
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": cfg.workload.isl,
            "osl": 1,  # CTX-only: minimal output
            "concurrencies": str(cfg.ctx_config.benchmark_concurrency),
            "req_rate": "inf",
            "random_range_ratio": 1.0,  # CTX-only: use exact ISL, no variance
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
        },
        "dynamo": {"install": False},
    }

    if output_path:
        _write_yaml(config, output_path)
    return config


# ---------------------------------------------------------------------------
# GEN-only SOL config
# ---------------------------------------------------------------------------

def generate_gen_sol_config(
    cfg: RateMatchingSweepConfig,
    gen_item: GenSweepItem,
    output_path: str | None = None,
) -> dict:
    """Generate GEN-only SOL benchmark config.

    Runs on a single decode node to measure per-worker decode throughput.
    Uses skip_initial_test and GEN isolation env vars.
    """
    wp = _workload_params(cfg)
    prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_PREFILL_ENV)

    # GEN-only: add isolation env vars to decode
    decode_env = cfg.backend.decode_environment or dict(_DEFAULT_DECODE_ENV)
    decode_env = {**decode_env, **_GEN_SOL_DECODE_ENV_EXTRAS}
    # CRITICAL: TLLM_BENCHMARK_REQ_QUEUES_SIZE must equal the concurrency.
    # This env var sets the minimum number of requests the decode worker
    # queues before processing begins. It is read once at worker startup
    # and cannot change without restarting the model. A mismatch between
    # queue depth and concurrency produces incorrect prev_device_step_time
    # values, which directly corrupt the SOL throughput calculation.
    # The pipeline enforces one job per concurrency to guarantee this.
    conc = gen_item.concurrency if isinstance(gen_item.concurrency, int) else gen_item.concurrency[0]
    decode_env["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] = str(conc)

    mtp_suffix = f"_mtp{gen_item.mtp_num}" if gen_item.mtp_num > 0 else ""
    conc_str = _format_concurrency(gen_item.concurrency)
    name = f"gen_sol_{cfg.workload.isl // 1000}k{cfg.workload.osl // 1000}k_{gen_item.mode}_c{conc_str}{mtp_suffix}"

    config = {
        "name": name,
        "model": {
            "path": cfg.model.path,
            "container": cfg.model.container,
            "precision": cfg.model.precision,
        },
        "sbatch_directives": {"cpus-per-gpu": "16"},
        "resources": {
            "gpu_type": cfg.resources.gpu_type,
            "prefill_nodes": 1,
            "prefill_workers": 1,
            "decode_nodes": 1,
            "decode_workers": 1,
            "gpus_per_node": cfg.resources.gpus_per_node,
        },
        "backend": {
            "type": cfg.engine_type,
            "prefill_environment": prefill_env,
            "decode_environment": decode_env,
            "trtllm_config": {
                "prefill": _prefill_config(cfg, wp, mtp_num=gen_item.mtp_num),
                "decode": _decode_config(cfg, wp, gen_item, for_gen_sol=True),
            },
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": cfg.workload.isl,
            "osl": cfg.workload.osl,
            "concurrencies": _format_concurrency(gen_item.concurrency),
            "req_rate": "inf",
            "skip_initial_test": True,
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
        },
        "dynamo": {"install": False},
    }

    if output_path:
        _write_yaml(config, output_path)
    return config


# ---------------------------------------------------------------------------
# E2E validation config
# ---------------------------------------------------------------------------

def get_recipe_filename(
    per_worker_conc: int,
    ctx_instances: int,
    gen_instances: int,
    mode: str,
    tp_size: int,
    batch_size: int,
    mtp_num: int = 0,
    eplb_num_slots: int = 0,
    multiplier: float = 1.0,
) -> str:
    """Generate recipe-convention filename.

    Format: c{pw_conc}_ctx{N}_gen{M}_{mode}{tp}_batch{B}_eplb{E}_mtp{M}[_1.05x]
    """
    name = (
        f"c{per_worker_conc}_ctx{ctx_instances}_gen{gen_instances}"
        f"_{mode}{tp_size}_batch{batch_size}"
        f"_eplb{eplb_num_slots}_mtp{mtp_num}"
    )
    if multiplier != 1.0:
        name += f"_{multiplier:.2f}x"
    return name


def generate_e2e_config(
    cfg: RateMatchingSweepConfig,
    pareto_point: dict,
    concurrency_multiplier: float = 1.0,
    output_path: str | None = None,
) -> dict:
    """Generate E2E validation config from a Pareto optimal point.

    CRITICAL: E2E concurrency scaling.
      SOL measures at per-worker concurrency C on 1 decode worker.
      E2E with N decode workers needs system_concurrency = C * N * multiplier
      to maintain the same per-worker load.

    Args:
        cfg: Sweep configuration
        pareto_point: Dict from pareto_frontier with allocation info
        concurrency_multiplier: Applied to system_conc (1.0 = baseline, 1.05 = headroom)
        output_path: Optional path to write YAML
    """
    wp = _workload_params(cfg)
    prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_PREFILL_ENV)
    decode_env = cfg.backend.decode_environment or dict(_DEFAULT_DECODE_ENV)

    # Extract allocation from Pareto point
    ctx_instances = pareto_point["ctx_instances"]
    gen_instances = pareto_point["gen_instances"]
    per_worker_conc = pareto_point["concurrency"]
    batch_size = pareto_point["batch_size"]
    mode = pareto_point["mode"]
    tp_size = pareto_point.get("tp_size", cfg.resources.gen_gpus_per_instance)
    mtp_num = pareto_point.get("mtp_num", 0)
    eplb_num_slots = pareto_point.get("eplb_num_slots", 0)
    max_num_tokens = pareto_point.get("max_num_tokens")
    gpu_memory_fraction = pareto_point.get("gpu_memory_fraction")

    # System concurrency = per_worker_conc * gen_instances * multiplier
    system_concurrency = int(per_worker_conc * gen_instances * concurrency_multiplier)

    recipe_name = get_recipe_filename(
        per_worker_conc, ctx_instances, gen_instances,
        mode, tp_size, batch_size, mtp_num, eplb_num_slots,
        multiplier=concurrency_multiplier,
    )

    # Build a synthetic GenSweepItem for decode config generation
    gen_item = GenSweepItem(
        mode=mode,
        batch_size=batch_size,
        concurrency=per_worker_conc,
        tp_size=tp_size,
        mtp_num=mtp_num,
        max_num_tokens=max_num_tokens,
        gpu_memory_fraction=gpu_memory_fraction,
        eplb_num_slots=eplb_num_slots,
    )

    ratio_str = pareto_point.get("ratio_str", f"{ctx_instances}:{gen_instances}")
    sol_tpot = pareto_point.get("tpot_ms", 0)
    sol_inter = pareto_point.get("interactivity", 0)
    sol_tput = pareto_point.get("output_tput_per_gpu", 0)
    pareto_rank = pareto_point.get("pareto_rank", 0)

    # Header comment
    mtp_label = f"MTP-{mtp_num}" if mtp_num > 0 else "STP"
    mult_label = f" ({concurrency_multiplier:.2f}x headroom)" if concurrency_multiplier != 1.0 else ""
    header = (
        f"# E2E Config: {cfg.workload.isl // 1000}k/{cfg.workload.osl // 1000}k "
        f"{mode.upper()} {mtp_label} on {cfg.resources.gpu_type.upper()}\n"
        f"# Rate-matched: {ratio_str} "
        f"({ctx_instances * cfg.resources.gpus_per_node} CTX + "
        f"{gen_instances * cfg.resources.gpus_per_node} GEN = "
        f"{(ctx_instances + gen_instances) * cfg.resources.gpus_per_node} GPUs)\n"
        f"# Per-worker concurrency: {per_worker_conc}  (invariant from SOL)\n"
        f"# System concurrency:     {system_concurrency}  "
        f"(= {per_worker_conc} x {gen_instances} x {concurrency_multiplier}){mult_label}\n"
        f"#\n"
        f"# SOL Predictions:\n"
        f"#   Interactivity: {sol_inter:.2f}\n"
        f"#   TPOT: {sol_tpot:.2f} ms\n"
        f"#   Throughput/GPU: {sol_tput:.2f} tok/s/GPU\n"
        f"#\n"
        f"# Pareto rank: {pareto_rank}\n"
        f"# Run with: srtctl apply -f {recipe_name}.yaml\n"
    )

    config = {
        "name": recipe_name,
        "model": {
            "path": cfg.model.path,
            "container": cfg.model.container,
            "precision": cfg.model.precision,
        },
        "sbatch_directives": {"cpus-per-gpu": "16"},
        "resources": {
            "gpu_type": cfg.resources.gpu_type,
            "prefill_nodes": ctx_instances,
            "prefill_workers": ctx_instances,
            "decode_nodes": gen_instances,
            "decode_workers": gen_instances,
            "gpus_per_node": cfg.resources.gpus_per_node,
        },
        "backend": {
            "type": cfg.engine_type,
            "prefill_environment": prefill_env,
            "decode_environment": decode_env,
            "trtllm_config": {
                "prefill": _prefill_config(cfg, wp, mtp_num=mtp_num),
                "decode": _decode_config(cfg, wp, gen_item, for_gen_sol=False),
            },
        },
        "benchmark": {
            "type": "sa-bench",
            "isl": cfg.workload.isl,
            "osl": cfg.workload.osl,
            "concurrencies": str(system_concurrency),
            "req_rate": "inf",
        },
        "frontend": {
            "type": "dynamo",
            "enable_multiple_frontends": False,
        },
        "dynamo": {"install": False},
    }

    if output_path:
        _write_yaml(config, output_path, header=header)
    return config


def generate_e2e_configs_from_pareto(
    cfg: RateMatchingSweepConfig,
    pareto_frontier: list[dict],
    output_dir: str,
) -> list[dict]:
    """Generate E2E configs for every Pareto point x concurrency multiplier.

    Returns list of dicts: [{config_path, pareto_rank, multiplier, system_concurrency}, ...]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    multipliers = cfg.settings.e2e_validation.concurrency_multipliers
    results = []

    for pp in pareto_frontier:
        for mult in multipliers:
            recipe_name = get_recipe_filename(
                pp["concurrency"], pp["ctx_instances"], pp["gen_instances"],
                pp["mode"], pp.get("tp_size", cfg.resources.gen_gpus_per_instance),
                pp["batch_size"], pp.get("mtp_num", 0),
                pp.get("eplb_num_slots", 0), multiplier=mult,
            )
            config_path = str(out / f"{recipe_name}.yaml")
            config = generate_e2e_config(cfg, pp, concurrency_multiplier=mult, output_path=config_path)

            gen_instances = pp["gen_instances"]
            pw_conc = pp["concurrency"]
            sys_conc = int(pw_conc * gen_instances * mult)

            results.append({
                "config_path": config_path,
                "pareto_rank": pp.get("pareto_rank", 0),
                "multiplier": mult,
                "per_worker_concurrency": pw_conc,
                "system_concurrency": sys_conc,
                "config_name": recipe_name,
            })

    return results


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def _write_yaml(config: dict, path: str, header: str = "") -> None:
    """Write config dict to YAML file with optional header comment."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(header + "\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
