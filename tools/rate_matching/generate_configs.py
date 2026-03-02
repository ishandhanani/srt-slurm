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

# SGLang default environment
_DEFAULT_SGLANG_ENV = {
    "PYTHONUNBUFFERED": "1",
    "NCCL_NVLS_ENABLE": "1",
}

# SGLang disaggregated environment (extends default with NIXL/bootstrap timeouts)
_DEFAULT_SGLANG_DISAGG_ENV = {
    **_DEFAULT_SGLANG_ENV,
    "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE": "100000",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "100000",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "100000",
    "SGLANG_RECORD_STEP_TIME": "1",
}


# ---------------------------------------------------------------------------
# SGLang config helpers
# ---------------------------------------------------------------------------

def _sglang_agg_config(
    cfg: RateMatchingSweepConfig,
    *,
    max_running_requests: int | None = None,
    mtp_num: int = 0,
    decode_log_interval: int = 1,
    item_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SGLang aggregated-mode sglang_config dict.

    Returns CLI-flag-style dict consumed by SGLangServerConfig.aggregated.
    Used for both CTX SOL and GEN SOL in aggregated mode.

    Merge priority (highest wins):
      item_overrides → cfg.backend.sglang_aggregated_overrides → defaults
    """
    tp = cfg.resources.ctx_gpus_per_instance
    config: dict[str, Any] = {
        "trust-remote-code": True,
        "tp-size": tp,
        "disable-radix-cache": True,
        "mem-fraction-static": 0.82,
        "chunked-prefill-size": 32768,
        "max-prefill-tokens": 32768,
        "decode-log-interval": decode_log_interval,
        "stream-interval": 30,
        "enable-flashinfer-allreduce-fusion": True,
    }

    if max_running_requests is not None:
        config["max-running-requests"] = max_running_requests

    if mtp_num > 0:
        config["speculative-algorithm"] = "NEXTN"
        config["speculative-num-steps"] = mtp_num
        config["speculative-eagle-topk"] = 1

    if cfg.backend.sglang_aggregated_overrides:
        config = _deep_merge(config, cfg.backend.sglang_aggregated_overrides)

    if item_overrides:
        config = _deep_merge(config, item_overrides)

    return config


def _sglang_prefill_config(
    cfg: RateMatchingSweepConfig,
    *,
    mtp_num: int = 0,
    item_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SGLang prefill-mode sglang_config dict (for disaggregated).

    NOTE: ``disaggregation-mode`` and ``disaggregation-bootstrap-port`` are
    injected automatically by srt-slurm's backend code (sglang.py).  Only
    ``disaggregation-transfer-backend`` needs to be set here.
    """
    tp = cfg.resources.ctx_gpus_per_instance
    config: dict[str, Any] = {
        "trust-remote-code": True,
        "tp-size": tp,
        "disable-radix-cache": True,
        "mem-fraction-static": 0.82,
        "chunked-prefill-size": 32768,
        "max-prefill-tokens": 32768,
        "max-running-requests": 256,
        "decode-log-interval": 1,
        "stream-interval": 30,
        "enable-flashinfer-allreduce-fusion": True,
        "watchdog-timeout": 1000000,
        "disaggregation-transfer-backend": "nixl",
    }

    if mtp_num > 0:
        config["speculative-algorithm"] = "NEXTN"
        config["speculative-num-steps"] = mtp_num
        config["speculative-eagle-topk"] = 1

    if cfg.backend.sglang_prefill_overrides:
        config = _deep_merge(config, cfg.backend.sglang_prefill_overrides)

    if item_overrides:
        config = _deep_merge(config, item_overrides)

    return config


def _sglang_decode_config(
    cfg: RateMatchingSweepConfig,
    gen_item: GenSweepItem,
    *,
    item_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SGLang decode-mode sglang_config dict (for disaggregated).

    NOTE: ``disaggregation-mode`` and ``disaggregation-bootstrap-port`` are
    injected automatically by srt-slurm's backend code (sglang.py).  Only
    ``disaggregation-transfer-backend`` needs to be set here.
    """
    conc = gen_item.concurrency if isinstance(gen_item.concurrency, int) else gen_item.concurrency[0]
    config: dict[str, Any] = {
        "trust-remote-code": True,
        "tp-size": gen_item.tp_size,
        "disable-radix-cache": True,
        "mem-fraction-static": 0.82,
        "max-running-requests": conc,
        "decode-log-interval": 1,
        "cuda-graph-max-bs": max(128, conc),
        "stream-interval": 30,
        "enable-flashinfer-allreduce-fusion": True,
        "scheduler-recv-interval": 30,
        "watchdog-timeout": 1000000,
        "disaggregation-transfer-backend": "nixl",
    }

    if gen_item.mtp_num > 0:
        config["speculative-algorithm"] = "NEXTN"
        config["speculative-num-steps"] = gen_item.mtp_num
        config["speculative-eagle-topk"] = 1

    if cfg.backend.sglang_decode_overrides:
        config = _deep_merge(config, cfg.backend.sglang_decode_overrides)

    if item_overrides:
        config = _deep_merge(config, item_overrides)

    return config


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


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base dict (overrides win at leaf level)."""
    result = dict(base)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _prefill_config(
    cfg: RateMatchingSweepConfig,
    wp: dict,
    mtp_num: int = 0,
    *,
    item_prefill_overrides: dict | None = None,
) -> dict:
    """Build the prefill trtllm_config section.

    Merge priority (highest wins):
      1. item_prefill_overrides  (per-group / per-item from gen_sweep)
      2. cfg.backend.trtllm_prefill_overrides  (global from sweep YAML)
      3. Tool defaults (this function's base config)
    """
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

    # Layer 1: global overrides from sweep YAML backend.trtllm_prefill_overrides
    if cfg.backend.trtllm_prefill_overrides:
        config = _deep_merge(config, cfg.backend.trtllm_prefill_overrides)

    # Layer 2: per-item / per-group overrides (highest priority)
    if item_prefill_overrides:
        config = _deep_merge(config, item_prefill_overrides)

    return config


def _decode_config(
    cfg: RateMatchingSweepConfig,
    wp: dict,
    gen_item: GenSweepItem,
    *,
    for_gen_sol: bool = False,
) -> dict:
    """Build the decode trtllm_config section.

    Merge priority (highest wins):
      1. gen_item.decode_overrides  (per-group / per-item from gen_sweep)
      2. cfg.backend.trtllm_decode_overrides  (global from sweep YAML)
      3. Tool defaults (this function's base config)

    This allows e.g. TEP groups to use moe_config.backend=TRTLLM while
    DEP groups use CUTEDSL, without duplicating the entire config.
    """
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

    # Layer 1: global overrides from sweep YAML backend.trtllm_decode_overrides
    if cfg.backend.trtllm_decode_overrides:
        config = _deep_merge(config, cfg.backend.trtllm_decode_overrides)

    # Layer 2: per-item / per-group overrides (highest priority)
    if gen_item.decode_overrides:
        config = _deep_merge(config, gen_item.decode_overrides)

    return config


def _nodes_for_tp(tp_size: int, gpus_per_node: int) -> int:
    """Compute number of nodes needed for a given TP size."""
    return math.ceil(tp_size / gpus_per_node)


def _format_concurrency(conc) -> str:
    """Format concurrency for sa-bench (x-separated string)."""
    if isinstance(conc, list):
        return "x".join(str(c) for c in conc)
    return str(conc)


def _ctx_sol_decode_stub(cfg: RateMatchingSweepConfig, wp: dict) -> dict:
    """Build a minimal decode config for the CTX SOL benchmark.

    The decode worker is a stub (CTX SOL uses osl=1, so decode barely runs),
    but it still needs to load the model.  Apply trtllm_decode_overrides so
    FP4-critical settings like nvfp4_gemm_config are present.
    """
    stub = {
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
    }
    # Apply user overrides (e.g. nvfp4_gemm_config, moe_config.backend)
    if cfg.backend.trtllm_decode_overrides:
        stub = _deep_merge(stub, cfg.backend.trtllm_decode_overrides)
    return stub


# ---------------------------------------------------------------------------
# vLLM config helpers (DP+EP mode with NixlConnector PD disagg)
#
# Based on:
#   - PR #33844: PD disagg config for Kimi-K2/DeepSeek-R1 on GB200
#   - recipes/vllm/deepseek-r1/disagg-h100-16gpu.yaml
#   - src/srtctl/backends/vllm.py (srt-slurm vLLM backend)
# ---------------------------------------------------------------------------

_DEFAULT_VLLM_BASE_ENV = {
    "VLLM_USE_DEEP_GEMM": "1",
    "VLLM_SKIP_P2P_CHECK": "1",
    "VLLM_RANDOMIZE_DP_DUMMY_INPUTS": "1",
    "NVIDIA_GDRCOPY": "enabled",
    "PYTHONUNBUFFERED": "1",
    "VLLM_LOG_STATS_INTERVAL": "1",
}

_VLLM_GEN_SOL_EXTRAS: dict[str, str] = {}

_VLLM_DECODE_EXTRAS = {
    "VLLM_MOE_DP_CHUNK_SIZE": "384",
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "8192",
}

_VLLM_DISAGG_ENV = {
    "NVIDIA_GDRCOPY": "1",
    "NVSHMEM_IB_ENABLE_IBGDA": "1",
    "VLLM_SKIP_P2P_CHECK": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_MNNVL_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_TIMEOUT": "1800",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
    "VLLM_USE_FLASHINFER_MOE_FP4": "1",
    "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL": "0",
    "VLLM_USE_NCCL_SYMM_MEM": "1",
    "UCX_IB_ROCE_REACHABILITY_MODE": "local_subnet",
    "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT": "300",
}


def _vllm_prefill_config(
    cfg: RateMatchingSweepConfig,
    wp: dict,
    *,
    item_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build vLLM prefill vllm_config section (CLI flag dict).

    Merge priority (highest wins):
      item_overrides → cfg.backend.vllm_prefill_overrides → defaults
    """
    dp_size = cfg.resources.ctx_gpus_per_instance
    config: dict[str, Any] = {
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-expert-parallel": True,
        "data-parallel-size": dp_size,
        "data-parallel-rpc-port": 13345,
        "data-parallel-hybrid-lb": True,
        "max-model-len": wp["prefill_max_seq_len"],
        "max-num-seqs": cfg.ctx_config.max_batch_size or 8,
        "enforce-eager": True,
        "gpu-memory-utilization": cfg.ctx_config.free_gpu_memory_fraction or 0.9,
        "max-num-batched-tokens": cfg.ctx_config.max_num_tokens or 16384,
        "no-enable-chunked-prefill": True,
        "swap-space": 16,
        "kv-cache-dtype": "fp8",
        "all2all-backend": "deepep_high_throughput",
        "async-scheduling": True,
        "enable-dbo": True,
        "dbo-decode-token-threshold": 32,
        "no-enable-prefix-caching": True,
        "trust-remote-code": True,
    }

    if cfg.backend.vllm_prefill_overrides:
        config = _deep_merge(config, cfg.backend.vllm_prefill_overrides)

    if item_overrides:
        config = _deep_merge(config, item_overrides)

    return config


def _vllm_decode_config(
    cfg: RateMatchingSweepConfig,
    wp: dict,
    gen_item: GenSweepItem,
    *,
    for_gen_sol: bool = False,
) -> dict[str, Any]:
    """Build vLLM decode vllm_config section (CLI flag dict).

    Merge priority (highest wins):
      gen_item.decode_overrides → cfg.backend.vllm_decode_overrides → defaults
    """
    dp_size = gen_item.dp_size or gen_item.tp_size
    batch_size = gen_item.batch_size
    max_num_tokens = gen_item.max_num_tokens or 4096
    gpu_mem = gen_item.gpu_memory_fraction or 0.9

    compilation_cfg = (
        '{"cudagraph_mode":"FULL_DECODE_ONLY",'
        '"custom_ops":["+rms_norm"],'
        '"pass_config":{"enable_fusion":true,"enable_attn_fusion":true,"enable_noop":true}}'
    )

    eplb_cfg = (
        '{"window_size":"1000","step_interval":"3000",'
        '"num_redundant_experts":"32","log_balancedness":"False"}'
    )

    config: dict[str, Any] = {
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-expert-parallel": True,
        "data-parallel-size": dp_size,
        "data-parallel-rpc-port": 13345,
        "data-parallel-hybrid-lb": True,
        "max-model-len": wp["decode_max_seq_len"],
        "max-num-seqs": batch_size,
        "gpu-memory-utilization": gpu_mem,
        "max-num-batched-tokens": max_num_tokens,
        "max-cudagraph-capture-size": batch_size,
        "compilation-config": compilation_cfg,
        "kv-cache-dtype": "fp8",
        "all2all-backend": gen_item.all2all_backend or "deepep_low_latency",
        "async-scheduling": True,
        "stream-interval": 50,
        "enable-dbo": True,
        "dbo-decode-token-threshold": 32,
        "no-enable-prefix-caching": True,
        "enable-eplb": True,
        "eplb-config": eplb_cfg,
        "trust-remote-code": True,
    }

    if cfg.backend.vllm_decode_overrides:
        config = _deep_merge(config, cfg.backend.vllm_decode_overrides)

    if gen_item.decode_overrides:
        config = _deep_merge(config, gen_item.decode_overrides)

    return config


def _vllm_ctx_sol_decode_stub(
    cfg: RateMatchingSweepConfig,
    wp: dict,
) -> dict[str, Any]:
    """Minimal vLLM decode config for the CTX SOL benchmark.

    The decode worker barely runs during CTX SOL (osl=1) but still needs
    to load the model.
    """
    config: dict[str, Any] = {
        "tensor-parallel-size": 1,
        "pipeline-parallel-size": 1,
        "enable-expert-parallel": True,
        "data-parallel-size": cfg.resources.gen_gpus_per_instance,
        "data-parallel-rpc-port": 13345,
        "data-parallel-hybrid-lb": True,
        "max-model-len": wp["decode_max_seq_len"],
        "max-num-seqs": 32,
        "gpu-memory-utilization": 0.85,
        "max-num-batched-tokens": 4096,
        "kv-cache-dtype": "fp8",
        "all2all-backend": "deepep_low_latency",
        "async-scheduling": True,
        "trust-remote-code": True,
        "no-enable-prefix-caching": True,
    }

    if cfg.backend.vllm_decode_overrides:
        config = _deep_merge(config, cfg.backend.vllm_decode_overrides)

    return config


def _vllm_env(
    cfg: RateMatchingSweepConfig,
    role: str,
    *,
    for_gen_sol: bool = False,
) -> dict[str, str]:
    """Build vLLM environment dict for prefill or decode."""
    env = dict(_DEFAULT_VLLM_BASE_ENV)
    env.update(_VLLM_DISAGG_ENV)

    if role == "decode":
        env.update(_VLLM_DECODE_EXTRAS)
        if for_gen_sol:
            env.update(_VLLM_GEN_SOL_EXTRAS)

    user_env = None
    if role == "prefill":
        user_env = cfg.backend.prefill_environment
    elif role == "decode":
        user_env = cfg.backend.decode_environment

    if user_env:
        env.update(user_env)

    return env


# ---------------------------------------------------------------------------
# CTX-only SOL config
# ---------------------------------------------------------------------------

def generate_ctx_sol_config(
    cfg: RateMatchingSweepConfig,
    output_path: str | None = None,
) -> dict:
    """Generate CTX-only SOL benchmark config.

    Runs with ISL/<osl=1> to measure pure prefill throughput.

    TRT-LLM: disaggregated mode (single prefill + decode stub).
    SGLang:  aggregated mode with osl=1 (decode is minimal).  Aggregated is
             intentional for CTX SOL — no KV transfer needed for single-worker
             prefill measurement.
    """
    wp = _workload_params(cfg)

    config_name = f"{cfg.name}_ctx_sol"

    if cfg.engine_type == "sglang":
        agg_env = cfg.backend.aggregated_environment or dict(_DEFAULT_SGLANG_ENV)
        config = {
            "name": config_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "agg_nodes": 1,
                "agg_workers": 1,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": "sglang",
                "aggregated_environment": agg_env,
                "sglang_config": {
                    "aggregated": _sglang_agg_config(
                        cfg,
                        max_running_requests=cfg.ctx_config.benchmark_concurrency,
                    ),
                },
            },
            "benchmark": {
                "type": "sa-bench",
                "isl": cfg.workload.isl,
                "osl": 1,
                "concurrencies": str(cfg.ctx_config.benchmark_concurrency),
                "req_rate": "inf",
            },
        }
    elif cfg.engine_type == "vllm":
        prefill_env = _vllm_env(cfg, "prefill")
        decode_env = _vllm_env(cfg, "decode")
        config = {
            "name": config_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": 1,
                "prefill_workers": 1,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": 1,
                "decode_workers": 1,
                "gpus_per_decode": cfg.resources.gen_gpus_per_instance,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": "vllm",
                "connector": cfg.backend.connector,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "vllm_config": {
                    "prefill": _vllm_prefill_config(cfg, wp),
                    "decode": _vllm_ctx_sol_decode_stub(cfg, wp),
                },
                **({"vllm_kimi_fp4_patch": True} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
            },
            "benchmark": {
                "type": "sa-bench",
                "isl": cfg.workload.isl,
                "osl": 1,
                "concurrencies": str(cfg.ctx_config.benchmark_concurrency),
                "req_rate": "inf",
            },
            "frontend": {
                "type": "dynamo",
                "enable_multiple_frontends": False,
            },
            "dynamo": {"install": False},
            **({"extra_mount": ["${SRTCTL_SOURCE_DIR}/tools/rate_matching/patches:/patches"]} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
        }
    else:
        # TRT-LLM path (original logic)
        prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_PREFILL_ENV)
        decode_env = cfg.backend.decode_environment or dict(_DEFAULT_DECODE_ENV)
        ctx_decode_nodes = _nodes_for_tp(cfg.resources.gen_gpus_per_instance, cfg.resources.gpus_per_node)

        config = {
            "name": config_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "sbatch_directives": {},
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": 1,
                "prefill_workers": 1,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": ctx_decode_nodes,
                "decode_workers": 1,
                "gpus_per_decode": cfg.resources.gen_gpus_per_instance,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": cfg.engine_type,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "trtllm_config": {
                    "prefill": _prefill_config(cfg, wp),
                    "decode": _ctx_sol_decode_stub(cfg, wp),
                },
            },
            "benchmark": {
                "type": "sa-bench",
                "isl": cfg.workload.isl,
                "osl": 1,
                "concurrencies": str(cfg.ctx_config.benchmark_concurrency),
                "req_rate": "inf",
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

    TRT-LLM: disaggregated mode with TLLM_BENCHMARK_REQ_QUEUES_SIZE.
    SGLang:  disaggregated PD mode (prefill stub + decode worker) with NIXL.
             Decode worker uses max-running-requests=C + log filtering for
             isolated decode measurement, matching TRT-LLM's approach.
    """
    wp = _workload_params(cfg)
    conc = gen_item.concurrency if isinstance(gen_item.concurrency, int) else gen_item.concurrency[0]
    mtp_suffix = f"_mtp{gen_item.mtp_num}" if gen_item.mtp_num > 0 else ""
    conc_str = _format_concurrency(gen_item.concurrency)
    name = f"{cfg.name}_gen_{gen_item.mode}_c{conc_str}{mtp_suffix}"

    if cfg.engine_type == "sglang":
        prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_SGLANG_DISAGG_ENV)
        decode_env = cfg.backend.decode_environment or dict(_DEFAULT_SGLANG_DISAGG_ENV)
        decode_nodes = _nodes_for_tp(gen_item.tp_size, cfg.resources.gpus_per_node)

        config = {
            "name": name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": 1,
                "prefill_workers": 1,
                "decode_nodes": decode_nodes,
                "decode_workers": 1,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "frontend": {
                "type": "sglang",
            },
            "backend": {
                "type": "sglang",
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "sglang_config": {
                    "prefill": _sglang_prefill_config(
                        cfg,
                        mtp_num=gen_item.mtp_num,
                    ),
                    "decode": _sglang_decode_config(
                        cfg, gen_item,
                        item_overrides=gen_item.decode_overrides,
                    ),
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
        }
    elif cfg.engine_type == "vllm":
        # vLLM: disaggregated mode (prefill stub + decode worker).
        # Uses --max-num-seqs = batch_size (per-replica) + log filtering
        # for exact concurrency match, same strategy as SGLang.
        prefill_env = _vllm_env(cfg, "prefill")
        decode_env = _vllm_env(cfg, "decode", for_gen_sol=True)
        dp_size = gen_item.dp_size or gen_item.tp_size

        config = {
            "name": name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": 1,
                "prefill_workers": 1,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": math.ceil(dp_size / cfg.resources.gpus_per_node),
                "decode_workers": 1,
                "gpus_per_decode": dp_size,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": "vllm",
                "connector": cfg.backend.connector,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "vllm_config": {
                    "prefill": _vllm_prefill_config(cfg, wp),
                    "decode": _vllm_decode_config(
                        cfg, wp, gen_item, for_gen_sol=True,
                    ),
                },
                **({"vllm_kimi_fp4_patch": True} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
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
            **({"extra_mount": ["${SRTCTL_SOURCE_DIR}/tools/rate_matching/patches:/patches"]} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
        }
    else:
        # TRT-LLM path (original logic)
        prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_PREFILL_ENV)
        decode_env = cfg.backend.decode_environment or dict(_DEFAULT_DECODE_ENV)
        decode_env = {**decode_env, **_GEN_SOL_DECODE_ENV_EXTRAS}
        decode_env["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] = str(conc)

        decode_nodes_per_worker = _nodes_for_tp(gen_item.tp_size, cfg.resources.gpus_per_node)

        config = {
            "name": name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "sbatch_directives": {},
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": 1,
                "prefill_workers": 1,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": decode_nodes_per_worker,
                "decode_workers": 1,
                "gpus_per_decode": gen_item.tp_size,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": cfg.engine_type,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "trtllm_config": {
                    "prefill": _prefill_config(
                        cfg, wp, mtp_num=gen_item.mtp_num,
                        item_prefill_overrides=gen_item.prefill_overrides,
                    ),
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
    sweep_name: str = "",
) -> str:
    """Generate recipe-convention filename.

    Format: {sweep}_e2e_{ctx}P{gen}D_c{pw_conc}_{mode}[_mtp{M}][_1.05x]
    Falls back to legacy format if sweep_name is empty.
    """
    mtp_suffix = f"_mtp{mtp_num}" if mtp_num > 0 else ""
    if sweep_name:
        name = f"{sweep_name}_e2e_{ctx_instances}P{gen_instances}D_c{per_worker_conc}_{mode}{mtp_suffix}"
    else:
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
        sweep_name=cfg.name,
    )

    # Per-item overrides carried through from the original GenSweepItem
    item_decode_overrides = pareto_point.get("decode_overrides")
    item_prefill_overrides = pareto_point.get("prefill_overrides")

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
        decode_overrides=item_decode_overrides,
        prefill_overrides=item_prefill_overrides,
        dp_size=pareto_point.get("dp_size"),
        all2all_backend=pareto_point.get("all2all_backend"),
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

    if cfg.engine_type == "sglang":
        # SGLang E2E: disaggregated PD mode with NIXL KV transfer.
        # Validated on B200 (job 4756: Qwen3.5-397B, 1P+1D, TP8, NIXL/UCX).
        prefill_env = cfg.backend.prefill_environment or dict(_DEFAULT_SGLANG_DISAGG_ENV)
        decode_env = cfg.backend.decode_environment or dict(_DEFAULT_SGLANG_DISAGG_ENV)
        decode_nodes_per_worker = _nodes_for_tp(tp_size, cfg.resources.gpus_per_node)

        config = {
            "name": recipe_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": ctx_instances,
                "prefill_workers": ctx_instances,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": gen_instances * decode_nodes_per_worker,
                "decode_workers": gen_instances,
                "gpus_per_decode": tp_size,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "frontend": {
                "type": "sglang",
            },
            "backend": {
                "type": "sglang",
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "sglang_config": {
                    "prefill": _sglang_prefill_config(
                        cfg,
                        mtp_num=mtp_num,
                        item_overrides=item_prefill_overrides,
                    ),
                    "decode": _sglang_decode_config(
                        cfg, gen_item,
                        item_overrides=item_decode_overrides,
                    ),
                },
            },
            "benchmark": {
                "type": "sa-bench",
                "isl": cfg.workload.isl,
                "osl": cfg.workload.osl,
                "concurrencies": str(system_concurrency),
                "req_rate": "inf",
            },
        }
    elif cfg.engine_type == "vllm":
        # vLLM E2E: disaggregated PD mode with NixlConnector.
        # prefill_instances = ctx_instances (each runs DP+EP on one node)
        # decode_instances = gen_instances (each runs DP+EP on one node)
        prefill_env = _vllm_env(cfg, "prefill")
        decode_env = _vllm_env(cfg, "decode")
        dp_size = gen_item.dp_size or gen_item.tp_size

        config = {
            "name": recipe_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": ctx_instances,
                "prefill_workers": ctx_instances,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": gen_instances * math.ceil(dp_size / cfg.resources.gpus_per_node),
                "decode_workers": gen_instances,
                "gpus_per_decode": dp_size,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": "vllm",
                "connector": cfg.backend.connector,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "vllm_config": {
                    "prefill": _vllm_prefill_config(
                        cfg, wp, item_overrides=item_prefill_overrides,
                    ),
                    "decode": _vllm_decode_config(
                        cfg, wp, gen_item, for_gen_sol=False,
                    ),
                },
                **({"vllm_kimi_fp4_patch": True} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
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
            **({"extra_mount": ["${SRTCTL_SOURCE_DIR}/tools/rate_matching/patches:/patches"]} if getattr(cfg.backend, "vllm_kimi_fp4_patch", False) else {}),
        }
    else:
        # TRT-LLM E2E path (original logic)
        decode_nodes_per_worker = _nodes_for_tp(tp_size, cfg.resources.gpus_per_node)

        config = {
            "name": recipe_name,
            "model": {
                "path": cfg.model.path,
                "container": cfg.model.container,
                "precision": cfg.model.precision,
            },
            "sbatch_directives": {},
            "resources": {
                "gpu_type": cfg.resources.gpu_type,
                "prefill_nodes": ctx_instances,
                "prefill_workers": ctx_instances,
                "gpus_per_prefill": cfg.resources.ctx_gpus_per_instance,
                "decode_nodes": gen_instances * decode_nodes_per_worker,
                "decode_workers": gen_instances,
                "gpus_per_decode": tp_size,
                "gpus_per_node": cfg.resources.gpus_per_node,
            },
            "backend": {
                "type": cfg.engine_type,
                "prefill_environment": prefill_env,
                "decode_environment": decode_env,
                "trtllm_config": {
                    "prefill": _prefill_config(
                        cfg, wp, mtp_num=mtp_num,
                        item_prefill_overrides=item_prefill_overrides,
                    ),
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
                sweep_name=cfg.name,
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
# Base / Override mode — CTX SOL from user-provided base
# ---------------------------------------------------------------------------

def generate_ctx_sol_from_base(
    cfg: RateMatchingSweepConfig,
    output_path: str | None = None,
) -> dict:
    """Generate CTX SOL config directly from cfg.ctx_sol_base.

    The user provides a complete srt-slurm config dict; we write it as-is.
    Returns the config dict (same interface as generate_ctx_sol_config).
    """
    if cfg.ctx_sol_base is None:
        raise ValueError("ctx_sol_base is not set in sweep config")
    config = dict(cfg.ctx_sol_base)
    if output_path:
        _write_yaml(config, output_path)
    return config


# ---------------------------------------------------------------------------
# Base / Override mode — GEN SOL group delta + concurrency overrides
# ---------------------------------------------------------------------------

def _gen_group_base_delta_trtllm(
    cfg: RateMatchingSweepConfig,
    gen_item: GenSweepItem,
    wp: dict,
) -> dict:
    """Compute group-specific delta to deep-merge onto gen_sol_base (TRT-LLM).

    Only includes fields that are group-specific: batch_size-dependent decode
    settings, concurrency, mode-dependent flags, and group overrides.
    The user's gen_sol_base provides all other fields (model, prefill config,
    environments, etc.) as the source of truth.
    """
    conc = gen_item.concurrency if isinstance(gen_item.concurrency, int) else gen_item.concurrency[0]
    batch_size = gen_item.batch_size
    max_num_tokens = gen_item.max_num_tokens or batch_size
    is_dep = gen_item.mode == "dep"
    decode_nodes_per_worker = _nodes_for_tp(gen_item.tp_size, cfg.resources.gpus_per_node)

    decode_delta: dict[str, Any] = {
        "tensor_parallel_size": gen_item.tp_size,
        "moe_expert_parallel_size": gen_item.tp_size,
        "enable_attention_dp": is_dep,
        "enable_lm_head_tp_in_adp": is_dep,
        "max_batch_size": batch_size,
        "max_num_tokens": max_num_tokens,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": _cuda_graph_batch_sizes(batch_size),
        },
    }

    if gen_item.mtp_num > 0:
        decode_delta["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": gen_item.mtp_num,
        }

    if gen_item.decode_overrides:
        decode_delta = _deep_merge(decode_delta, gen_item.decode_overrides)

    delta: dict[str, Any] = {
        "resources": {
            "decode_nodes": decode_nodes_per_worker,
            "gpus_per_decode": gen_item.tp_size,
        },
        "backend": {
            "decode_environment": {
                "TLLM_BENCHMARK_REQ_QUEUES_SIZE": str(conc),
            },
            "trtllm_config": {
                "decode": decode_delta,
            },
        },
        "benchmark": {
            "concurrencies": _format_concurrency(gen_item.concurrency),
        },
    }

    # Include prefill overrides only if the group specifies them
    if gen_item.prefill_overrides:
        delta["backend"]["trtllm_config"]["prefill"] = gen_item.prefill_overrides

    return delta


def _gen_group_base_delta_sglang(
    cfg: RateMatchingSweepConfig,
    gen_item: GenSweepItem,
    wp: dict,
) -> dict:
    """Compute group-specific delta for SGLang gen_sol_base.

    Only sets concurrency-dependent decode fields and group-level
    decode_overrides.  The user's gen_sol_base provides the complete
    prefill config and common decode settings.
    """
    decode_nodes = _nodes_for_tp(gen_item.tp_size, cfg.resources.gpus_per_node)
    conc = gen_item.concurrency if isinstance(gen_item.concurrency, int) else gen_item.concurrency[0]

    decode_delta: dict[str, Any] = {
        "max-running-requests": conc,
        "cuda-graph-max-bs": max(128, conc),
    }

    # MTP settings on decode (if not already in gen_sol_base)
    if gen_item.mtp_num > 0:
        decode_delta["speculative-algorithm"] = "NEXTN"
        decode_delta["speculative-num-steps"] = gen_item.mtp_num
        decode_delta["speculative-eagle-topk"] = 1

    # Group-level decode overrides (e.g. stream-interval, scheduler-recv-interval)
    if gen_item.decode_overrides:
        decode_delta.update(gen_item.decode_overrides)

    delta: dict[str, Any] = {
        "resources": {
            "decode_nodes": decode_nodes,
        },
        "backend": {
            "sglang_config": {
                "decode": decode_delta,
            },
        },
        "benchmark": {
            "concurrencies": str(conc),
        },
    }
    return delta


def _gen_group_base_delta_vllm(
    cfg: RateMatchingSweepConfig,
    gen_item: GenSweepItem,
    wp: dict,
) -> dict:
    """Compute group-specific delta for vLLM gen_sol_base.

    Only sets batch_size-dependent decode fields and group-level overrides.
    The user's gen_sol_base provides the complete prefill config and common
    decode settings.
    """
    dp_size = gen_item.dp_size or gen_item.tp_size
    batch_size = gen_item.batch_size
    max_num_tokens = gen_item.max_num_tokens or 4096

    decode_delta: dict[str, Any] = {
        "data-parallel-size": dp_size,
        "max-num-seqs": batch_size,
        "max-num-batched-tokens": max_num_tokens,
        "max-cudagraph-capture-size": batch_size,
    }
    if gen_item.all2all_backend:
        decode_delta["all2all-backend"] = gen_item.all2all_backend

    if gen_item.decode_overrides:
        decode_delta = _deep_merge(decode_delta, gen_item.decode_overrides)

    delta: dict[str, Any] = {
        "resources": {
            "decode_nodes": math.ceil(dp_size / cfg.resources.gpus_per_node),
            "gpus_per_decode": dp_size,
        },
        "backend": {
            "vllm_config": {
                "decode": decode_delta,
            },
        },
        "benchmark": {
            "concurrencies": _format_concurrency(gen_item.concurrency),
        },
    }
    return delta


def _gen_concurrency_override_trtllm(conc: int) -> dict:
    """Per-concurrency override for TRT-LLM.

    Only the fields that change per concurrency: TLLM_BENCHMARK_REQ_QUEUES_SIZE
    and benchmark.concurrencies.
    """
    return {
        "backend": {
            "decode_environment": {
                "TLLM_BENCHMARK_REQ_QUEUES_SIZE": str(conc),
            },
        },
        "benchmark": {
            "concurrencies": str(conc),
        },
    }


def _gen_concurrency_override_sglang(conc: int) -> dict:
    """Per-concurrency override for SGLang.

    Changes max-running-requests and cuda-graph-max-bs on the decode worker.
    """
    return {
        "backend": {
            "sglang_config": {
                "decode": {
                    "max-running-requests": conc,
                    "cuda-graph-max-bs": max(128, conc),
                },
            },
        },
        "benchmark": {
            "concurrencies": str(conc),
        },
    }


def _gen_concurrency_override_vllm(conc: int) -> dict:
    """Per-concurrency override for vLLM.

    Changes max-num-seqs and max-cudagraph-capture-size on the decode worker.
    """
    return {
        "backend": {
            "vllm_config": {
                "decode": {
                    "max-num-seqs": conc,
                    "max-cudagraph-capture-size": conc,
                },
            },
        },
        "benchmark": {
            "concurrencies": str(conc),
        },
    }


def generate_gen_sol_override_config(
    cfg: RateMatchingSweepConfig,
    group_name: str,
    group_items: list[GenSweepItem],
    output_path: str | None = None,
) -> dict:
    """Generate a base/override YAML for one GEN SOL group.

    Uses cfg.gen_sol_base as the foundation, deep-merges group-specific
    settings into a `base:` key, then creates `override_c{N}:` keys for
    each concurrency level. The first item's concurrency is used for
    the base; remaining concurrencies become overrides.

    The output is compatible with srt-slurm's `generate_override_configs()`
    and can be submitted as `srtctl apply -f gen_sol_group.yaml:override_cN`.

    Returns:
        The full multi-document dict (with "base", "override_c*" keys).
    """
    if cfg.gen_sol_base is None:
        raise ValueError("gen_sol_base is not set in sweep config")

    wp = _workload_params(cfg)

    # Select the engine-specific delta builder
    delta_builder = {
        "trtllm": _gen_group_base_delta_trtllm,
        "sglang": _gen_group_base_delta_sglang,
        "vllm": _gen_group_base_delta_vllm,
    }[cfg.engine_type]

    conc_override_builder = {
        "trtllm": _gen_concurrency_override_trtllm,
        "sglang": _gen_concurrency_override_sglang,
        "vllm": _gen_concurrency_override_vllm,
    }[cfg.engine_type]

    # Use the first item as the "base" representative for this group.
    # All items in a group share the same mode/tp_size/mtp_num (via defaults),
    # only concurrency and batch_size vary.
    first_item = group_items[0]
    first_conc = first_item.concurrency if isinstance(first_item.concurrency, int) else first_item.concurrency[0]

    # Build the base: deep_merge(gen_sol_base, group_delta)
    group_delta = delta_builder(cfg, first_item, wp)
    base_config = _deep_merge(dict(cfg.gen_sol_base), group_delta)

    # Set the name for the base
    mtp_suffix = f"_mtp{first_item.mtp_num}" if first_item.mtp_num > 0 else ""
    base_config["name"] = f"{cfg.name}_gen_{group_name}{mtp_suffix}"

    # Build overrides for all concurrencies (including the first, so every
    # concurrency can be targeted individually via selector)
    result: dict[str, Any] = {"base": base_config}

    # Collect all unique concurrencies from the group
    all_concurrencies: list[int] = []
    for item in group_items:
        conc_list = item.concurrency if isinstance(item.concurrency, list) else [item.concurrency]
        for c in conc_list:
            if c not in all_concurrencies:
                all_concurrencies.append(c)

    for conc in sorted(all_concurrencies):
        if conc == first_conc:
            # The base already has this concurrency; skip override
            continue
        override = conc_override_builder(conc)
        result[f"override_c{conc}"] = override

    if output_path:
        _write_yaml(result, output_path)
    return result


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
