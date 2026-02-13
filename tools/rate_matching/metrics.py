"""
Rate-matching metric calculations.

Core formulas:
  gen_req_rate = output_throughput / (osl * avg_random_ratio)
  ctx_gen_inst_ratio = gen_req_rate / ctx_request_rate
  output_tput_per_gpu = output_throughput / (ctx_gpus * ctx_gen_inst_ratio + ep_rank)

Where:
  avg_random_ratio = (random_ratio + 1) / 2  (1.0 for random data)
  output_throughput = throughput_per_user * concurrency
  throughput_per_user = (1 / elapsed_time_avg) * mtp_accept_rate

Terminology:
  - per-worker concurrency: requests handled by one decode worker (SOL context)
  - system concurrency: total requests across all decode workers in E2E
    (= per_worker_conc * gen_instances * multiplier)
"""

from __future__ import annotations

import math
from typing import Any


def compute_rate_matching(
    ctx_result: dict,
    gen_result: dict,
    osl: int,
    random_ratio: float = 1.0,
    gpus_per_ctx_instance: int = 8,
    gpus_per_gen_instance: int = 8,
    max_total_gpus: int = 64,
    isl: int = 0,
) -> dict:
    """Compute rate-matched allocation from CTX and GEN SOL results.

    Args:
        ctx_result: Output of process_ctx_results (needs 'request_rate_req_per_s',
                    and optionally 'avg_prev_device_step_time_ms' for E2E latency
                    estimation).
        gen_result: Output of process_gen_results (needs 'concurrency', 'tpot_ms',
                    'output_throughput', 'throughput_per_user', 'mode', etc.)
        osl: Output sequence length (needed for gen_req_rate calculation)
        random_ratio: Random ratio from config (1.0 for random data, <1.0 for
                      partially deterministic). avg_random_ratio = (random_ratio + 1) / 2
        gpus_per_ctx_instance: GPUs per CTX worker (typically tp_size)
        gpus_per_gen_instance: GPUs per GEN worker (ep_rank/tp_size)
        max_total_gpus: Maximum total GPU budget for allocation search
        isl: Input sequence length (needed for total_throughput and
             estimate_e2e_latency calculations).

    Returns:
        Dict with allocation info: ctx_instances, gen_instances, ratio_str,
        output_tput_per_gpu, total_gpus, gen_req_rate, and derived metrics
        (total_throughput, output_tput_per_gen_gpu, estimate_e2e_latency).
    """
    ctx_request_rate = ctx_result.get('request_rate_req_per_s',
                                      ctx_result.get('request_rate', 0))
    gen_concurrency = gen_result['concurrency']
    gen_tpot_ms = gen_result['tpot_ms']
    gen_output_throughput = gen_result['output_throughput']
    gen_throughput_per_user = gen_result['throughput_per_user']
    gen_interactivity = gen_result.get('interactivity', gen_throughput_per_user)
    avg_step_time_ms = gen_result.get('avg_step_time_ms', gen_tpot_ms)
    mode = gen_result.get('mode', 'tep')
    batch_size = gen_result.get('batch_size', gen_concurrency)
    mtp_num = gen_result.get('mtp', 0)
    mtp_accept_rate = gen_result.get('mtp_accept_rate', 1.0)
    tp_size = gen_result.get('tp_size', gpus_per_gen_instance)
    ep_rank = gen_result.get('ep_rank', tp_size)

    # -----------------------------------------------------------------------
    # gen_req_rate: new-request arrival capacity of one GEN instance (req/s)
    # Original formula: gen_req_rate = output_throughput / (osl * avg_random_ratio)
    # -----------------------------------------------------------------------
    avg_random_ratio = (random_ratio + 1.0) / 2.0
    gen_req_rate = gen_output_throughput / (osl * avg_random_ratio) if osl > 0 else 0

    # -----------------------------------------------------------------------
    # ctx_gen_inst_ratio: how many CTX instances per GEN instance are needed
    # to keep the GEN instance fully loaded.
    # -----------------------------------------------------------------------
    if ctx_request_rate > 0 and gen_req_rate > 0:
        ctx_gen_inst_ratio = gen_req_rate / ctx_request_rate
    else:
        ctx_gen_inst_ratio = 0

    # -----------------------------------------------------------------------
    # output_tput_per_gpu (SOL prediction for one GEN instance):
    # Denominator = GPUs supporting one GEN instance (its share of CTX + GEN)
    # -----------------------------------------------------------------------
    denom_per_gen = gpus_per_ctx_instance * ctx_gen_inst_ratio + ep_rank
    output_tput_per_gpu = gen_output_throughput / denom_per_gen if denom_per_gen > 0 else 0

    # -----------------------------------------------------------------------
    # E2E ratio: find integer CTX:GEN allocation within GPU budget
    # -----------------------------------------------------------------------
    best = _find_best_allocation(
        ctx_gen_inst_ratio=ctx_gen_inst_ratio,
        gpus_per_ctx=gpus_per_ctx_instance,
        gpus_per_gen=ep_rank,
        max_total_gpus=max_total_gpus,
    )

    # Compute total output throughput for the best allocation
    total_output_throughput = gen_output_throughput * best["gen_instances"]
    best_total_gpus = best["total_gpus"]
    total_output_tput_per_gpu = total_output_throughput / best_total_gpus if best_total_gpus > 0 else 0

    # -----------------------------------------------------------------------
    # Additional derived metrics
    # -----------------------------------------------------------------------

    # output_tput_per_gen_gpu: throughput normalised by GEN GPUs only.
    # Useful for comparing GEN efficiency across modes independently of the
    # CTX allocation.
    output_tput_per_gen_gpu = gen_output_throughput / ep_rank if ep_rank > 0 else 0

    # total_throughput / total_tput_per_gpu: input + output token throughput.
    if isl > 0 and osl > 0:
        io_ratio = (isl + osl) / osl
        total_throughput = gen_output_throughput * io_ratio
        total_tput_per_gpu = output_tput_per_gpu * io_ratio
    else:
        total_throughput = gen_output_throughput
        total_tput_per_gpu = output_tput_per_gpu

    # estimate_e2e_latency: TTFT + decode time for the expected output length.
    # e2e_latency = ctx_step_time + (expected_output_tokens - 1) / throughput_per_user
    # ctx_step_time comes from CTX results (avg_prev_device_step_time in seconds).
    ctx_step_time_ms = ctx_result.get('avg_prev_device_step_time_ms')
    if ctx_step_time_ms is not None and gen_throughput_per_user > 0 and osl > 0:
        ctx_step_time_s = ctx_step_time_ms / 1000.0
        expected_output_tokens = osl * avg_random_ratio
        estimate_e2e_latency_s = ctx_step_time_s + (expected_output_tokens - 1) / gen_throughput_per_user
    else:
        estimate_e2e_latency_s = None

    result = {
        "config_name": f"{mode}_c{gen_concurrency}_b{batch_size}",
        "mode": mode,
        "batch_size": batch_size,
        "concurrency": gen_concurrency,
        "mtp_num": mtp_num,
        "mtp_accept_rate": mtp_accept_rate,
        "tp_size": tp_size,
        "ep_rank": ep_rank,
        "avg_step_time_ms": avg_step_time_ms,
        "throughput_per_user": gen_throughput_per_user,
        "interactivity": gen_interactivity,
        "tpot_ms": gen_tpot_ms,
        "output_throughput": gen_output_throughput,
        "output_tput_per_gen_gpu": output_tput_per_gen_gpu,
        "total_throughput": total_throughput,
        "total_tput_per_gpu": total_tput_per_gpu,
        "ctx_request_rate": ctx_request_rate,
        "gen_req_rate": gen_req_rate,
        "ctx_gen_inst_ratio": ctx_gen_inst_ratio,
        "output_tput_per_gpu": output_tput_per_gpu,
        "total_output_throughput": total_output_throughput,
        "total_output_tput_per_gpu": total_output_tput_per_gpu,
        "ratio_str": f"{best['ctx_instances']}:{best['gen_instances']}",
        **best,
    }

    if estimate_e2e_latency_s is not None:
        result["estimate_e2e_latency_s"] = round(estimate_e2e_latency_s, 4)

    return result


def _find_best_allocation(
    ctx_gen_inst_ratio: float,
    gpus_per_ctx: int = 8,
    gpus_per_gen: int = 8,
    max_total_gpus: int = 64,
) -> dict:
    """Find optimal integer CTX:GEN allocation within GPU budget.

    Finds ctx, gen such that:
      - ctx/gen >= ctx_gen_inst_ratio
      - gpus_per_ctx * ctx + gpus_per_gen * gen <= max_total_gpus
      - Excess ratio is minimized (tightest fit)

    Returns:
        Dict with ctx_instances, gen_instances, ctx_gpus, gen_gpus, total_gpus.
    """
    best_ctx = 1
    best_gen = 1
    best_ratio_diff = float('inf')

    max_gen = max_total_gpus // gpus_per_gen

    for gen in range(1, max_gen + 1):
        # Minimum CTX needed: ctx/gen >= ctx_gen_inst_ratio
        ctx = max(1, math.ceil(ctx_gen_inst_ratio * gen))

        # Check GPU budget
        total = gpus_per_ctx * ctx + gpus_per_gen * gen
        if total > max_total_gpus:
            continue

        actual_ratio = ctx / gen
        ratio_diff = actual_ratio - ctx_gen_inst_ratio

        # Want smallest ratio >= target
        if ratio_diff >= 0 and ratio_diff < best_ratio_diff:
            best_ctx = ctx
            best_gen = gen
            best_ratio_diff = ratio_diff

    # Simplify ratio
    g = math.gcd(best_ctx, best_gen)
    simplified_ctx = best_ctx // g
    simplified_gen = best_gen // g

    ctx_gpus = best_ctx * gpus_per_ctx
    gen_gpus = best_gen * gpus_per_gen
    total_gpus = ctx_gpus + gen_gpus

    return {
        "ctx_instances": best_ctx,
        "gen_instances": best_gen,
        "simplified_ratio": f"{simplified_ctx}:{simplified_gen}",
        "ctx_gpus": ctx_gpus,
        "gen_gpus": gen_gpus,
        "total_gpus": total_gpus,
    }


def compare_sol_vs_e2e(
    sol_prediction: dict,
    e2e_result: dict,
    total_gpus: int,
    gen_instances: int = 1,
    per_worker_conc: int = 0,
    tolerances: dict | None = None,
) -> dict:
    """Compare SOL predictions against E2E measurements.

    SOL values come from the rate-matching computation (device-side metrics).
    E2E values come from sa-bench client-side JSON results.

    Args:
        sol_prediction: Rate-matching result for this Pareto point
        e2e_result: sa-bench JSON result from E2E run
        total_gpus: Total GPUs used in E2E
        gen_instances: Number of decode workers
        per_worker_conc: Per-worker concurrency from SOL
        tolerances: Dict with tpot_tolerance_pct, throughput_tolerance_pct

    Returns:
        Dict with sol, e2e, diff_pct, pass/fail, total_gpus
    """
    if tolerances is None:
        tolerances = {"tpot_tolerance_pct": 15.0, "throughput_tolerance_pct": 20.0}

    # Extract E2E metrics from sa-bench result
    e2e_output_throughput = e2e_result.get('output_throughput', 0)
    e2e_median_tpot = e2e_result.get('median_tpot_ms', 0)

    # If median_tpot_ms not directly available, try to compute from output
    if e2e_median_tpot == 0:
        tpot_percentiles = e2e_result.get('tpot_percentiles', {})
        e2e_median_tpot = tpot_percentiles.get('p50', 0)

    if e2e_median_tpot == 0 and e2e_output_throughput > 0:
        max_conc = e2e_result.get('max_concurrency', per_worker_conc * gen_instances)
        if max_conc > 0:
            e2e_median_tpot = max_conc * 1000.0 / e2e_output_throughput

    e2e_interactivity = 1000.0 / e2e_median_tpot if e2e_median_tpot > 0 else 0
    e2e_tput_per_gpu = e2e_output_throughput / total_gpus if total_gpus > 0 else 0

    # SOL values (from rate-matching computation)
    sol_tpot = sol_prediction.get('tpot_ms', 0)
    sol_interactivity = sol_prediction.get('interactivity', 0)
    sol_output_tput_per_gpu = sol_prediction.get('output_tput_per_gpu', 0)

    # SOL total throughput = output_throughput_per_gen_instance * gen_instances
    sol_total_output_throughput = sol_prediction.get(
        'total_output_throughput',
        sol_prediction.get('output_throughput', 0) *
        sol_prediction.get('gen_instances', gen_instances)
    )
    sol_total_tput_per_gpu = sol_total_output_throughput / total_gpus if total_gpus > 0 else 0

    # Diff percentages
    # Positive = E2E is worse (higher TPOT, lower throughput)
    def _pct_diff(sol_val, e2e_val):
        if sol_val == 0:
            return 0
        return ((e2e_val - sol_val) / sol_val) * 100

    tpot_diff = _pct_diff(sol_tpot, e2e_median_tpot)
    inter_diff = _pct_diff(sol_interactivity, e2e_interactivity)
    tput_diff = _pct_diff(sol_total_output_throughput, e2e_output_throughput)
    tput_gpu_diff = _pct_diff(sol_total_tput_per_gpu, e2e_tput_per_gpu)

    # Pass/fail
    tpot_pass = abs(tpot_diff) <= tolerances["tpot_tolerance_pct"]
    tput_pass = abs(tput_gpu_diff) <= tolerances["throughput_tolerance_pct"]

    return {
        "sol": {
            "tpot_ms": sol_tpot,
            "interactivity": sol_interactivity,
            "output_throughput": sol_total_output_throughput,
            "output_tput_per_gpu": sol_total_tput_per_gpu,
        },
        "e2e": {
            "tpot_ms": e2e_median_tpot,
            "interactivity": e2e_interactivity,
            "output_throughput": e2e_output_throughput,
            "output_tput_per_gpu": e2e_tput_per_gpu,
        },
        "diff_pct": {
            "tpot": round(tpot_diff, 2),
            "interactivity": round(inter_diff, 2),
            "output_throughput": round(tput_diff, 2),
            "output_tput_per_gpu": round(tput_gpu_diff, 2),
        },
        "pass": {
            "tpot": tpot_pass,
            "throughput": tput_pass,
            "overall": tpot_pass and tput_pass,
        },
        "total_gpus": total_gpus,
    }
