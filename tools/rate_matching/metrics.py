"""
Rate-matching metric calculations.

Core math:
  1. Given CTX request_rate and GEN gen_req_rate, compute the optimal
     CTX:GEN instance ratio and GPU allocation.
  2. Given SOL predictions and E2E measurements, compute diff percentages
     and pass/fail status.

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
    gpus_per_instance: int = 8,
) -> dict:
    """Compute rate-matched allocation from CTX and GEN SOL results.

    Args:
        ctx_result: Output of process_ctx_results (needs 'request_rate_req_per_s')
        gen_result: Output of process_gen_results (needs 'concurrency', 'tpot_ms',
                    'output_throughput', 'throughput_per_user', 'mode', etc.)
        gpus_per_instance: GPUs per worker (TP size)

    Returns:
        Dict with allocation info: ctx_instances, gen_instances, ratio_str,
        output_tput_per_gpu, total_gpus, efficiency_pct, etc.
    """
    ctx_request_rate = ctx_result.get('request_rate_req_per_s', ctx_result.get('request_rate', 0))
    gen_concurrency = gen_result['concurrency']
    gen_tpot_ms = gen_result['tpot_ms']
    gen_output_throughput = gen_result['output_throughput']
    gen_throughput_per_user = gen_result['throughput_per_user']
    gen_interactivity = gen_result.get('interactivity', gen_throughput_per_user)
    mode = gen_result.get('mode', 'tep')
    batch_size = gen_result.get('batch_size', gen_concurrency)
    mtp_num = gen_result.get('mtp', 0)
    tp_size = gen_result.get('tp_size', gpus_per_instance)
    num_gpus = gen_result.get('num_gpus', gpus_per_instance)

    # GEN request rate: how many new requests per second one decode worker can accept
    # = concurrency / (osl * tpot_ms / 1000)
    # Simpler: gen_req_rate = ctx_request_rate when balanced (we solve for ratio)
    # Actually: gen_req_rate = output_throughput / osl (but we don't know osl here)
    # The standard formula: gen_req_rate = concurrency / avg_request_lifetime
    # For rate-matching: gen_req_rate = throughput_per_user (requests completing per second per user)
    # ... which is also interactivity

    # The ratio: how many GEN instances per CTX instance
    # ctx produces requests at ctx_request_rate (per instance)
    # gen consumes at gen_req_rate (per instance)
    # Balance: ctx_instances * ctx_request_rate = gen_instances * gen_req_rate
    # Ratio = ctx_instances / gen_instances = gen_req_rate / ctx_request_rate

    # gen_req_rate per decode instance = concurrency * (1000 / tpot_ms) / concurrency = 1000/tpot_ms
    # Actually, gen_req_rate = throughput / concurrent_users_served
    # Simplification: gen_req_rate = ctx_request_rate for a single decode worker at this concurrency

    # The correct formulation from the original rate-matching repo:
    # gen_req_rate = output_throughput / (concurrency * osl_effective)
    # But we have throughput_per_user = output_throughput / concurrency
    # And request_rate = throughput_per_user / osl

    # For the allocation formula we need:
    # ctx_gen_inst_ratio = ctx_request_rate / gen_req_rate_per_instance
    # where gen_req_rate_per_instance = how fast one GEN instance drains requests

    # From the previous sweep state, gen_req_rate was computed as:
    # gen_req_rate = concurrency / (avg_lifetime_s)
    # With avg_lifetime_s ~ osl / throughput_per_user
    # So gen_req_rate ~ throughput_per_user * concurrency / osl? No...
    # gen_req_rate = throughput_per_user / osl? That's requests/s/user...

    # Let's use the simple ratio formula that was in the previous sweep:
    # The GEN side processes `concurrency` requests simultaneously.
    # Each request takes ~osl steps, each step takes tpot_ms.
    # Time to finish one request = osl * tpot_ms / 1000 seconds.
    # Throughput of one GEN instance = concurrency / (osl * tpot_ms / 1000) req/s
    # But we don't have osl explicitly. However, output_throughput = total tok/s,
    # and if each request produces osl tokens, then gen_req_rate = output_throughput / osl.

    # Actually the simplest is: we know output_throughput (tok/s) from the GEN SOL.
    # And we know ctx_request_rate (req/s) from the CTX SOL, where each request
    # has ISL input tokens. The GEN request rate per instance in req/s is
    # output_throughput / osl. But we don't need osl because:
    #
    # ctx produces req/s = ctx_request_rate (per CTX instance)
    # gen consumes at some rate per GEN instance
    # At steady state: gen is fully loaded at `concurrency` in-flight requests.
    # Each request's decode time = osl / throughput_per_user seconds.
    # GEN throughput in req/s = concurrency / (osl / throughput_per_user)
    #                         = concurrency * throughput_per_user / osl
    #
    # This means gen_req_rate depends on osl. Rather than requiring it,
    # we can derive it from output_throughput:
    # gen_req_rate = output_throughput / osl  (but osl cancels out in ratio)

    # For ratio: ctx_instances * ctx_rr = gen_instances * gen_rr
    # gen_rr = output_throughput / osl (per GEN instance)
    # ctx_rr = ctx_request_rate (per CTX instance, each producing osl tokens to decode)
    # => ctx_i / gen_i = gen_rr / ctx_rr = (output_throughput / osl) / ctx_request_rate
    #
    # But this requires osl. Let's store it when we have it. For now, we use the
    # approach from the previous implementation:
    # gen_req_rate is embedded in the gen_result from the sweep pipeline
    # (where osl IS known from the workload config).

    # FALLBACK: use gen_req_rate if present (computed upstream with osl knowledge)
    gen_req_rate = gen_result.get('gen_req_rate')
    if gen_req_rate is None:
        # If not present, it will be computed in run_sweep with osl
        gen_req_rate = 0

    if gen_req_rate > 0 and ctx_request_rate > 0:
        ctx_gen_inst_ratio = ctx_request_rate / gen_req_rate
    else:
        ctx_gen_inst_ratio = 0

    # Optimal allocation
    total_ratio = 1.0 + ctx_gen_inst_ratio  # GEN + CTX relative to GEN
    # For N total instance slots, gen_instances = N / total_ratio, ctx = N - gen
    # We try a range of total slots and pick the best

    best = None
    for total_instances in range(2, 16):
        gen_i = max(1, round(total_instances / total_ratio))
        ctx_i = total_instances - gen_i
        if ctx_i < 1:
            ctx_i = 1
            gen_i = total_instances - 1
        if gen_i < 1:
            continue

        ctx_gpus = ctx_i * gpus_per_instance
        gen_gpus = gen_i * gpus_per_instance
        total_gpus = ctx_gpus + gen_gpus

        # Throughput: limited by the bottleneck side
        ctx_capacity = ctx_i * ctx_request_rate  # req/s
        gen_capacity = gen_i * gen_req_rate if gen_req_rate > 0 else float('inf')
        effective_req_rate = min(ctx_capacity, gen_capacity)

        # Output throughput = effective_req_rate * tokens_per_request
        # tokens_per_request = output_throughput / gen_req_rate (per instance)
        if gen_req_rate > 0:
            tokens_per_req = gen_output_throughput / gen_req_rate
            total_output_throughput = effective_req_rate * tokens_per_req
        else:
            total_output_throughput = gen_output_throughput * gen_i

        output_tput_per_gpu = total_output_throughput / total_gpus

        # Efficiency: how well utilised is the bottleneck
        if gen_req_rate > 0 and ctx_request_rate > 0:
            ctx_util = effective_req_rate / ctx_capacity
            gen_util = effective_req_rate / gen_capacity
            efficiency = min(ctx_util, gen_util) * 100
        else:
            efficiency = 0

        candidate = {
            "ctx_instances": ctx_i,
            "gen_instances": gen_i,
            "ctx_gpus": ctx_gpus,
            "gen_gpus": gen_gpus,
            "total_gpus": total_gpus,
            "output_tput_per_gpu": output_tput_per_gpu,
            "total_output_throughput": total_output_throughput,
            "efficiency_pct": efficiency,
        }

        if best is None or output_tput_per_gpu > best["output_tput_per_gpu"]:
            best = candidate

    if best is None:
        best = {
            "ctx_instances": 1, "gen_instances": 1,
            "ctx_gpus": gpus_per_instance, "gen_gpus": gpus_per_instance,
            "total_gpus": 2 * gpus_per_instance,
            "output_tput_per_gpu": gen_output_throughput / (2 * gpus_per_instance),
            "total_output_throughput": gen_output_throughput,
            "efficiency_pct": 0,
        }

    result = {
        "config_name": f"{mode}_c{gen_concurrency}_b{batch_size}",
        "mode": mode,
        "batch_size": batch_size,
        "concurrency": gen_concurrency,
        "mtp_num": mtp_num,
        "tp_size": tp_size,
        "throughput_per_user": gen_throughput_per_user,
        "interactivity": gen_interactivity,
        "tpot_ms": gen_tpot_ms,
        "output_throughput": gen_output_throughput,
        "ctx_request_rate": ctx_request_rate,
        "gen_req_rate": gen_req_rate,
        "ctx_gen_inst_ratio": ctx_gen_inst_ratio,
        "ratio_str": f"{best['ctx_instances']}:{best['gen_instances']}",
        **best,
    }
    return result


def compare_sol_vs_e2e(
    sol_prediction: dict,
    e2e_result: dict,
    total_gpus: int,
    gen_instances: int = 1,
    per_worker_conc: int = 0,
    tolerances: dict | None = None,
) -> dict:
    """Compare SOL predictions against E2E measurements.

    Args:
        sol_prediction: Rate-matching result for this Pareto point
        e2e_result: sa-bench JSON result from E2E run
        total_gpus: Total GPUs used in E2E
        gen_instances: Number of decode workers
        per_worker_conc: Per-worker concurrency from SOL
        tolerances: Dict with tpot_tolerance_pct, throughput_tolerance_pct

    Returns:
        Dict with sol, e2e, diff_pct, pass/fail, total_gpus, bottleneck
    """
    if tolerances is None:
        tolerances = {"tpot_tolerance_pct": 15.0, "throughput_tolerance_pct": 20.0}

    # Extract E2E metrics from sa-bench result
    e2e_output_throughput = e2e_result.get('output_throughput', 0)
    e2e_median_tpot = e2e_result.get('median_tpot_ms', 0)

    # If median_tpot_ms not directly available, try to compute from output
    if e2e_median_tpot == 0:
        # Try per_output_token percentiles
        tpot_percentiles = e2e_result.get('tpot_percentiles', {})
        e2e_median_tpot = tpot_percentiles.get('p50', 0)

    if e2e_median_tpot == 0 and e2e_output_throughput > 0:
        max_conc = e2e_result.get('max_concurrency', per_worker_conc * gen_instances)
        if max_conc > 0:
            e2e_median_tpot = max_conc * 1000.0 / e2e_output_throughput

    e2e_interactivity = 1000.0 / e2e_median_tpot if e2e_median_tpot > 0 else 0
    e2e_tput_per_gpu = e2e_output_throughput / total_gpus if total_gpus > 0 else 0

    # SOL values
    sol_tpot = sol_prediction.get('tpot_ms', 0)
    sol_interactivity = sol_prediction.get('interactivity', 0)
    sol_output_throughput = sol_prediction.get('total_output_throughput',
                                               sol_prediction.get('output_throughput', 0) *
                                               sol_prediction.get('gen_instances', gen_instances))
    sol_tput_per_gpu = sol_output_throughput / total_gpus if total_gpus > 0 else 0

    # Diff percentages (positive = E2E is worse/higher for TPOT, lower for throughput)
    def _pct_diff(sol_val, e2e_val):
        if sol_val == 0:
            return 0
        return ((e2e_val - sol_val) / sol_val) * 100

    tpot_diff = _pct_diff(sol_tpot, e2e_median_tpot)
    inter_diff = _pct_diff(sol_interactivity, e2e_interactivity)
    tput_diff = _pct_diff(sol_output_throughput, e2e_output_throughput)
    tput_gpu_diff = _pct_diff(sol_tput_per_gpu, e2e_tput_per_gpu)

    # Pass/fail
    tpot_pass = abs(tpot_diff) <= tolerances["tpot_tolerance_pct"]
    tput_pass = abs(tput_gpu_diff) <= tolerances["throughput_tolerance_pct"]

    return {
        "sol": {
            "tpot_ms": sol_tpot,
            "interactivity": sol_interactivity,
            "output_throughput": sol_output_throughput,
            "output_tput_per_gpu": sol_tput_per_gpu,
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
        "bottleneck": "GEN",  # In disaggregated serving, GEN is typically the bottleneck
    }
