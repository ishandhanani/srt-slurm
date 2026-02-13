#!/usr/bin/env python3
"""
Rate-matching sweep orchestrator.

Drives the full pipeline:
  Phase 1: Generate configs (CTX SOL + GEN SOL)
  Phase 2: Submit & process CTX-only SOL
  Phase 3: Submit & process GEN-only SOL jobs
  Phase 4: Compute rate-matching metrics
  Phase 5: Extract Pareto frontier
  Phase 6: Generate & run E2E validation (multiple concurrency variants)
  Phase 7: Dashboard export & summary

State is persisted to sweep_state.json after each phase for resume support.

Usage (via CLI):
    srtctl-rate-match run -c sweep.yaml -o ./sweeps/my_sweep
    srtctl-rate-match run -c sweep.yaml -o ./sweeps/my_sweep --resume
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure this directory is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from generate_configs import (
    generate_ctx_sol_config,
    generate_e2e_configs_from_pareto,
    generate_gen_sol_config,
)
from metrics import compare_sol_vs_e2e, compute_rate_matching
from pareto import extract_pareto_frontier
from process_ctx_results import find_prefill_log, parse_log_file as parse_ctx_log, process_ctx_data
from process_gen_results import (
    find_decode_log,
    get_mtp_accept_rate,
    parse_log_file as parse_gen_log,
    process_gen_data,
    process_gen_data_all_concurrencies,
)
from schema import RateMatchingSweepConfig, load_sweep_config


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class SweepState:
    """Mutable sweep state, serialised to JSON after each phase."""

    def __init__(self):
        self.sweep_name: str = ""
        self.sweep_config_path: str = ""
        self.output_dir: str = ""
        self.created_at: str = ""
        self.last_updated: str = ""
        self.phase: str = "init"  # init, ctx, gen, rate_match, pareto, e2e, complete

        # Job tracking
        self.ctx_job: dict = {}
        self.gen_jobs: list[dict] = []
        self.ctx_result: dict = {}
        self.gen_results: list[dict] = []

        # Analysis
        self.rate_matching_results: list[dict] = []
        self.pareto_frontier: list[dict] = []

        # E2E
        self.e2e_configs: list[dict] = []  # list of {config_path, pareto_rank, multiplier, ...}
        self.e2e_jobs: list[dict] = []
        self.e2e_results: list[dict] = []
        self.sol_vs_e2e: list[dict] = []

    def save(self, path: str | None = None):
        target = path or str(Path(self.output_dir) / "sweep_state.json")
        self.last_updated = datetime.now().isoformat()
        with open(target, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SweepState":
        state = cls()
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# ---------------------------------------------------------------------------
# SLURM helpers (uses srtctl apply)
# ---------------------------------------------------------------------------

def submit_job(config_path: str, verbose: bool = True) -> str:
    """Submit a job via `srtctl apply -f <config>` and return the SLURM job ID."""
    cmd = ["srtctl", "apply", "-f", config_path]
    if verbose:
        print(f"  Submitting: srtctl apply -f {config_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if verbose and stdout:
        # Print the srtctl output (includes tracking commands)
        for line in stdout.split("\n"):
            print(f"    {line}")

    # Parse job ID from sbatch output embedded in srtctl output
    # srtctl prints "Job XXXXX submitted!" or similar
    job_id = None
    for line in stdout.split("\n"):
        # Look for "Job NNNNN submitted" pattern
        if "job" in line.lower() and "submitted" in line.lower():
            for word in line.split():
                if word.isdigit():
                    job_id = word
                    break
        # Also check raw sbatch output "Submitted batch job NNNNN"
        if "submitted batch job" in line.lower():
            parts = line.strip().split()
            if parts:
                job_id = parts[-1]

    if job_id is None:
        raise RuntimeError(
            f"Failed to parse job ID from srtctl output.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    return job_id


def poll_job(job_id: str, poll_interval: int = 300, verbose: bool = True) -> str:
    """Poll SLURM job until completion. Returns final status."""
    while True:
        try:
            result = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True,
            )
            status = result.stdout.strip()
        except Exception:
            status = ""

        if not status:
            # Job no longer in queue -- check sacct for final status
            try:
                result = subprocess.run(
                    ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                    capture_output=True, text=True,
                )
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                if lines:
                    status = lines[0]
                else:
                    status = "COMPLETED"
            except Exception:
                status = "COMPLETED"
            break

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            break

        if verbose:
            print(f"  Job {job_id}: {status} (next check in {poll_interval}s)")
        time.sleep(poll_interval)

    if verbose:
        print(f"  Job {job_id}: {status}")
    return status


def get_job_output_dir(job_id: str) -> str:
    """Get the output directory for a SLURM job."""
    # srt-slurm stores outputs in <srtctl_root>/outputs/<job_id>/
    srtctl_root = Path(__file__).resolve().parent.parent.parent
    output_dir = srtctl_root / "outputs" / str(job_id)
    return str(output_dir)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase1_generate_configs(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 1: Generate CTX SOL and GEN SOL configs."""
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Generating configs")
        print("=" * 60)

    configs_dir = Path(state.output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # CTX SOL
    ctx_path = str(configs_dir / "ctx_sol.yaml")
    generate_ctx_sol_config(cfg, output_path=ctx_path)
    state.ctx_job = {"config_path": ctx_path, "status": "pending"}
    if verbose:
        print(f"  CTX SOL config: {ctx_path}")

    # GEN SOL -- one per sweep item
    state.gen_jobs = []
    for i, gen_item in enumerate(cfg.gen_sweep):
        conc_str = str(gen_item.concurrency) if isinstance(gen_item.concurrency, int) else "x".join(str(c) for c in gen_item.concurrency)
        mtp_suffix = f"_mtp{gen_item.mtp_num}" if gen_item.mtp_num > 0 else ""
        fname = f"gen_sol_{gen_item.mode}_c{conc_str}{mtp_suffix}.yaml"
        gen_path = str(configs_dir / fname)
        generate_gen_sol_config(cfg, gen_item, output_path=gen_path)
        state.gen_jobs.append({
            "config_path": gen_path,
            "status": "pending",
            "gen_item_index": i,
        })
        if verbose:
            print(f"  GEN SOL config: {fname}")

    if verbose:
        print(f"\n  Total: 1 CTX + {len(state.gen_jobs)} GEN = {1 + len(state.gen_jobs)} configs")

    state.phase = "ctx"
    state.save()


def phase2_ctx_sol(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 2: Submit CTX-only SOL, poll, process results."""
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Running CTX-only SOL benchmark")
        print("=" * 60)

    ctx_job = state.ctx_job

    if ctx_job.get("status") == "completed" and state.ctx_result:
        if verbose:
            print("  CTX SOL already completed, skipping.")
        return

    if dry_run:
        if verbose:
            print(f"  [DRY-RUN] Would submit: {ctx_job['config_path']}")
        return

    # Submit
    if not ctx_job.get("job_id"):
        job_id = submit_job(ctx_job["config_path"], verbose=verbose)
        ctx_job["job_id"] = int(job_id)
        ctx_job["submit_time"] = datetime.now().isoformat()
        ctx_job["status"] = "running"
        ctx_job["output_dir"] = get_job_output_dir(job_id)
        state.save()

    # Poll
    status = poll_job(str(ctx_job["job_id"]), cfg.settings.poll_interval, verbose=verbose)
    ctx_job["status"] = "completed" if "COMPLETED" in status.upper() else "failed"
    ctx_job["complete_time"] = datetime.now().isoformat()
    state.save()

    if ctx_job["status"] != "completed":
        raise RuntimeError(f"CTX SOL job {ctx_job['job_id']} failed with status: {status}")

    # Process results
    if verbose:
        print("  Processing CTX results...")
    logs_dir = Path(ctx_job["output_dir"]) / "logs"
    log_file = find_prefill_log(logs_dir)
    if log_file is None:
        raise RuntimeError(f"No prefill log found in {logs_dir}")

    data = parse_ctx_log(log_file, verbose=False)
    if not data:
        raise RuntimeError(f"No data parsed from {log_file}")

    ctx_result = process_ctx_data(data, isl=cfg.workload.isl, verbose=False)
    if "error" in ctx_result:
        raise RuntimeError(f"CTX processing error: {ctx_result['error']}")

    state.ctx_result = ctx_result
    if verbose:
        print(f"  CTX request rate: {ctx_result['request_rate_req_per_s']:.4f} req/s")
        print(f"  CTX throughput:   {ctx_result['ctx_throughput_tokens_per_s']:,.0f} tok/s")

    state.phase = "gen"
    state.save()


def phase3_gen_sol(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 3: Submit GEN-only SOL jobs, poll, process results."""
    if verbose:
        parallel = cfg.settings.parallel_submissions
        mode = "parallel" if parallel else "serialised"
        print(f"\n{'=' * 60}")
        print(f"PHASE 3: Running GEN-only SOL benchmarks ({mode})")
        print(f"{'=' * 60}")

    if dry_run:
        for gj in state.gen_jobs:
            if verbose:
                print(f"  [DRY-RUN] Would submit: {gj['config_path']}")
        return

    # Submit all (or one at a time)
    pending = [gj for gj in state.gen_jobs if gj.get("status") not in ("completed",)]
    if not pending:
        if verbose:
            print("  All GEN SOL jobs already completed.")
        state.phase = "rate_match"
        state.save()
        return

    if cfg.settings.parallel_submissions:
        # Submit all at once
        for gj in pending:
            if not gj.get("job_id"):
                job_id = submit_job(gj["config_path"], verbose=verbose)
                gj["job_id"] = int(job_id)
                gj["submit_time"] = datetime.now().isoformat()
                gj["status"] = "running"
                gj["output_dir"] = get_job_output_dir(job_id)
        state.save()

        # Poll all
        for gj in pending:
            status = poll_job(str(gj["job_id"]), cfg.settings.poll_interval, verbose=verbose)
            gj["status"] = "completed" if "COMPLETED" in status.upper() else "failed"
            gj["complete_time"] = datetime.now().isoformat()
            state.save()
    else:
        # Serial: submit, poll, process one at a time
        for gj in pending:
            if not gj.get("job_id"):
                job_id = submit_job(gj["config_path"], verbose=verbose)
                gj["job_id"] = int(job_id)
                gj["submit_time"] = datetime.now().isoformat()
                gj["status"] = "running"
                gj["output_dir"] = get_job_output_dir(job_id)
                state.save()

            status = poll_job(str(gj["job_id"]), cfg.settings.poll_interval, verbose=verbose)
            gj["status"] = "completed" if "COMPLETED" in status.upper() else "failed"
            gj["complete_time"] = datetime.now().isoformat()
            state.save()

    # Process results for all completed GEN jobs
    # METHODOLOGY: Always use decode worker logs (prev_device_step_time), NOT
    # sa-bench client-side JSONs.  The original rate-matching repo processes
    # per-iteration decode logs to get the true device step time, then applies
    # hardcoded MTP accept rates.  sa-bench JSONs capture E2E client metrics
    # (including network overhead) and are NOT used for GEN SOL.
    #
    # Multi-concurrency handling: when sa-bench runs concurrencies "8x32x64"
    # sequentially, the decode log is continuous.  The exact-match filter on
    # num_scheduled_requests (Step 4 in the methodology) naturally segments
    # iterations by concurrency.
    state.gen_results = []
    isl = cfg.workload.isl
    for idx, gj in enumerate(state.gen_jobs):
        if gj["status"] != "completed":
            if verbose:
                print(f"  WARNING: GEN job {gj.get('job_id', '?')} not completed, skipping.")
            continue

        gen_item = cfg.gen_sweep[gj.get("gen_item_index", idx)]
        conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]
        num_gpus = cfg.resources.gen_gpus_per_instance

        # Always parse the decode worker log (not sa-bench JSONs)
        logs_dir = Path(gj.get("output_dir", "")) / "logs"
        log_file = find_decode_log(logs_dir)
        if log_file is None:
            if verbose:
                print(f"  WARNING: No decode log for job {gj.get('job_id', '?')}")
            continue

        data = parse_gen_log(log_file, verbose=False)
        if not data:
            if verbose:
                print(f"  WARNING: No iteration data from {log_file}")
            continue

        if verbose:
            print(f"  Parsed {len(data)} iterations from {log_file.name}")

        # Determine ep_rank for DEP mode
        ep_rank = gen_item.tp_size  # ep_rank = rank_num = tp

        # Process each concurrency from the continuous decode log
        # The exact-match filter on num_scheduled_requests naturally isolates
        # iterations for each concurrency level.
        gj["results"] = []
        for conc in conc_list:
            result = process_gen_data(
                data,
                concurrency=conc,
                mode=gen_item.mode,
                tp=gen_item.tp_size,
                ep_rank=ep_rank,
                mtp=gen_item.mtp_num,
                isl=isl,
                num_gpus=num_gpus,
                verbose=verbose,
            )
            if "error" in result:
                if verbose:
                    print(f"  WARNING: GEN c{conc} processing error: {result['error']}")
                continue

            # Attach config metadata to result
            result["batch_size"] = gen_item.batch_size
            result["max_num_tokens"] = gen_item.max_num_tokens
            result["gpu_memory_fraction"] = gen_item.gpu_memory_fraction
            result["eplb_num_slots"] = gen_item.eplb_num_slots
            result["tp_size"] = gen_item.tp_size

            gj["results"].append(result)
            state.gen_results.append(result)
            if verbose:
                print(
                    f"  {gen_item.mode.upper()} c{conc} mtp{gen_item.mtp_num}: "
                    f"step={result['avg_step_time_ms']:.2f}ms "
                    f"TPOT={result['tpot_ms']:.2f}ms "
                    f"tput/gpu={result.get('throughput_per_gpu', 0):.1f} "
                    f"accept_rate={result['mtp_accept_rate']}"
                )

    state.save()

    if verbose:
        print(f"\n  Processed {len(state.gen_results)}/{len(state.gen_jobs)} GEN results")

    state.phase = "rate_match"
    state.save()


def phase4_rate_matching(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 4: Compute rate-matching metrics for each GEN result."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("PHASE 4: Computing rate-matching allocations")
        print(f"{'=' * 60}")

    osl = cfg.workload.osl
    isl = cfg.workload.isl
    random_ratio = getattr(cfg.workload, 'random_ratio', 1.0)
    state.rate_matching_results = []

    for gen_result in state.gen_results:
        rm = compute_rate_matching(
            ctx_result=state.ctx_result,
            gen_result=gen_result,
            osl=osl,
            random_ratio=random_ratio,
            gpus_per_ctx_instance=cfg.resources.ctx_gpus_per_instance,
            gpus_per_gen_instance=cfg.resources.gen_gpus_per_instance,
            max_total_gpus=getattr(cfg.resources, 'max_total_gpus', 64),
        )

        # Carry forward extra fields
        rm["max_num_tokens"] = gen_result.get("max_num_tokens")
        rm["gpu_memory_fraction"] = gen_result.get("gpu_memory_fraction")
        rm["eplb_num_slots"] = gen_result.get("eplb_num_slots", 0)

        state.rate_matching_results.append(rm)

        if verbose:
            print(
                f"  {rm['config_name']}: "
                f"ratio={rm['ratio_str']} "
                f"gen_req_rate={rm['gen_req_rate']:.2f} req/s "
                f"tput/gpu={rm['output_tput_per_gpu']:.1f} "
                f"ctx:gen={rm['ctx_instances']}:{rm['gen_instances']}"
            )

    state.phase = "pareto"
    state.save()


def phase5_pareto(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 5: Extract Pareto frontier."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("PHASE 5: Extracting Pareto frontier")
        print(f"{'=' * 60}")

    frontier = extract_pareto_frontier(state.rate_matching_results)
    state.pareto_frontier = frontier

    if verbose:
        print(f"  {len(frontier)} Pareto-optimal points from {len(state.rate_matching_results)} configs:")
        for p in frontier:
            print(
                f"    Rank {p['pareto_rank']}: c{p['concurrency']} {p['mode'].upper()} "
                f"mtp{p.get('mtp_num', 0)} "
                f"ratio={p['ratio_str']} "
                f"inter={p['interactivity']:.1f} "
                f"tput/gpu={p['output_tput_per_gpu']:.1f}"
            )

    # Export results
    _export_results(cfg, state, verbose=verbose)

    state.phase = "e2e" if cfg.settings.run_e2e_validation else "complete"
    state.save()


def phase6_e2e_validation(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 6: Generate E2E configs, submit, poll, process, compare SOL vs E2E."""
    if verbose:
        parallel = cfg.settings.parallel_submissions
        mode_str = "parallel" if parallel else "serialised"
        mults = cfg.settings.e2e_validation.concurrency_multipliers
        print(f"\n{'=' * 60}")
        print(f"PHASE 6: E2E validation ({mode_str})")
        print(f"  Concurrency multipliers: {mults}")
        print(f"  Pareto points: {len(state.pareto_frontier)}")
        print(f"  Total E2E jobs: {len(state.pareto_frontier) * len(mults)}")
        print(f"{'=' * 60}")

    # Generate E2E configs if not already done
    if not state.e2e_configs:
        e2e_dir = Path(state.output_dir) / "e2e_pareto_configs"
        state.e2e_configs = generate_e2e_configs_from_pareto(
            cfg, state.pareto_frontier, str(e2e_dir),
        )
        state.save()
        if verbose:
            for ec in state.e2e_configs:
                print(f"  Config: {Path(ec['config_path']).name} "
                      f"(rank {ec['pareto_rank']}, {ec['multiplier']}x, "
                      f"sys_conc={ec['system_concurrency']})")

    if dry_run:
        if verbose:
            print(f"\n  [DRY-RUN] Would submit {len(state.e2e_configs)} E2E jobs")
        return

    # Submit E2E jobs
    if not state.e2e_jobs:
        state.e2e_jobs = [
            {"config_path": ec["config_path"], "status": "pending",
             "pareto_rank": ec["pareto_rank"], "multiplier": ec["multiplier"],
             "per_worker_concurrency": ec["per_worker_concurrency"],
             "system_concurrency": ec["system_concurrency"],
             "config_name": ec["config_name"]}
            for ec in state.e2e_configs
        ]
        state.save()

    pending = [ej for ej in state.e2e_jobs if ej.get("status") not in ("completed",)]
    if not pending and state.sol_vs_e2e:
        if verbose:
            print("  All E2E jobs already completed and processed.")
        return

    if cfg.settings.parallel_submissions:
        # Submit all at once
        for ej in pending:
            if not ej.get("job_id"):
                job_id = submit_job(ej["config_path"], verbose=verbose)
                ej["job_id"] = int(job_id)
                ej["submit_time"] = datetime.now().isoformat()
                ej["status"] = "running"
                ej["output_dir"] = get_job_output_dir(job_id)
        state.save()

        # Poll all
        for ej in pending:
            status = poll_job(str(ej["job_id"]), cfg.settings.poll_interval, verbose=verbose)
            ej["status"] = "completed" if "COMPLETED" in status.upper() else "failed"
            ej["complete_time"] = datetime.now().isoformat()
            state.save()
    else:
        for ej in pending:
            if not ej.get("job_id"):
                job_id = submit_job(ej["config_path"], verbose=verbose)
                ej["job_id"] = int(job_id)
                ej["submit_time"] = datetime.now().isoformat()
                ej["status"] = "running"
                ej["output_dir"] = get_job_output_dir(job_id)
                state.save()

            status = poll_job(str(ej["job_id"]), cfg.settings.poll_interval, verbose=verbose)
            ej["status"] = "completed" if "COMPLETED" in status.upper() else "failed"
            ej["complete_time"] = datetime.now().isoformat()
            state.save()

    # Process E2E results
    _process_e2e_results(cfg, state, verbose=verbose)

    state.phase = "complete"
    state.save()


def _process_e2e_results(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Process E2E job results and compare against SOL predictions."""
    tolerances = {
        "tpot_tolerance_pct": cfg.settings.e2e_validation.tpot_tolerance_pct,
        "throughput_tolerance_pct": cfg.settings.e2e_validation.throughput_tolerance_pct,
    }
    ttft_constraint = cfg.settings.e2e_validation.ttft_constraint_ms

    # Build pareto map for quick lookup
    pareto_map = {p["pareto_rank"]: p for p in state.pareto_frontier}

    state.e2e_results = []
    state.sol_vs_e2e = []

    for ej in state.e2e_jobs:
        if ej.get("status") != "completed":
            continue

        if not ej.get("result"):
            # Load sa-bench result from output dir
            output_dir = ej.get("output_dir", get_job_output_dir(ej.get("job_id", "")))
            sa_result = _load_sa_bench_result(output_dir)
            if sa_result is None:
                if verbose:
                    print(f"  WARNING: No sa-bench result for E2E job {ej.get('job_id', '?')}")
                continue
            ej["result"] = sa_result

        sa_result = ej["result"]
        pareto_rank = ej["pareto_rank"]
        multiplier = ej["multiplier"]
        pp = pareto_map.get(pareto_rank)
        if pp is None:
            continue

        gen_instances = pp["gen_instances"]
        ctx_instances = pp["ctx_instances"]
        total_gpus = pp["total_gpus"]
        per_worker_conc = pp["concurrency"]
        system_conc = ej["system_concurrency"]

        if verbose:
            print(f"\n  Processing E2E (rank {pareto_rank}, {multiplier}x)...")
            print(f"    System concurrency: {system_conc} "
                  f"(per-worker {per_worker_conc} x {gen_instances} workers x {multiplier})")

        comparison = compare_sol_vs_e2e(
            sol_prediction=pp,
            e2e_result=sa_result,
            total_gpus=total_gpus,
            gen_instances=gen_instances,
            per_worker_conc=per_worker_conc,
            tolerances=tolerances,
        )

        # TTFT
        e2e_ttft_ms = sa_result.get("median_ttft_ms", 0)
        if e2e_ttft_ms == 0:
            ttft_percentiles = sa_result.get("ttft_percentiles", {})
            e2e_ttft_ms = ttft_percentiles.get("p50", 0)
        ttft_pass = True
        if ttft_constraint is not None and e2e_ttft_ms > 0:
            ttft_pass = e2e_ttft_ms <= ttft_constraint

        entry = {
            "pareto_rank": pareto_rank,
            "multiplier": multiplier,
            "per_worker_concurrency": per_worker_conc,
            "system_concurrency": system_conc,
            "e2e_ttft_ms": e2e_ttft_ms,
            "ttft_constraint_ms": ttft_constraint,
            "ttft_pass": ttft_pass,
            "config_name": ej.get("config_name", ""),
            **comparison,
        }
        state.sol_vs_e2e.append(entry)
        state.e2e_results.append({"pareto_rank": pareto_rank, "multiplier": multiplier, **sa_result})

        if verbose:
            print(f"    TPOT diff: {comparison['diff_pct']['tpot']:+.1f}%")
            print(f"    Throughput diff: {comparison['diff_pct']['output_tput_per_gpu']:+.1f}%")
            if ttft_constraint is not None:
                status = "PASS" if ttft_pass else "FAIL"
                print(f"    TTFT: {e2e_ttft_ms:.0f}ms ({status}, constraint: {ttft_constraint}ms)")

    state.save()


def _load_sa_bench_result(output_dir: str) -> dict | None:
    """Load the sa-bench JSON result from a job output directory.

    When multiple concurrencies were run, returns the last (highest) one.
    """
    logs_dir = Path(output_dir) / "logs"
    if not logs_dir.exists():
        return None

    result_files = list(logs_dir.glob("sa-bench_*/results_*.json"))
    if not result_files:
        result_files = list(logs_dir.glob("**/results_*.json"))
    if not result_files:
        return None

    result_files.sort()
    try:
        with open(result_files[-1]) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_results(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Export rate-matching results and Pareto frontier to CSV/JSON."""
    results_dir = Path(state.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prefix = state.sweep_name

    # All results JSON
    with open(results_dir / f"{prefix}_all.json", "w") as f:
        json.dump(state.rate_matching_results, f, indent=2)

    # Frontier JSON
    with open(results_dir / f"{prefix}_frontier.json", "w") as f:
        json.dump(state.pareto_frontier, f, indent=2)

    # CSV exports
    try:
        import csv
        cols = [
            "config_name", "mode", "batch_size", "concurrency", "mtp_num",
            "mtp_accept_rate", "avg_step_time_ms",
            "interactivity", "tpot_ms", "output_tput_per_gpu",
            "gen_req_rate", "ctx_request_rate", "ctx_gen_inst_ratio",
            "ctx_instances", "gen_instances", "total_gpus", "ratio_str",
        ]

        for suffix, data in [("all", state.rate_matching_results), ("frontier", state.pareto_frontier)]:
            with open(results_dir / f"{prefix}_{suffix}.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
    except Exception as e:
        if verbose:
            print(f"  CSV export warning: {e}")

    if verbose:
        print(f"  Results exported to: {results_dir}")


# ---------------------------------------------------------------------------
# Phase 7: Dashboard / summary
# ---------------------------------------------------------------------------

def phase7_summary(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 7: Generate charts, print final summary."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("SWEEP COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {state.output_dir}")
        print(f"  - Configs: {state.output_dir}/configs/")
        print(f"  - Results: {state.output_dir}/results/")
        if state.e2e_configs:
            print(f"  - E2E Pareto: {state.output_dir}/e2e_pareto_configs/")
        print(f"  - State: {state.output_dir}/sweep_state.json")

    # Generate charts
    try:
        from dashboard_export import generate_charts, export_rate_matching_results

        dashboard_dir = Path(state.output_dir) / "dashboard"
        export_state = {
            "sweep_name": state.sweep_name,
            "rate_matching_results": state.rate_matching_results,
            "pareto_frontier": state.pareto_frontier,
            "sol_vs_e2e": state.sol_vs_e2e,
            "ctx_result": state.ctx_result,
        }
        export_rate_matching_results(export_state, str(dashboard_dir))
        charts = generate_charts(export_state, str(dashboard_dir))
        if charts and verbose:
            print(f"\nInteractive charts (open in browser):")
            for chart_path in charts:
                print(f"  file://{chart_path}")
    except Exception as e:
        if verbose:
            print(f"\n  (Dashboard export skipped: {e})")

    # TTFT summary
    if state.sol_vs_e2e and verbose:
        ttft_constraint = cfg.settings.e2e_validation.ttft_constraint_ms
        if ttft_constraint:
            ttft_results = [e for e in state.sol_vs_e2e if e.get("e2e_ttft_ms", 0) > 0]
            ttft_pass = sum(1 for e in ttft_results if e.get("ttft_pass", False))
            print(f"\nTTFT CONSTRAINT CHECK (soft limit: {ttft_constraint}ms)")
            if ttft_pass == len(ttft_results):
                print(f"  All {len(ttft_results)} E2E points meet the TTFT constraint.")
            else:
                print(f"  {ttft_pass}/{len(ttft_results)} E2E points meet the TTFT constraint.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(
    config_path: str,
    output_dir: str,
    resume: bool = False,
    skip_e2e: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> SweepState:
    """Run the full rate-matching sweep pipeline.

    Args:
        config_path: Path to sweep YAML config
        output_dir: Output directory for configs, results, state
        resume: If True, resume from saved state
        skip_e2e: If True, skip E2E validation phase
        dry_run: If True, generate configs only (no SLURM submission)
        verbose: Print progress

    Returns:
        Final SweepState
    """
    # Load config
    cfg = load_sweep_config(config_path)

    # Create or resume state
    state_path = Path(output_dir) / "sweep_state.json"
    if resume and state_path.exists():
        state = SweepState.load(str(state_path))
        if verbose:
            print(f"Resuming sweep from phase: {state.phase}")
    else:
        state = SweepState()
        state.sweep_name = cfg.name
        state.sweep_config_path = str(Path(config_path).resolve())
        state.output_dir = str(Path(output_dir).resolve())
        state.created_at = datetime.now().isoformat()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run phases in order, skipping completed ones
    if state.phase in ("init", "ctx"):
        if state.phase == "init":
            phase1_generate_configs(cfg, state, verbose=verbose)

    if state.phase == "ctx":
        phase2_ctx_sol(cfg, state, dry_run=dry_run, verbose=verbose)

    if state.phase == "gen":
        phase3_gen_sol(cfg, state, dry_run=dry_run, verbose=verbose)

    if state.phase == "rate_match":
        phase4_rate_matching(cfg, state, verbose=verbose)

    if state.phase == "pareto":
        phase5_pareto(cfg, state, verbose=verbose)

    if state.phase in ("e2e",) and not skip_e2e and cfg.settings.run_e2e_validation:
        phase6_e2e_validation(cfg, state, dry_run=dry_run, verbose=verbose)

    phase7_summary(cfg, state, verbose=verbose)

    if not dry_run:
        state.phase = "complete"
        state.save()

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rate-matching sweep orchestrator")
    parser.add_argument("-c", "--config", required=True, help="Sweep config YAML")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E validation")
    parser.add_argument("--dry-run", action="store_true", help="Config generation only")
    args = parser.parse_args()

    run_sweep(
        config_path=args.config,
        output_dir=args.output,
        resume=args.resume,
        skip_e2e=args.skip_e2e,
        dry_run=args.dry_run,
    )
