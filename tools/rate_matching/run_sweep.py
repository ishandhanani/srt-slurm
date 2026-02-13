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

# Force line-buffered stdout so output appears immediately in non-TTY contexts
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

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
from parser_base import CTXLogParser, GENLogParser
from process_ctx_results import TrtllmCTXLogParser
from process_gen_results import TrtllmGENLogParser
from schema import RateMatchingSweepConfig, load_sweep_config

# Engine-specific parsers.  Swap these for vLLM/SGLang support.
_ctx_parser: CTXLogParser = TrtllmCTXLogParser()
_gen_parser: GENLogParser = TrtllmGENLogParser()


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


def _submit_and_poll(
    job_dict: dict,
    config_path: str,
    poll_interval: int,
    max_retries: int,
    state: "SweepState",
    verbose: bool = True,
) -> str:
    """Submit a SLURM job, poll to completion, and retry on failure.

    This is the common submit -> poll -> retry loop used by CTX, GEN, and
    E2E phases.  On failure the job is resubmitted up to *max_retries* times
    (total attempts = 1 + max_retries).  Each attempt's metadata is appended
    to ``job_dict["retry_history"]``.

    Args:
        job_dict: Mutable dict from ``state.ctx_job``, ``state.gen_jobs[i]``,
            or ``state.e2e_jobs[i]``.  Updated in-place with ``job_id``,
            ``status``, ``output_dir``, timestamps, and ``retry_history``.
        config_path: Path to the srt-slurm YAML config to submit.
        poll_interval: Seconds between ``squeue`` polls.
        max_retries: Maximum number of *retries* (not including the first
            attempt).
        state: The parent ``SweepState`` -- ``state.save()`` is called after
            each status change so progress is durable.
        verbose: Print progress.

    Returns:
        Final SLURM status string (e.g. ``"COMPLETED"``).

    Raises:
        RuntimeError: If the job fails after all retries are exhausted.
    """
    if "retry_history" not in job_dict:
        job_dict["retry_history"] = []

    attempts = 0
    max_attempts = 1 + max_retries

    while attempts < max_attempts:
        attempts += 1
        attempt_label = f"(attempt {attempts}/{max_attempts})" if max_attempts > 1 else ""

        # Submit (skip if already running from a prior resume)
        if not job_dict.get("job_id") or job_dict.get("status") in ("failed", "pending"):
            try:
                job_id = submit_job(config_path, verbose=verbose)
            except RuntimeError as exc:
                if verbose:
                    print(f"  Submit failed {attempt_label}: {exc}")
                job_dict["retry_history"].append({
                    "attempt": attempts,
                    "event": "submit_failed",
                    "error": str(exc),
                    "time": datetime.now().isoformat(),
                })
                state.save()
                if attempts < max_attempts:
                    if verbose:
                        print(f"  Retrying in {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue
                raise RuntimeError(
                    f"Job submission failed after {attempts} attempt(s): {exc}"
                ) from exc

            job_dict["job_id"] = int(job_id)
            job_dict["submit_time"] = datetime.now().isoformat()
            job_dict["status"] = "running"
            job_dict["output_dir"] = get_job_output_dir(job_id)
            job_dict["retry_history"].append({
                "attempt": attempts,
                "event": "submitted",
                "job_id": int(job_id),
                "time": datetime.now().isoformat(),
            })
            state.save()

        # Poll
        status = poll_job(str(job_dict["job_id"]), poll_interval, verbose=verbose)
        is_success = "COMPLETED" in status.upper()
        job_dict["status"] = "completed" if is_success else "failed"
        job_dict["complete_time"] = datetime.now().isoformat()

        job_dict["retry_history"].append({
            "attempt": attempts,
            "event": "completed" if is_success else "failed",
            "job_id": job_dict["job_id"],
            "slurm_status": status,
            "time": datetime.now().isoformat(),
        })
        state.save()

        if is_success:
            return status

        # Failed -- retry?
        if attempts < max_attempts:
            if verbose:
                print(
                    f"  Job {job_dict['job_id']} failed ({status}) {attempt_label}. "
                    f"Retrying..."
                )
            # Clear job_id so the next iteration submits a fresh job
            job_dict["job_id"] = None
            job_dict["status"] = "pending"
            state.save()
        else:
            raise RuntimeError(
                f"Job failed with status {status} after {attempts} attempt(s). "
                f"Config: {config_path}"
            )

    # Should not reach here, but just in case
    return status  # type: ignore[possibly-undefined]


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
    ctx_config_dict = generate_ctx_sol_config(cfg, output_path=ctx_path)

    # Extract the effective max_batch_size from the generated config.
    # This is the ground truth (resolves YAML overrides + workload defaults)
    # and is used later as the num_ctx_requests threshold when processing
    # CTX logs, ensuring the filter is GPU-agnostic.
    ctx_max_batch_size = ctx_config_dict["backend"]["trtllm_config"]["prefill"]["max_batch_size"]

    state.ctx_job = {
        "config_path": ctx_path,
        "status": "pending",
        "max_batch_size": ctx_max_batch_size,
    }
    if verbose:
        print(f"  CTX SOL config: {ctx_path}")
        print(f"  CTX max_batch_size (threshold): {ctx_max_batch_size}")

    # GEN SOL -- one SLURM job per concurrency level.
    #
    # WHY: The TRT-LLM decode worker reads TLLM_BENCHMARK_REQ_QUEUES_SIZE
    # once at startup to set the minimum number of requests queued before
    # processing begins. This value CANNOT be changed without restarting
    # the model. If multiple concurrencies share a single job, the queue
    # depth is wrong for all but one, producing inaccurate step_time
    # measurements and corrupting the SOL throughput calculation.
    #
    # By splitting into one job per concurrency we guarantee:
    #   1. TLLM_BENCHMARK_REQ_QUEUES_SIZE == concurrency (exact match)
    #   2. A clean model launch with no carry-over state between runs
    #   3. Methodologically correct SOL measurements at every point
    from schema import GenSweepItem as _GSI

    state.gen_jobs = []
    for i, gen_item in enumerate(cfg.gen_sweep):
        conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]
        mtp_suffix = f"_mtp{gen_item.mtp_num}" if gen_item.mtp_num > 0 else ""

        for conc in conc_list:
            single_item = _GSI(
                mode=gen_item.mode,
                batch_size=gen_item.batch_size,
                concurrency=conc,
                tp_size=gen_item.tp_size,
                mtp_num=gen_item.mtp_num,
                max_num_tokens=gen_item.max_num_tokens,
                gpu_memory_fraction=gen_item.gpu_memory_fraction,
                eplb_num_slots=gen_item.eplb_num_slots,
            )
            fname = f"gen_sol_{gen_item.mode}_c{conc}{mtp_suffix}.yaml"
            gen_path = str(configs_dir / fname)
            generate_gen_sol_config(cfg, single_item, output_path=gen_path)
            state.gen_jobs.append({
                "config_path": gen_path,
                "status": "pending",
                "gen_item_index": i,
                "concurrency": conc,
            })
            if verbose:
                print(f"  GEN SOL config: {fname}  (queue_size={conc})")

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

    # Submit, poll, and retry on failure
    _submit_and_poll(
        job_dict=ctx_job,
        config_path=ctx_job["config_path"],
        poll_interval=cfg.settings.poll_interval,
        max_retries=cfg.settings.max_retries,
        state=state,
        verbose=verbose,
    )

    # Process results
    if verbose:
        print("  Processing CTX results...")
    logs_dir = Path(ctx_job["output_dir"]) / "logs"
    log_file = _ctx_parser.find_log(logs_dir)
    if log_file is None:
        raise RuntimeError(f"No prefill log found in {logs_dir}")

    data = _ctx_parser.parse(log_file, verbose=False)
    if not data:
        raise RuntimeError(f"No data parsed from {log_file}")

    # Pass the effective max_batch_size from the generated config so the
    # num_ctx_requests threshold matches the actual deployed batch size
    # (GPU-agnostic -- works for H200, GB200, etc.).
    ctx_mbs = ctx_job.get("max_batch_size")
    ctx_result = _ctx_parser.process(
        data, isl=cfg.workload.isl, verbose=False, max_batch_size=ctx_mbs,
    )
    if "error" in ctx_result:
        raise RuntimeError(f"CTX processing error: {ctx_result['error']}")

    state.ctx_result = ctx_result
    if verbose:
        print(f"  CTX request rate: {ctx_result['request_rate_req_per_s']:.4f} req/s")
        print(f"  CTX throughput:   {ctx_result['ctx_throughput_tokens_per_s']:,.0f} tok/s")
        print(f"  Threshold used:   num_ctx_requests >= {ctx_result.get('threshold_used', '?')}")

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

    max_retries = cfg.settings.max_retries

    if cfg.settings.parallel_submissions:
        # Parallel: submit all at once, poll all, then retry failures
        for attempt_round in range(1 + max_retries):
            to_submit = [gj for gj in pending if gj.get("status") not in ("completed",)]
            if not to_submit:
                break
            if attempt_round > 0 and verbose:
                print(f"  Retry round {attempt_round}/{max_retries} for {len(to_submit)} failed job(s)")

            for gj in to_submit:
                if not gj.get("job_id") or gj.get("status") in ("failed", "pending"):
                    try:
                        job_id = submit_job(gj["config_path"], verbose=verbose)
                        gj["job_id"] = int(job_id)
                        gj["submit_time"] = datetime.now().isoformat()
                        gj["status"] = "running"
                        gj["output_dir"] = get_job_output_dir(job_id)
                        gj.setdefault("retry_history", []).append({
                            "attempt": attempt_round + 1,
                            "event": "submitted",
                            "job_id": int(job_id),
                            "time": datetime.now().isoformat(),
                        })
                    except RuntimeError as exc:
                        if verbose:
                            print(f"  Submit failed: {exc}")
                        gj.setdefault("retry_history", []).append({
                            "attempt": attempt_round + 1,
                            "event": "submit_failed",
                            "error": str(exc),
                            "time": datetime.now().isoformat(),
                        })
                        gj["status"] = "failed"
            state.save()

            for gj in to_submit:
                if gj.get("status") != "running":
                    continue
                status = poll_job(str(gj["job_id"]), cfg.settings.poll_interval, verbose=verbose)
                is_ok = "COMPLETED" in status.upper()
                gj["status"] = "completed" if is_ok else "failed"
                gj["complete_time"] = datetime.now().isoformat()
                gj.setdefault("retry_history", []).append({
                    "attempt": attempt_round + 1,
                    "event": "completed" if is_ok else "failed",
                    "job_id": gj["job_id"],
                    "slurm_status": status,
                    "time": datetime.now().isoformat(),
                })
                if not is_ok:
                    gj["job_id"] = None  # clear so next round resubmits
                state.save()
    else:
        # Serial: submit, poll, retry per job
        for gj in pending:
            try:
                _submit_and_poll(
                    job_dict=gj,
                    config_path=gj["config_path"],
                    poll_interval=cfg.settings.poll_interval,
                    max_retries=max_retries,
                    state=state,
                    verbose=verbose,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ERROR: GEN job exhausted retries: {exc}")
                # Continue to other GEN jobs -- the failed one stays "failed"

    # Process results for all completed GEN jobs
    # METHODOLOGY: Always use decode worker logs (prev_device_step_time), NOT
    # sa-bench client-side JSONs.  Decode logs give the true device step time,
    # which combined with hardcoded MTP accept rates yields SOL throughput.
    # sa-bench JSONs capture E2E client metrics (including network overhead)
    # and are NOT used for GEN SOL.
    #
    # Multi-concurrency handling: when sa-bench runs concurrencies "8x32x64"
    # sequentially, the decode log is continuous.  The exact-match filter on
    # num_scheduled_requests (Step 4) naturally segments iterations by
    # concurrency.
    state.gen_results = []
    isl = cfg.workload.isl
    mtp_overrides = getattr(cfg.workload, 'mtp_accept_rates', None)
    for idx, gj in enumerate(state.gen_jobs):
        if gj["status"] != "completed":
            if verbose:
                print(f"  WARNING: GEN job {gj.get('job_id', '?')} not completed, skipping.")
            continue

        gen_item = cfg.gen_sweep[gj.get("gen_item_index", idx)]
        # If the job was split per-concurrency, use the stored single value.
        # Otherwise, use the sweep item's full concurrency list.
        if "concurrency" in gj:
            conc_list = [gj["concurrency"]]
        else:
            conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]
        num_gpus = cfg.resources.gen_gpus_per_instance

        # Always parse the decode worker log (not sa-bench JSONs)
        logs_dir = Path(gj.get("output_dir", "")) / "logs"
        log_file = _gen_parser.find_log(logs_dir)
        if log_file is None:
            if verbose:
                print(f"  WARNING: No decode log for job {gj.get('job_id', '?')}")
            continue

        data = _gen_parser.parse(log_file, verbose=False)
        if not data:
            if verbose:
                print(f"  WARNING: No iteration data from {log_file}")
            continue

        if verbose:
            print(f"  Parsed {len(data)} iterations from {log_file.name}")

        # Determine ep_rank for DEP mode
        ep_rank = gen_item.tp_size  # ep_rank = rank_num = tp

        # Process each concurrency from the decode log.
        # When split per-concurrency, there's exactly one.
        # When multi-concurrency, the exact-match filter on
        # num_scheduled_requests isolates each level.
        gj["results"] = []
        for conc in conc_list:
            result = _gen_parser.process(
                data,
                concurrency=conc,
                mode=gen_item.mode,
                tp=gen_item.tp_size,
                ep_rank=ep_rank,
                mtp=gen_item.mtp_num,
                isl=isl,
                num_gpus=num_gpus,
                verbose=verbose,
                mtp_accept_rate_overrides=mtp_overrides,
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
            isl=isl,
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

    max_retries = cfg.settings.max_retries

    if cfg.settings.parallel_submissions:
        # Parallel: submit all, poll all, retry failures
        for attempt_round in range(1 + max_retries):
            to_submit = [ej for ej in pending if ej.get("status") not in ("completed",)]
            if not to_submit:
                break
            if attempt_round > 0 and verbose:
                print(f"  Retry round {attempt_round}/{max_retries} for {len(to_submit)} failed E2E job(s)")

            for ej in to_submit:
                if not ej.get("job_id") or ej.get("status") in ("failed", "pending"):
                    try:
                        job_id = submit_job(ej["config_path"], verbose=verbose)
                        ej["job_id"] = int(job_id)
                        ej["submit_time"] = datetime.now().isoformat()
                        ej["status"] = "running"
                        ej["output_dir"] = get_job_output_dir(job_id)
                        ej.setdefault("retry_history", []).append({
                            "attempt": attempt_round + 1,
                            "event": "submitted",
                            "job_id": int(job_id),
                            "time": datetime.now().isoformat(),
                        })
                    except RuntimeError as exc:
                        if verbose:
                            print(f"  E2E submit failed: {exc}")
                        ej.setdefault("retry_history", []).append({
                            "attempt": attempt_round + 1,
                            "event": "submit_failed",
                            "error": str(exc),
                            "time": datetime.now().isoformat(),
                        })
                        ej["status"] = "failed"
            state.save()

            for ej in to_submit:
                if ej.get("status") != "running":
                    continue
                status = poll_job(str(ej["job_id"]), cfg.settings.poll_interval, verbose=verbose)
                is_ok = "COMPLETED" in status.upper()
                ej["status"] = "completed" if is_ok else "failed"
                ej["complete_time"] = datetime.now().isoformat()
                ej.setdefault("retry_history", []).append({
                    "attempt": attempt_round + 1,
                    "event": "completed" if is_ok else "failed",
                    "job_id": ej["job_id"],
                    "slurm_status": status,
                    "time": datetime.now().isoformat(),
                })
                if not is_ok:
                    ej["job_id"] = None  # clear so next round resubmits
                state.save()
    else:
        # Serial: submit, poll, retry per job
        for ej in pending:
            try:
                _submit_and_poll(
                    job_dict=ej,
                    config_path=ej["config_path"],
                    poll_interval=cfg.settings.poll_interval,
                    max_retries=max_retries,
                    state=state,
                    verbose=verbose,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ERROR: E2E job exhausted retries: {exc}")
                # Continue to other E2E jobs

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
            "output_tput_per_gen_gpu", "total_throughput", "total_tput_per_gpu",
            "gen_req_rate", "ctx_request_rate", "ctx_gen_inst_ratio",
            "ctx_instances", "gen_instances", "total_gpus", "ratio_str",
            "estimate_e2e_latency_s",
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
# Reprocess (re-derive everything from existing logs, no SLURM submission)
# ---------------------------------------------------------------------------

def reprocess_sweep(
    output_dir: str,
    config_path: str | None = None,
    skip_e2e: bool = False,
    verbose: bool = True,
) -> SweepState:
    """Re-process an existing sweep from log files without resubmitting jobs.

    This is useful when:
      - You've changed parameters in the sweep YAML (e.g., mtp_accept_rates,
        random_ratio) and want to re-derive metrics.
      - You've fixed a bug in processing logic and need to recompute.
      - You want to regenerate dashboards/exports.

    The function re-reads raw TRT-LLM logs from each job's output_dir and
    re-runs phases 2 (CTX processing) -> 3 (GEN processing) -> 4 (rate-match)
    -> 5 (Pareto) -> 6 (E2E comparison, if applicable) -> 7 (dashboard).

    No SLURM jobs are submitted.

    Args:
        output_dir: Sweep output directory containing sweep_state.json.
        config_path: Optional path to an updated sweep YAML config. If None,
            the original config from the sweep state is used.
        skip_e2e: If True, skip E2E re-processing even if E2E results exist.
        verbose: Print progress.

    Returns:
        Updated SweepState.
    """
    # Load state
    state_path = Path(output_dir) / "sweep_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No sweep state found at {state_path}")
    state = SweepState.load(str(state_path))

    # Load config (allow override)
    cfg_path = config_path or state.sweep_config_path
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Sweep config not found: {cfg_path}")
    cfg = load_sweep_config(cfg_path)

    if verbose:
        print(f"{'=' * 60}")
        print(f"REPROCESS: {state.sweep_name}")
        print(f"{'=' * 60}")
        print(f"  Output dir: {output_dir}")
        print(f"  Config:     {cfg_path}")
        if config_path:
            print(f"  (using updated config)")

    # --- Re-run Phase 2: CTX processing (no submission) ---
    if verbose:
        print(f"\n{'=' * 60}")
        print("REPROCESS Phase 2: Re-processing CTX results from logs")
        print(f"{'=' * 60}")

    ctx_job = state.ctx_job
    if ctx_job.get("status") == "completed" and ctx_job.get("output_dir"):
        logs_dir = Path(ctx_job["output_dir"]) / "logs"
        log_file = _ctx_parser.find_log(logs_dir)
        if log_file is None:
            raise RuntimeError(f"No prefill log found in {logs_dir}")

        data = _ctx_parser.parse(log_file, verbose=False)
        if not data:
            raise RuntimeError(f"No data parsed from {log_file}")

        ctx_mbs = ctx_job.get("max_batch_size")
        ctx_result = _ctx_parser.process(
            data, isl=cfg.workload.isl, verbose=False, max_batch_size=ctx_mbs,
        )
        if "error" in ctx_result:
            raise RuntimeError(f"CTX processing error: {ctx_result['error']}")

        state.ctx_result = ctx_result
        if verbose:
            print(f"  CTX request rate: {ctx_result['request_rate_req_per_s']:.4f} req/s")
            print(f"  CTX throughput:   {ctx_result['ctx_throughput_tokens_per_s']:,.0f} tok/s")
            print(f"  Threshold used:   num_ctx_requests >= {ctx_result.get('threshold_used', '?')}")
    else:
        if verbose:
            print("  WARNING: CTX job not completed, keeping existing ctx_result")

    # --- Re-run Phase 3: GEN processing (no submission) ---
    if verbose:
        print(f"\n{'=' * 60}")
        print("REPROCESS Phase 3: Re-processing GEN results from logs")
        print(f"{'=' * 60}")

    isl = cfg.workload.isl
    mtp_overrides = getattr(cfg.workload, 'mtp_accept_rates', None)
    num_gpus = cfg.resources.gen_gpus_per_instance
    state.gen_results = []

    for idx, gj in enumerate(state.gen_jobs):
        if gj["status"] != "completed":
            if verbose:
                print(f"  WARNING: GEN job {gj.get('job_id', '?')} not completed, skipping.")
            continue

        # Re-read gen_item from config using the stored index
        gen_item = cfg.gen_sweep[gj.get("gen_item_index", idx)]
        if "concurrency" in gj:
            conc_list = [gj["concurrency"]]
        else:
            conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]

        logs_dir = Path(gj.get("output_dir", "")) / "logs"
        log_file = _gen_parser.find_log(logs_dir)
        if log_file is None:
            if verbose:
                print(f"  WARNING: No decode log in {logs_dir}, skipping GEN job {idx}")
            continue

        data = _gen_parser.parse(log_file, verbose=False)
        if not data:
            if verbose:
                print(f"  WARNING: No data parsed from {log_file}")
            continue

        if verbose:
            print(f"  Parsed {len(data)} iterations from {log_file.name}")

        ep_rank = gen_item.tp_size  # ep_rank = rank_num = tp

        for conc in conc_list:
            result = _gen_parser.process(
                data,
                concurrency=conc,
                mode=gen_item.mode,
                tp=gen_item.tp_size,
                ep_rank=ep_rank,
                mtp=gen_item.mtp_num,
                isl=isl,
                num_gpus=num_gpus,
                verbose=False,
                mtp_accept_rate_overrides=mtp_overrides,
            )
            if "error" in result:
                if verbose:
                    print(f"  WARNING: GEN c{conc} processing error: {result['error']}")
                continue

            result["batch_size"] = gen_item.batch_size
            result["max_num_tokens"] = gen_item.max_num_tokens
            result["gpu_memory_fraction"] = gen_item.gpu_memory_fraction
            result["eplb_num_slots"] = gen_item.eplb_num_slots
            result["tp_size"] = gen_item.tp_size

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
        print(f"\n  Re-processed {len(state.gen_results)} GEN results")

    # --- Re-run Phase 4: Rate matching ---
    phase4_rate_matching(cfg, state, verbose=verbose)

    # --- Re-run Phase 5: Pareto ---
    phase5_pareto(cfg, state, verbose=verbose)

    # Override phase transition from phase5 (which sets "e2e" or "complete").
    # If reprocess crashes after this point, a subsequent `run_sweep --resume`
    # should NOT try to submit new E2E jobs.  Setting to "complete" is safe
    # because run_sweep's phase checks use `==` (not `in`), and "complete"
    # only triggers phase7_summary (always called anyway).
    state.phase = "complete"
    state.save()

    # --- Re-run Phase 6: E2E comparison (from existing results, no submission) ---
    if not skip_e2e and state.e2e_jobs:
        completed_e2e = [ej for ej in state.e2e_jobs if ej.get("status") == "completed"]
        if completed_e2e:
            if verbose:
                print(f"\n{'=' * 60}")
                print("REPROCESS Phase 6: Re-processing E2E results")
                print(f"{'=' * 60}")
            _process_e2e_results(cfg, state, verbose=verbose)

    # --- Re-run Phase 7: Dashboard ---
    phase7_summary(cfg, state, verbose=verbose)

    state.phase = "complete"
    state.save()

    if verbose:
        print(f"\nReprocessing complete. State saved to: {state_path}")

    return state


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
