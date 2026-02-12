#!/usr/bin/env python3
"""
Human-readable status dashboard for a rate-matching sweep.

Reads sweep_state.json and prints progress tables for each phase,
including job status, Pareto frontier, and E2E validation results.

Usage:
    srtctl-rate-match status -o /path/to/sweep/output
    srtctl-rate-match status -o /path/to/sweep/output --live
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def load_state(output_dir: str) -> dict:
    """Load sweep_state.json from output directory."""
    state_path = Path(output_dir) / "sweep_state.json"
    if not state_path.exists():
        print(f"Error: No sweep_state.json found in {output_dir}")
        sys.exit(1)
    with open(state_path) as f:
        return json.load(f)


def _slurm_status(job_id: str | int) -> str:
    """Query current SLURM status for a job."""
    try:
        result = subprocess.run(
            ["squeue", "--job", str(job_id), "--noheader", "--format=%T"],
            capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip()
        if status:
            return status
    except Exception:
        pass

    # Try sacct
    try:
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return "UNKNOWN"


def print_status(output_dir: str, live: bool = False) -> None:
    """Print the sweep status dashboard."""
    state = load_state(output_dir)

    print(f"\n{'=' * 72}")
    print(f"  SWEEP: {state.get('sweep_name', 'unknown')}")
    print(f"  Phase: {state.get('phase', 'unknown').upper()}")
    print(f"  Output: {state.get('output_dir', output_dir)}")
    print(f"  Created: {state.get('created_at', '?')}")
    print(f"  Updated: {state.get('last_updated', '?')}")
    print(f"{'=' * 72}")

    # CTX SOL
    ctx = state.get("ctx_job", {})
    if ctx:
        slurm_status = _slurm_status(ctx.get("job_id", "")) if live and ctx.get("job_id") else ""
        status = slurm_status or ctx.get("status", "?")
        print(f"\n  CTX-ONLY SOL")
        print(f"  {'Job ID':<12} {'Status':<12} {'Config'}")
        print(f"  {'-' * 50}")
        job_id = ctx.get("job_id", "-")
        config = Path(ctx.get("config_path", "?")).name
        print(f"  {str(job_id):<12} {status:<12} {config}")

    ctx_result = state.get("ctx_result", {})
    if ctx_result:
        print(f"\n  CTX Results: request_rate={ctx_result.get('request_rate_req_per_s', 0):.4f} req/s  "
              f"throughput={ctx_result.get('ctx_throughput_tokens_per_s', 0):,.0f} tok/s")

    # GEN SOL
    gen_jobs = state.get("gen_jobs", [])
    if gen_jobs:
        print(f"\n  GEN-ONLY SOL ({len(gen_jobs)} jobs)")
        print(f"  {'Job ID':<10} {'Status':<12} {'Mode':<5} {'Conc':<6} {'MTP':<4} "
              f"{'TPOT(ms)':<10} {'Inter':<8} {'Tput/GPU':<10} {'Config'}")
        print(f"  {'-' * 90}")
        for gj in gen_jobs:
            job_id = gj.get("job_id", "-")
            slurm_status = _slurm_status(job_id) if live and job_id != "-" else ""
            status = slurm_status or gj.get("status", "?")
            config = Path(gj.get("config_path", "?")).name
            r = gj.get("result", {})
            tpot = f"{r['tpot_ms']:.2f}" if r.get("tpot_ms") else "-"
            inter = f"{r['interactivity']:.1f}" if r.get("interactivity") else "-"
            tput = f"{r['throughput_per_gpu']:.1f}" if r.get("throughput_per_gpu") else "-"
            mode = r.get("mode", "?")
            conc = r.get("concurrency", "?")
            mtp = r.get("mtp", "?")
            print(f"  {str(job_id):<10} {status:<12} {mode:<5} {str(conc):<6} {str(mtp):<4} "
                  f"{tpot:<10} {inter:<8} {tput:<10} {config}")

    # Pareto frontier
    pareto = state.get("pareto_frontier", [])
    if pareto:
        print(f"\n  PARETO FRONTIER ({len(pareto)} points)")
        print(f"  {'Rank':<5} {'Config':<25} {'Mode':<5} {'PW Conc':<8} {'MTP':<4} "
              f"{'Ratio':<8} {'Inter':<8} {'Tput/GPU':<10} {'GPUs':<6} {'Eff%'}")
        print(f"  {'-' * 100}")
        for p in pareto:
            print(
                f"  {p.get('pareto_rank', '?'):<5} "
                f"{p.get('config_name', '?'):<25} "
                f"{p.get('mode', '?'):<5} "
                f"{str(p.get('concurrency', '?')):<8} "
                f"{str(p.get('mtp_num', 0)):<4} "
                f"{p.get('ratio_str', '?'):<8} "
                f"{p.get('interactivity', 0):<8.1f} "
                f"{p.get('output_tput_per_gpu', 0):<10.1f} "
                f"{str(p.get('total_gpus', '?')):<6} "
                f"{p.get('efficiency_pct', 0):.1f}"
            )

    # E2E Validation
    sol_vs_e2e = state.get("sol_vs_e2e", [])
    if sol_vs_e2e:
        print(f"\n  E2E VALIDATION ({len(sol_vs_e2e)} results)")
        print(f"  {'Rank':<5} {'Mult':<6} {'PW Conc':<8} {'Sys Conc':<9} "
              f"{'SOL TPOT':<10} {'E2E TPOT':<10} {'TPOT Diff':<10} "
              f"{'SOL Tput':<10} {'E2E Tput':<10} {'Tput Diff':<10} "
              f"{'TTFT(ms)':<10} {'TTFT':<6} {'Pass'}")
        print(f"  {'-' * 120}")
        for e in sol_vs_e2e:
            sol = e.get("sol", {})
            e2e = e.get("e2e", {})
            diff = e.get("diff_pct", {})
            pass_info = e.get("pass", {})
            ttft_ms = e.get("e2e_ttft_ms", 0)
            ttft_status = "PASS" if e.get("ttft_pass", True) else "FAIL"
            overall = "PASS" if pass_info.get("overall", True) else "FAIL"
            print(
                f"  {e.get('pareto_rank', '?'):<5} "
                f"{e.get('multiplier', 1.0):<6.2f} "
                f"{str(e.get('per_worker_concurrency', '?')):<8} "
                f"{str(e.get('system_concurrency', '?')):<9} "
                f"{sol.get('tpot_ms', 0):<10.2f} "
                f"{e2e.get('tpot_ms', 0):<10.2f} "
                f"{diff.get('tpot', 0):+<10.1f} "
                f"{sol.get('output_tput_per_gpu', 0):<10.1f} "
                f"{e2e.get('output_tput_per_gpu', 0):<10.1f} "
                f"{diff.get('output_tput_per_gpu', 0):+<10.1f} "
                f"{ttft_ms:<10.0f} "
                f"{ttft_status:<6} "
                f"{overall}"
            )

    # E2E Jobs (if still running)
    e2e_jobs = state.get("e2e_jobs", [])
    running_e2e = [ej for ej in e2e_jobs if ej.get("status") not in ("completed", "failed")]
    if running_e2e and live:
        print(f"\n  E2E JOBS IN PROGRESS ({len(running_e2e)} remaining)")
        for ej in running_e2e:
            job_id = ej.get("job_id", "-")
            slurm_status = _slurm_status(job_id) if job_id != "-" else "?"
            print(f"    Job {job_id}: {slurm_status} "
                  f"(rank {ej.get('pareto_rank')}, {ej.get('multiplier')}x)")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rate-matching sweep status")
    parser.add_argument("-o", "--output", required=True, help="Sweep output directory")
    parser.add_argument("--live", action="store_true", help="Query live SLURM status")
    args = parser.parse_args()
    print_status(args.output, live=args.live)


if __name__ == "__main__":
    main()
