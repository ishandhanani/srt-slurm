#!/usr/bin/env python3
"""
Rate-matching sweep automation CLI.

Entry point for the srtctl-rate-match command. Can be invoked as:
    srtctl-rate-match run -f sweep.yaml
    srtctl-rate-match dry-run -f sweep.yaml
    srtctl-rate-match status -o ./sweeps/my_sweep
    srtctl-rate-match cancel -o ./sweeps/my_sweep
    srtctl-rate-match reprocess -o ./sweeps/my_sweep
    srtctl-rate-match reprocess -o ./sweeps/my_sweep -f updated.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure tools/rate_matching is on the path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _resolve_output_dir(config_path: str, output: str | None) -> str:
    """Auto-generate output directory from config name + timestamp if not provided."""
    if output:
        return output
    from schema import load_sweep_config
    cfg = load_sweep_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(_HERE) / "sweeps" / f"{cfg.name}_{timestamp}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a rate-matching sweep (submit to SLURM)."""
    from run_sweep import run_sweep

    output_dir = _resolve_output_dir(args.config, args.output)

    state = run_sweep(
        config_path=args.config,
        output_dir=output_dir,
        resume=args.resume,
        skip_e2e=args.skip_e2e,
        dry_run=False,
        verbose=True,
    )

    print(f"\nTo check status:")
    print(f"  srtctl-rate-match status -o {state.output_dir}")
    print(f"  srtctl-rate-match status -o {state.output_dir} --live")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Validate config and generate sweep configs without submitting."""
    from run_sweep import run_sweep

    output_dir = _resolve_output_dir(args.config, args.output)

    state = run_sweep(
        config_path=args.config,
        output_dir=output_dir,
        resume=False,
        skip_e2e=False,
        dry_run=True,
        verbose=True,
    )
    print(f"\nConfigs generated in: {state.output_dir}/configs/")
    print("To submit:")
    print(f"  srtctl-rate-match run -f {args.config} -o {state.output_dir}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show sweep status."""
    from sweep_status import print_status
    print_status(args.output, live=args.live)


def cmd_cancel(args: argparse.Namespace) -> None:
    """Cancel all SLURM jobs associated with a sweep."""
    import json
    import subprocess

    state_path = Path(args.output) / "sweep_state.json"
    if not state_path.exists():
        print(f"No sweep state found at: {state_path}")
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    # Collect all job IDs from ctx, gen, and e2e jobs
    job_ids: list[str] = []

    ctx_job = state.get("ctx_job", {})
    if ctx_job.get("job_id"):
        job_ids.append(str(ctx_job["job_id"]))

    for gj in state.get("gen_jobs", []):
        if gj.get("job_id"):
            job_ids.append(str(gj["job_id"]))

    for ej in state.get("e2e_jobs", []):
        if ej.get("job_id"):
            job_ids.append(str(ej["job_id"]))

    if not job_ids:
        print("No job IDs found in sweep state.")
        return

    # Check which are still active in SLURM
    active_ids = []
    for jid in job_ids:
        result = subprocess.run(
            ["squeue", "--job", jid, "--noheader", "--format=%i %T"],
            capture_output=True, text=True,
        )
        status_line = result.stdout.strip()
        if status_line:
            active_ids.append((jid, status_line.split()[-1]))

    if not active_ids:
        print(f"No active SLURM jobs found (checked {len(job_ids)} job IDs).")
        return

    print(f"Found {len(active_ids)} active job(s):")
    for jid, status in active_ids:
        print(f"  Job {jid}: {status}")

    if not args.yes:
        try:
            confirm = input("Cancel all? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = ""
        if confirm != "y":
            print("Aborted.")
            return

    # Cancel all active jobs
    ids_to_cancel = [jid for jid, _ in active_ids]
    result = subprocess.run(
        ["scancel"] + ids_to_cancel,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Cancelled {len(ids_to_cancel)} job(s): {', '.join(ids_to_cancel)}")
    else:
        print(f"scancel failed: {result.stderr.strip()}")
        sys.exit(1)


def cmd_reprocess(args: argparse.Namespace) -> None:
    """Re-process an existing sweep from log files (no SLURM submission)."""
    from run_sweep import reprocess_sweep

    state = reprocess_sweep(
        output_dir=args.output,
        config_path=args.config,
        skip_e2e=args.skip_e2e,
        verbose=True,
    )
    print(f"\nReprocessing complete. Results in: {state.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="srtctl-rate-match",
        description="Rate-matching sweep automation for srt-slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run a sweep
  %(prog)s run -f sweep.yaml
  %(prog)s run -f sweep.yaml -o ./sweeps/my_sweep
  %(prog)s run -f sweep.yaml --resume

  # Validate and generate configs without submitting
  %(prog)s dry-run -f sweep.yaml

  # Monitor and manage
  %(prog)s status -o ./sweeps/my_sweep
  %(prog)s status -o ./sweeps/my_sweep --live
  %(prog)s cancel -o ./sweeps/my_sweep
  %(prog)s cancel -o ./sweeps/my_sweep -y

  # Re-process results from existing logs
  %(prog)s reprocess -o ./sweeps/my_sweep
  %(prog)s reprocess -o ./sweeps/my_sweep -f updated_sweep.yaml
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run - submit and execute sweep (mirrors: srtctl apply)
    run_parser = subparsers.add_parser("run", help="Run a rate-matching sweep")
    run_parser.add_argument("-f", "--file", required=True, dest="config", help="Sweep config YAML")
    run_parser.add_argument("-o", "--output", default=None, help="Output directory (auto-generated if omitted)")
    run_parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    run_parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E validation phase")
    run_parser.set_defaults(func=cmd_run)

    # dry-run - validate and generate configs only (mirrors: srtctl dry-run)
    dry_run_parser = subparsers.add_parser("dry-run", help="Validate config and generate sweep configs without submitting")
    dry_run_parser.add_argument("-f", "--file", required=True, dest="config", help="Sweep config YAML")
    dry_run_parser.add_argument("-o", "--output", default=None, help="Output directory (auto-generated if omitted)")
    dry_run_parser.set_defaults(func=cmd_dry_run)

    # status
    status_parser = subparsers.add_parser("status", help="Show sweep status")
    status_parser.add_argument("-o", "--output", required=True, help="Sweep output directory")
    status_parser.add_argument("--live", action="store_true", help="Query live SLURM status")
    status_parser.set_defaults(func=cmd_status)

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel all SLURM jobs for a sweep")
    cancel_parser.add_argument("-o", "--output", required=True, help="Sweep output directory")
    cancel_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    cancel_parser.set_defaults(func=cmd_cancel)

    # reprocess
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Re-process existing sweep from logs (no SLURM submission)",
    )
    reprocess_parser.add_argument("-o", "--output", required=True, help="Sweep output directory")
    reprocess_parser.add_argument(
        "-f", "--file", default=None, dest="config",
        help="Updated sweep config YAML (optional; uses sweep's original if omitted)",
    )
    reprocess_parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E re-processing")
    reprocess_parser.set_defaults(func=cmd_reprocess)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
