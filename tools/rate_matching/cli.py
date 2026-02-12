#!/usr/bin/env python3
"""
Rate-matching sweep automation CLI.

Entry point for the srtctl-rate-match command. Can be invoked as:
    python tools/rate_matching/cli.py run -c sweep.yaml -o ./sweeps/my_sweep
    python tools/rate_matching/cli.py status -o ./sweeps/my_sweep
    python tools/rate_matching/cli.py gen-sweep -c sweep.yaml -o ./sweeps/my_sweep

Or via symlink/alias:
    srtctl-rate-match run -c sweep.yaml -o ./sweeps/my_sweep
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


def cmd_run(args: argparse.Namespace) -> None:
    """Run a rate-matching sweep."""
    from run_sweep import run_sweep

    output_dir = args.output
    if not output_dir:
        # Auto-generate output dir
        from schema import load_sweep_config
        cfg = load_sweep_config(args.config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(_HERE) / "sweeps" / f"{cfg.name}_{timestamp}")

    state = run_sweep(
        config_path=args.config,
        output_dir=output_dir,
        resume=args.resume,
        skip_e2e=args.skip_e2e,
        dry_run=args.dry_run,
        verbose=True,
    )

    if not args.dry_run:
        print(f"\nTo check status:")
        print(f"  python {__file__} status -o {state.output_dir}")
        print(f"  python {__file__} status -o {state.output_dir} --live")


def cmd_status(args: argparse.Namespace) -> None:
    """Show sweep status."""
    from sweep_status import print_status
    print_status(args.output, live=args.live)


def cmd_gen_sweep(args: argparse.Namespace) -> None:
    """Generate configs without submitting (dry-run mode)."""
    from run_sweep import run_sweep

    output_dir = args.output
    if not output_dir:
        from schema import load_sweep_config
        cfg = load_sweep_config(args.config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(_HERE) / "sweeps" / f"{cfg.name}_{timestamp}")

    state = run_sweep(
        config_path=args.config,
        output_dir=output_dir,
        resume=False,
        skip_e2e=False,
        dry_run=True,
        verbose=True,
    )
    print(f"\nConfigs generated in: {state.output_dir}/configs/")
    print("To submit, run without --dry-run:")
    print(f"  python {__file__} run -c {args.config} -o {state.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="srtctl-rate-match",
        description="Rate-matching sweep automation for srt-slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s run -c sweep.yaml -o ./sweeps/my_sweep
  %(prog)s run -c sweep.yaml -o ./sweeps/my_sweep --dry-run
  %(prog)s run -c sweep.yaml -o ./sweeps/my_sweep --resume
  %(prog)s status -o ./sweeps/my_sweep
  %(prog)s status -o ./sweeps/my_sweep --live
  %(prog)s gen-sweep -c sweep.yaml -o ./sweeps/my_sweep
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run a rate-matching sweep")
    run_parser.add_argument("-c", "--config", required=True, help="Sweep config YAML")
    run_parser.add_argument("-o", "--output", default=None, help="Output directory (auto-generated if omitted)")
    run_parser.add_argument("--dry-run", action="store_true", help="Generate configs only, don't submit")
    run_parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    run_parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E validation phase")
    run_parser.set_defaults(func=cmd_run)

    # status
    status_parser = subparsers.add_parser("status", help="Show sweep status")
    status_parser.add_argument("-o", "--output", required=True, help="Sweep output directory")
    status_parser.add_argument("--live", action="store_true", help="Query live SLURM status")
    status_parser.set_defaults(func=cmd_status)

    # gen-sweep
    gen_parser = subparsers.add_parser("gen-sweep", help="Generate sweep configs only")
    gen_parser.add_argument("-c", "--config", required=True, help="Sweep config YAML")
    gen_parser.add_argument("-o", "--output", default=None, help="Output directory")
    gen_parser.set_defaults(func=cmd_gen_sweep)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
