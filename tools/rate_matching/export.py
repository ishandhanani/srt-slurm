"""Result export helpers: sa-bench JSON loading and CSV/JSON export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from state import SweepState

# Avoid circular import; RateMatchingSweepConfig is only used for type hints.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from schema import RateMatchingSweepConfig


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
