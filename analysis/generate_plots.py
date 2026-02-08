#!/usr/bin/env python3
"""
Standalone script to generate benchmark analysis plots.

Produces the same visualizations as the Streamlit dashboard, but as static files.

Usage:
    python analysis/generate_plots.py /path/to/logs --output ./plots
    python analysis/generate_plots.py /path/to/logs --job-ids 12345 12346 --output ./plots
    python analysis/generate_plots.py /path/to/logs --format png --output ./plots
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from analysis.srtlog import NodeAnalyzer, RunLoader
from analysis.srtlog.visualizations import (
    aggregate_all_nodes,
    parse_elapsed_time,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_node_metrics(run_path: str) -> list[dict]:
    """Load node metrics from log files.

    Checks both run_path and run_path/logs for log files.
    """
    analyzer = NodeAnalyzer()

    # Try run_path first
    nodes = analyzer.parse_run_logs(run_path)

    # If no nodes found, try logs/ subdirectory
    if not nodes:
        logs_subdir = os.path.join(run_path, "logs")
        if os.path.exists(logs_subdir):
            nodes = analyzer.parse_run_logs(logs_subdir)

    # Convert to dicts for compatibility with visualization code
    result = []
    for node in nodes:
        node_dict = {
            "node_info": node.node_info,
            "prefill_batches": [],
            "memory_snapshots": [],
            "config": node.config,
            "run_id": node.run_id,
        }

        # Convert batches
        for batch in node.batches:
            batch_dict = {
                "timestamp": batch.timestamp,
                "dp": batch.dp,
                "tp": batch.tp,
                "ep": batch.ep,
                "type": batch.batch_type,
            }
            for field in [
                "new_seq",
                "new_token",
                "cached_token",
                "token_usage",
                "running_req",
                "queue_req",
                "prealloc_req",
                "inflight_req",
                "input_throughput",
                "gen_throughput",
                "transfer_req",
                "num_tokens",
                "preallocated_usage",
            ]:
                value = getattr(batch, field)
                if value is not None:
                    batch_dict[field] = value
            node_dict["prefill_batches"].append(batch_dict)

        # Convert memory snapshots
        for mem in node.memory_snapshots:
            mem_dict = {
                "timestamp": mem.timestamp,
                "dp": mem.dp,
                "tp": mem.tp,
                "ep": mem.ep,
                "type": mem.metric_type,
            }
            for field in ["avail_mem_gb", "mem_usage_gb", "kv_cache_gb", "kv_tokens"]:
                value = getattr(mem, field)
                if value is not None:
                    mem_dict[field] = value
            node_dict["memory_snapshots"].append(mem_dict)

        result.append(node_dict)

    return result


def save_figure(fig, output_dir: str, name: str, format: str = "png"):
    """Save a matplotlib figure to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.{format}")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {filepath}")


def generate_pareto_plots(
    df: pd.DataFrame,
    selected_runs: list[str],
    run_labels: dict[str, str],
    output_dir: str,
    format: str,
    show_frontier: bool = True,
):
    """Generate Pareto frontier plots using matplotlib."""
    colors = plt.cm.Set1.colors

    # (x_metric, y_metric, filename, title)
    plot_configs = [
        ("Output TPS/User", "Output TPS/GPU", "pareto_output_tps_per_gpu", "Pareto: Output TPS/GPU vs Output TPS/User"),
        ("Output TPS/User", "Total TPS/GPU", "pareto_total_tps_per_gpu", "Pareto: Total TPS/GPU vs Output TPS/User"),
        (
            "Output TPS/User @P90 Latency",
            "Output TPS/GPU",
            "pareto_tps_at_p90_latency",
            "Pareto: Output TPS/GPU vs Output TPS/User @P90 Latency",
        ),
    ]

    for x_metric, y_metric, filename, title in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 7))

        for idx, run_id in enumerate(selected_runs):
            run_data = df[df["Run ID"] == run_id].copy()
            run_data = run_data[(run_data[y_metric] != "N/A") & (run_data[x_metric] != "N/A")]

            if run_data.empty:
                continue

            label = run_labels.get(run_id, run_id)
            color = colors[idx % len(colors)]

            ax.plot(
                run_data[x_metric],
                run_data[y_metric],
                "o-",
                color=color,
                label=label,
                markersize=8,
                linewidth=2,
            )

        ax.set_xlabel(x_metric, fontsize=12)
        ax.set_ylabel(y_metric, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        save_figure(fig, output_dir, filename, format)


def generate_pareto_subplots(
    df: pd.DataFrame,
    selected_runs: list[str],
    run_labels: dict[str, str],
    output_dir: str,
    format: str,
):
    """Generate Pareto plots with each run as a subplot in a grid."""
    import math

    colors = plt.cm.Set1.colors
    n_runs = len(selected_runs)

    if n_runs == 0:
        return

    # Calculate grid dimensions
    ncols = min(2, n_runs)
    nrows = math.ceil(n_runs / ncols)

    for y_metric, filename in [
        ("Output TPS/GPU", "pareto_output_tps_per_gpu_grid"),
        ("Total TPS/GPU", "pareto_total_tps_per_gpu_grid"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

        # Compute global axis limits for consistent scales
        all_x, all_y = [], []
        for run_id in selected_runs:
            run_data = df[df["Run ID"] == run_id].copy()
            run_data = run_data[run_data[y_metric] != "N/A"]
            if not run_data.empty:
                all_x.extend(run_data["Output TPS/User"].tolist())
                all_y.extend(run_data[y_metric].tolist())

        x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)
        y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)
        x_pad = (x_max - x_min) * 0.1 or 0.1
        y_pad = (y_max - y_min) * 0.1 or 0.1

        for idx, run_id in enumerate(selected_runs):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            run_data = df[df["Run ID"] == run_id].copy()
            run_data = run_data[run_data[y_metric] != "N/A"]

            if run_data.empty:
                ax.set_visible(False)
                continue

            label = run_labels.get(run_id, run_id)
            color = colors[idx % len(colors)]

            ax.plot(
                run_data["Output TPS/User"],
                run_data[y_metric],
                "o-",
                color=color,
                markersize=8,
                linewidth=2,
            )

            ax.set_xlabel("Output TPS/User", fontsize=10)
            ax.set_ylabel(y_metric, fontsize=10)
            ax.set_title(label, fontsize=10)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_runs, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Pareto: {y_metric} vs Output TPS/User", fontsize=14)
        fig.tight_layout()
        save_figure(fig, output_dir, filename, format)


def generate_latency_plots(df: pd.DataFrame, selected_runs: list[str], output_dir: str, format: str):
    """Generate latency vs concurrency plots using matplotlib."""
    colors = plt.cm.Set1.colors

    # (metric_col, filename, title, use_log_scale)
    metrics = [
        ("Mean TTFT (ms)", "latency_ttft", "TTFT vs Concurrency", True),
        ("Mean TPOT (ms)", "latency_tpot", "TPOT vs Concurrency", False),
    ]

    for metric_col, filename, title, use_log_scale in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, run_id in enumerate(selected_runs):
            run_data = df[df["Run ID"] == run_id].sort_values("Concurrency")
            valid_data = run_data[run_data[metric_col] != "N/A"].copy()

            if valid_data.empty:
                continue

            color = colors[idx % len(colors)]

            ax.plot(
                valid_data["Concurrency"],
                valid_data[metric_col],
                "o-",
                color=color,
                label=run_id,
                markersize=8,
                linewidth=2,
            )

        ax.set_xlabel("Concurrency", fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.set_title(title, fontsize=14)
        if use_log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        save_figure(fig, output_dir, filename, format)


def generate_latency_subplots(
    df: pd.DataFrame, selected_runs: list[str], run_labels: dict[str, str], output_dir: str, format: str
):
    """Generate latency plots with each run as a subplot in a grid."""
    import math

    colors = plt.cm.Set1.colors
    n_runs = len(selected_runs)

    if n_runs == 0:
        return

    # Calculate grid dimensions
    ncols = min(2, n_runs)
    nrows = math.ceil(n_runs / ncols)

    # (metric_col, filename, title, use_log_scale)
    metrics = [
        ("Mean TTFT (ms)", "latency_ttft_grid", "TTFT vs Concurrency", True),
        ("Mean TPOT (ms)", "latency_tpot_grid", "TPOT vs Concurrency", False),
    ]

    for metric_col, filename, title, use_log_scale in metrics:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

        # Compute global axis limits for consistent scales
        all_x, all_y = [], []
        for run_id in selected_runs:
            run_data = df[df["Run ID"] == run_id].sort_values("Concurrency")
            valid_data = run_data[run_data[metric_col] != "N/A"].copy()
            if not valid_data.empty:
                all_x.extend(valid_data["Concurrency"].tolist())
                all_y.extend(valid_data[metric_col].tolist())

        x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 1)
        y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 1)
        x_pad = (x_max - x_min) * 0.1 or 0.1

        for idx, run_id in enumerate(selected_runs):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            run_data = df[df["Run ID"] == run_id].sort_values("Concurrency")
            valid_data = run_data[run_data[metric_col] != "N/A"].copy()

            if valid_data.empty:
                ax.set_visible(False)
                continue

            label = run_labels.get(run_id, run_id)
            color = colors[idx % len(colors)]

            ax.plot(
                valid_data["Concurrency"],
                valid_data[metric_col],
                "o-",
                color=color,
                markersize=8,
                linewidth=2,
            )

            ax.set_xlabel("Concurrency", fontsize=10)
            ax.set_ylabel(metric_col, fontsize=10)
            ax.set_title(label, fontsize=10)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            if use_log_scale:
                ax.set_yscale("log")
            else:
                y_pad = (y_max - y_min) * 0.1 or 0.1
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_runs, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        save_figure(fig, output_dir, filename, format)


def _extract_metric_timeseries(node_metrics: list[dict], metric_key: str, batch_filter=None, value_extractor=None):
    """Extract time series data from node metrics."""
    all_data = []

    for node_data in node_metrics:
        if not node_data.get("prefill_batches"):
            continue

        timestamps = []
        values = []

        for batch in node_data["prefill_batches"]:
            if batch_filter and not batch_filter(batch):
                continue

            ts = batch.get("timestamp")
            if not ts:
                continue

            value = value_extractor(batch) if value_extractor else batch.get(metric_key)

            if value is not None:
                timestamps.append(ts)
                values.append(value)

        if timestamps:
            elapsed = parse_elapsed_time(timestamps)
            node_info = node_data.get("node_info", {})
            label = f"{node_info.get('worker_type', 'unknown')} {node_info.get('worker_id', '')}"
            all_data.append({"elapsed": elapsed, "values": values, "label": label})

    return all_data


def generate_node_metric_plots(
    node_metrics: list[dict], output_dir: str, format: str, prefix: str = "", aggregate: bool = True
):
    """Generate node-level metric plots using matplotlib."""
    if not node_metrics:
        logger.warning(f"No node metrics to plot for {prefix}")
        return

    if aggregate:
        node_metrics = aggregate_all_nodes(node_metrics)

    colors = plt.cm.tab10.colors

    # Input Throughput
    data = _extract_metric_timeseries(node_metrics, "input_throughput")
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Input Throughput (tokens/s)", fontsize=12)
        ax.set_title("Input Throughput Over Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, f"{prefix}input_throughput", format)

    # KV Cache Utilization
    data = _extract_metric_timeseries(
        node_metrics, "token_usage", value_extractor=lambda b: b.get("token_usage", 0) * 100
    )
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Utilization (%)", fontsize=12)
        ax.set_title("KV Cache Utilization Over Time", fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, f"{prefix}kv_cache_utilization", format)

    # Queue Depth
    data = _extract_metric_timeseries(node_metrics, "queue_req")
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Number of Requests", fontsize=12)
        ax.set_title("Queued Requests Over Time", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, f"{prefix}queue_depth", format)


def generate_prefill_plots(prefill_nodes: list[dict], output_dir: str, format: str, aggregate: bool = True):
    """Generate prefill-specific node metric plots using matplotlib."""
    if not prefill_nodes:
        return

    if aggregate:
        prefill_nodes = aggregate_all_nodes(prefill_nodes)

    colors = plt.cm.tab10.colors

    # Inflight Requests
    data = _extract_metric_timeseries(prefill_nodes, "inflight_req")
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Number of Requests", fontsize=12)
        ax.set_title("Inflight Requests Over Time (Prefill)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, "prefill_inflight_requests", format)


def generate_decode_plots(decode_nodes: list[dict], output_dir: str, format: str, aggregate: bool = True):
    """Generate decode-specific node metric plots using matplotlib."""
    if not decode_nodes:
        return

    if aggregate:
        decode_nodes = aggregate_all_nodes(decode_nodes)

    colors = plt.cm.tab10.colors

    def decode_filter(b):
        return b.get("type") == "decode"

    # Running Requests
    data = _extract_metric_timeseries(decode_nodes, "running_req", batch_filter=decode_filter)
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Number of Requests", fontsize=12)
        ax.set_title("Running Requests Over Time (Decode)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, "decode_running_requests", format)

    # Generation Throughput
    data = _extract_metric_timeseries(decode_nodes, "gen_throughput", batch_filter=decode_filter)
    if data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, series in enumerate(data):
            ax.plot(series["elapsed"], series["values"], "o-", color=colors[idx % len(colors)], label=series["label"])
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
        ax.set_ylabel("Gen Throughput (tokens/s)", fontsize=12)
        ax.set_title("Generation Throughput Over Time (Decode)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, "decode_gen_throughput", format)

    # Disaggregation stacked chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect stacked data
    prealloc_data = _extract_metric_timeseries(decode_nodes, "prealloc_req", batch_filter=decode_filter)
    transfer_data = _extract_metric_timeseries(decode_nodes, "transfer_req", batch_filter=decode_filter)
    running_data = _extract_metric_timeseries(decode_nodes, "running_req", batch_filter=decode_filter)

    if prealloc_data or transfer_data or running_data:
        # Use the first series' timestamps as reference
        ref_data = prealloc_data or transfer_data or running_data
        if ref_data:
            elapsed = ref_data[0]["elapsed"]
            prealloc = prealloc_data[0]["values"] if prealloc_data else [0] * len(elapsed)
            transfer = transfer_data[0]["values"] if transfer_data else [0] * len(elapsed)
            running = running_data[0]["values"] if running_data else [0] * len(elapsed)

            ax.stackplot(
                elapsed,
                prealloc,
                transfer,
                running,
                labels=["Prealloc Queue", "Transfer Queue", "Running"],
                colors=["#636EFA50", "#EF553B50", "#00CC9650"],
                alpha=0.7,
            )
            ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
            ax.set_ylabel("Number of Requests", fontsize=12)
            ax.set_title("Disaggregation Request Flow (Stacked)", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            save_figure(fig, output_dir, "decode_disagg_flow", format)
    else:
        plt.close(fig)


def generate_rate_match_plot(prefill_nodes: list[dict], decode_nodes: list[dict], output_dir: str, format: str):
    """Generate rate match comparison plot using matplotlib."""
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    # Get prefill input throughput
    if prefill_nodes:
        all_prefill_batches = {}
        for p_node in prefill_nodes:
            for batch in p_node["prefill_batches"]:
                if batch.get("input_throughput") is not None:
                    ts = batch.get("timestamp", "")
                    if ts:
                        if ts not in all_prefill_batches:
                            all_prefill_batches[ts] = []
                        all_prefill_batches[ts].append(batch["input_throughput"])

        timestamps = []
        avg_input_tps = []

        for ts in sorted(all_prefill_batches.keys()):
            avg = sum(all_prefill_batches[ts]) / len(all_prefill_batches[ts])
            timestamps.append(ts)
            avg_input_tps.append(avg)

        if timestamps:
            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            elapsed = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds() for ts in timestamps]

            ax.plot(
                elapsed,
                avg_input_tps,
                "o-",
                color="orange",
                label=f"Prefill Input Rate (avg {len(prefill_nodes)} nodes)",
                linewidth=2,
                markersize=6,
            )
            has_data = True

    # Get decode gen throughput
    if decode_nodes:
        all_decode_batches = {}
        for d_node in decode_nodes:
            for batch in d_node["prefill_batches"]:
                if batch.get("gen_throughput") is not None and batch.get("gen_throughput") > 0:
                    ts = batch.get("timestamp", "")
                    if ts:
                        if ts not in all_decode_batches:
                            all_decode_batches[ts] = []
                        all_decode_batches[ts].append(batch["gen_throughput"])

        timestamps = []
        avg_gen_tps = []

        for ts in sorted(all_decode_batches.keys()):
            avg = sum(all_decode_batches[ts]) / len(all_decode_batches[ts])
            timestamps.append(ts)
            avg_gen_tps.append(avg)

        if timestamps:
            first_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            elapsed = [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - first_time).total_seconds() for ts in timestamps]

            ax.plot(
                elapsed,
                avg_gen_tps,
                "o-",
                color="green",
                label=f"Decode Gen Rate (avg {len(decode_nodes)} nodes)",
                linewidth=2,
                markersize=6,
            )
            has_data = True

    if has_data:
        ax.set_xlabel("Time Elapsed (seconds)", fontsize=12)
        ax.set_ylabel("Average Throughput (tokens/s per node)", fontsize=12)
        ax.set_title("Rate Match: Prefill Input vs Decode Generation", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        save_figure(fig, output_dir, "rate_match", format)
    else:
        plt.close(fig)


def export_data_csv(df: pd.DataFrame, output_dir: str):
    """Export benchmark data to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "benchmark_data.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"Exported data: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark analysis plots")
    parser.add_argument("logs_dir", help="Path to logs directory containing benchmark runs")
    parser.add_argument("--output", "-o", default="./plots", help="Output directory for plots (default: ./plots)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--job-ids", "-j", nargs="+", type=str, help="Specific job IDs to include (default: all available)"
    )
    parser.add_argument("--no-pareto", action="store_true", help="Skip Pareto plots")
    parser.add_argument("--no-latency", action="store_true", help="Skip latency plots")
    parser.add_argument("--no-node-metrics", action="store_true", help="Skip node-level metric plots")
    parser.add_argument("--no-rate-match", action="store_true", help="Skip rate match plot")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV data export")
    parser.add_argument(
        "--individual", action="store_true", help="Generate individual plots per run in addition to combined"
    )
    parser.add_argument(
        "--show-frontier", action="store_true", default=True, help="Show Pareto frontier (default: True)"
    )
    parser.add_argument("--no-aggregate", action="store_true", help="Show individual nodes instead of aggregated")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate logs directory
    if not os.path.exists(args.logs_dir):
        logger.error(f"Logs directory not found: {args.logs_dir}")
        sys.exit(1)

    # Load runs
    logger.info(f"Loading benchmark runs from {args.logs_dir}")
    loader = RunLoader(args.logs_dir)
    all_runs, skipped = loader.load_all_with_skipped()

    if skipped:
        logger.warning(f"Skipped {len(skipped)} runs:")
        for job_id, _run_dir, reason in skipped:
            logger.warning(f"  - Job {job_id}: {reason}")

    if not all_runs:
        logger.error("No benchmark runs found")
        sys.exit(1)

    logger.info(f"Found {len(all_runs)} benchmark runs")

    # Filter by job IDs if specified
    if args.job_ids:
        all_runs = [r for r in all_runs if r.job_id in args.job_ids]
        if not all_runs:
            logger.error(f"No runs found matching job IDs: {args.job_ids}")
            sys.exit(1)
        logger.info(f"Filtered to {len(all_runs)} runs matching specified job IDs")

    # Build run IDs and labels
    selected_runs = []
    run_labels = {}

    for run in all_runs:
        if run.metadata.is_aggregated:
            run_id = f"{run.job_id}_{run.metadata.agg_workers}A_{run.metadata.run_date}"
            total_gpus = run.metadata.agg_nodes * run.metadata.gpus_per_node
            label = (
                f"{run.job_id} | "
                f"{run.metadata.agg_workers}A | "
                f"{total_gpus} GPUs | "
                f"{run.profiler.isl}/{run.profiler.osl}"
            )
        else:
            run_id = (
                f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"
            )
            prefill_gpus = run.metadata.prefill_nodes * run.metadata.gpus_per_node
            decode_gpus = run.metadata.decode_nodes * run.metadata.gpus_per_node
            label = (
                f"{run.job_id} | "
                f"{run.metadata.prefill_workers}P{run.metadata.decode_workers}D | "
                f"{prefill_gpus}/{decode_gpus} | "
                f"{run.profiler.isl}/{run.profiler.osl}"
            )

        if run.metadata.gpu_type:
            label += f" | {run.metadata.gpu_type}"

        selected_runs.append(run_id)
        run_labels[run_id] = label

    # Get DataFrame
    df = loader.to_dataframe(all_runs)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    # Generate plots
    if not args.no_pareto:
        logger.info("Generating Pareto plots...")
        generate_pareto_plots(df, selected_runs, run_labels, args.output, args.format, args.show_frontier)
        if args.individual:
            logger.info("Generating Pareto subplot grid...")
            generate_pareto_subplots(df, selected_runs, run_labels, args.output, args.format)

    if not args.no_latency:
        logger.info("Generating latency plots...")
        generate_latency_plots(df, selected_runs, args.output, args.format)
        if args.individual:
            logger.info("Generating latency subplot grid...")
            generate_latency_subplots(df, selected_runs, run_labels, args.output, args.format)

    if not args.no_csv:
        logger.info("Exporting benchmark data to CSV...")
        export_data_csv(df, args.output)

    # Node-level metrics (requires parsing log files)
    if not args.no_node_metrics or not args.no_rate_match:
        logger.info("Loading node metrics from log files...")

        all_node_metrics = []
        for run in all_runs:
            run_path = run.metadata.path
            if run.metadata.is_aggregated:
                run_id = f"{run.job_id}_{run.metadata.agg_workers}A_{run.metadata.run_date}"
            else:
                run_id = f"{run.job_id}_{run.metadata.prefill_workers}P_{run.metadata.decode_workers}D_{run.metadata.run_date}"

            if run_path and os.path.exists(run_path):
                node_metrics = load_node_metrics(run_path)
                for node_data in node_metrics:
                    node_data["run_id"] = run_id
                    node_data["run_metadata"] = {
                        "job_id": run.job_id,
                        "is_aggregated": run.metadata.is_aggregated,
                    }
                all_node_metrics.extend(node_metrics)

        if all_node_metrics:
            # Split by type
            prefill_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "prefill"]
            decode_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] == "decode"]
            agg_nodes = [n for n in all_node_metrics if n["node_info"]["worker_type"] in ["agg", "aggregated"]]

            logger.info(f"Found {len(prefill_nodes)} prefill, {len(decode_nodes)} decode, {len(agg_nodes)} agg nodes")

            aggregate = not args.no_aggregate

            if not args.no_node_metrics:
                # Generate node metric plots
                if agg_nodes:
                    logger.info("Generating aggregated node metric plots...")
                    generate_node_metric_plots(agg_nodes, args.output, args.format, prefix="agg_", aggregate=aggregate)

                if prefill_nodes:
                    logger.info("Generating prefill node metric plots...")
                    generate_node_metric_plots(
                        prefill_nodes, args.output, args.format, prefix="prefill_", aggregate=aggregate
                    )
                    generate_prefill_plots(prefill_nodes, args.output, args.format, aggregate=aggregate)

                if decode_nodes:
                    logger.info("Generating decode node metric plots...")
                    generate_node_metric_plots(
                        decode_nodes, args.output, args.format, prefix="decode_", aggregate=aggregate
                    )
                    generate_decode_plots(decode_nodes, args.output, args.format, aggregate=aggregate)

            if not args.no_rate_match and prefill_nodes and decode_nodes:
                logger.info("Generating rate match plot...")
                generate_rate_match_plot(prefill_nodes, decode_nodes, args.output, args.format)
        else:
            logger.warning("No node metrics found in log files")

    logger.info(f"Done! Plots saved to {args.output}")


if __name__ == "__main__":
    """
    # Combined (all runs overlaid)
    # sa-bench (old runs)
        python outputs/generate_plots.py outputs/ --job-ids 1539007 1539009 1539470 1547123 1547869 --no-csv --individual -o ./plots
        python outputs/generate_plots.py outputs/ --job-ids 1539007 1539009 1539470 1547123 1547869 --no-csv -o ./plots/combined
        python outputs/generate_plots.py outputs/ --job-ids 1539007 --no-csv -o ./plots/tep8x1_tep8x3
        python outputs/generate_plots.py outputs/ --job-ids 1539009 --no-csv -o ./plots/tep8x2_tep8x2
        python outputs/generate_plots.py outputs/ --job-ids 1539470 --no-csv -o ./plots/tep8x3_tep8x1
        python outputs/generate_plots.py outputs/ --job-ids 1547123 --no-csv -o ./plots/dep8x2_dep8x2
        python outputs/generate_plots.py outputs/ --job-ids 1547869 --no-csv -o ./plots/agg_dep8x4

    # aiperf 
        python outputs/generate_plots.py outputs/ --job-ids 1551614 1551670 1551671 1551672 --no-csv -o ./plots/aiperf_runs

    """
    main()
