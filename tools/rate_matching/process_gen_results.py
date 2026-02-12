#!/usr/bin/env python3
"""
Process GEN-only benchmark results from srt-slurm per-iteration logs.

Parses decode worker logs to extract GEN throughput metrics using the
rate-matching methodology (filter for pure decode iterations).

ENGINE-SPECIFIC: TRT-LLM per-iteration log format.

Usage:
    python process_gen_results.py -i /path/to/job/logs
    python process_gen_results.py -i outputs/11301/logs --concurrency 8 --mode tep

Methodology:
    1. Parse per-iteration logs from decode worker (*_decode_w*.out or *_agg_w*.out)
    2. Filter for iterations where num_ctx_tokens == 0 (pure decode)
    3. Skip first 2 and last 2 iterations (warmup/cooldown)
    4. Filter outliers using median +/- 20%
    5. Calculate: tpot_ms = avg(prev_device_step_time_ms) / avg(num_generation_tokens) * concurrency
       Actually: tpot_ms = avg_step_time / (batch_utilisation)
       Simpler: throughput_per_user = 1000 / avg_step_time_ms (when batch is full)
    6. Report: tpot_ms, interactivity, output_throughput, throughput_per_gpu
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ENGINE-SPECIFIC: TRT-LLM per-iteration log format
LOG_PATTERN = re.compile(
    r'iter = (\d+), global_rank = (\d+), rank = (\d+), '
    r'currank_total_requests = (\d+)/(\d+), '
    r'host_step_time = ([\d.]+)ms, '
    r'prev_device_step_time = ([\d.]+|N/A)ms, '
    r'timestamp = ([^,]+), '
    r"num_scheduled_requests: (\d+), "
    r"states = \{'num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+), 'num_generation_tokens': (\d+)\}"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Process GEN-only benchmark results from srt-slurm logs',
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to logs directory')
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Benchmark concurrency used')
    parser.add_argument('--mode', type=str, default='tep',
                        choices=['tep', 'dep'], help='TEP or DEP mode')
    parser.add_argument('--tp', type=int, default=8, help='Tensor parallelism')
    parser.add_argument('--mtp', type=int, default=0, help='MTP layers (0=STP)')
    parser.add_argument('--num-gpus', type=int, default=8, help='GPUs used')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def find_decode_log(logs_dir: Path) -> Optional[Path]:
    """Find decode worker log file in the logs directory."""
    patterns = [
        '*_decode_w*.out',  # Disaggregated decode worker
        '*_agg_w*.out',     # Aggregated worker
    ]
    for pattern in patterns:
        matches = list(logs_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_log_file(log_file: Path, verbose: bool = False) -> list[dict]:
    """Parse per-iteration log entries from a decode log file."""
    data = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        matches = LOG_PATTERN.findall(content)
        if verbose:
            print(f"Found {len(matches)} iteration entries in {log_file.name}")

        for match in matches:
            prev_device_step_time_str = match[6]
            if prev_device_step_time_str == 'N/A':
                prev_device_step_time_ms = None
            else:
                prev_device_step_time_ms = float(prev_device_step_time_str)

            data.append({
                'iter': int(match[0]),
                'global_rank': int(match[1]),
                'rank': int(match[2]),
                'current_requests': int(match[3]),
                'total_requests': int(match[4]),
                'host_step_time_ms': float(match[5]),
                'prev_device_step_time_ms': prev_device_step_time_ms,
                'timestamp': match[7],
                'num_scheduled_requests': int(match[8]),
                'num_ctx_requests': int(match[9]),
                'num_ctx_tokens': int(match[10]),
                'num_generation_tokens': int(match[11]),
            })
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return data


def process_gen_data(
    data: list[dict],
    concurrency: int = 32,
    mode: str = 'tep',
    tp: int = 8,
    mtp: int = 0,
    num_gpus: int = 8,
    verbose: bool = False,
) -> dict:
    """Process GEN log data using the rate-matching methodology.

    Returns metrics dict with tpot_ms, throughput_per_user, output_throughput, etc.
    """
    if not data:
        return {'error': 'No data to process'}

    df = pd.DataFrame(data)

    # Remove rows where prev_device_step_time is None (first iteration)
    df = df[df['prev_device_step_time_ms'].notna()].copy()
    if verbose:
        print(f"After removing N/A: {len(df)} entries")

    # Filter for pure decode iterations (num_ctx_tokens == 0)
    df = df[df['num_ctx_tokens'] == 0].copy()
    if verbose:
        print(f"After filtering num_ctx_tokens == 0: {len(df)} entries")

    if df.empty:
        return {'error': 'No pure decode iterations found'}

    # Skip first 2 and last 2 iterations
    if len(df) > 4:
        df = df.iloc[2:-2].copy()
        if verbose:
            print(f"After warmup/cooldown trim: {len(df)} entries")

    if len(df) < 5:
        return {'error': f'Insufficient data after filtering: {len(df)} entries'}

    # Filter outliers using median +/- 20%
    median_step = df['prev_device_step_time_ms'].median()
    lower = median_step * 0.8
    upper = median_step * 1.2
    before = len(df)
    df = df[(df['prev_device_step_time_ms'] >= lower) &
            (df['prev_device_step_time_ms'] <= upper)].copy()
    if verbose:
        print(f"Filtered {before - len(df)} outliers (median +/- 20%)")

    if len(df) < 5:
        return {'error': f'Insufficient data after outlier filtering: {len(df)} entries'}

    # Calculate metrics
    avg_step_time_ms = df['prev_device_step_time_ms'].mean()
    avg_gen_tokens = df['num_generation_tokens'].mean()

    # MTP accept rate: if MTP, effective tokens per step = gen_tokens / batch
    # For MTP, each scheduled request generates (1 + accept_rate * mtp_num) tokens per step
    # But for TPOT calculation, what matters is the step time per token per user
    # tpot_ms = avg_step_time_ms (time for one decode step)
    # throughput_per_user = 1000 / tpot_ms (tokens/s that one user sees)
    # For MTP: tpot_ms per accepted token = avg_step_time_ms / effective_tokens_per_step_per_user

    # MTP accept rate estimation
    if mtp > 0 and concurrency > 0:
        # avg_gen_tokens = batch * (1 + accept_rate * mtp_num)
        # For full batch: expected_base = concurrency (one token per request per step)
        expected_base = concurrency
        if avg_gen_tokens > expected_base:
            mtp_accept_rate = (avg_gen_tokens / expected_base - 1.0) / mtp
            mtp_accept_rate = min(max(mtp_accept_rate, 0.0), 1.0)
        else:
            mtp_accept_rate = 0.0
        effective_tokens_per_user = 1.0 + mtp_accept_rate * mtp
    else:
        mtp_accept_rate = 1.0  # STP: no speculation
        effective_tokens_per_user = 1.0

    # TPOT = step_time / effective_tokens_per_user
    tpot_ms = avg_step_time_ms / effective_tokens_per_user

    # Throughput per user (interactivity) = 1000 / tpot_ms
    throughput_per_user = 1000.0 / tpot_ms if tpot_ms > 0 else 0

    # Total output throughput = tokens per second across all concurrent users
    # = avg_gen_tokens / (avg_step_time_ms / 1000)
    output_throughput = avg_gen_tokens / (avg_step_time_ms / 1000.0) if avg_step_time_ms > 0 else 0

    return {
        'interactivity': round(throughput_per_user, 2),
        'throughput_per_gpu': round(output_throughput / num_gpus, 2),
        'output_throughput': round(output_throughput, 2),
        'throughput_per_user': round(throughput_per_user, 3),
        'avg_step_time_ms': round(avg_step_time_ms, 4),
        'tpot_ms': round(tpot_ms, 4),
        'concurrency': concurrency,
        'mode': mode,
        'mtp': mtp,
        'mtp_accept_rate': round(mtp_accept_rate, 4),
        'num_gpus': num_gpus,
        'num_iterations': len(df),
    }


def main():
    args = parse_arguments()

    logs_dir = Path(args.input)
    if not logs_dir.exists():
        print(f"Error: Directory not found: {logs_dir}")
        sys.exit(1)

    log_file = find_decode_log(logs_dir)
    if log_file is None:
        print(f"Error: No decode worker log found in {logs_dir}")
        print("Expected patterns: *_decode_w*.out or *_agg_w*.out")
        sys.exit(1)

    print(f"Processing: {log_file}")
    data = parse_log_file(log_file, verbose=args.verbose)
    if not data:
        print("Error: No iteration data found in log file")
        sys.exit(1)

    print(f"Parsed {len(data)} iteration entries")

    results = process_gen_data(
        data,
        concurrency=args.concurrency,
        mode=args.mode,
        tp=args.tp,
        mtp=args.mtp,
        num_gpus=args.num_gpus,
        verbose=args.verbose,
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GEN-only Results")
    print("=" * 60)
    print(f"  TPOT:                {results['tpot_ms']:.4f} ms")
    print(f"  Interactivity:       {results['interactivity']:.2f} tok/s/user")
    print(f"  Output Throughput:   {results['output_throughput']:.2f} tok/s")
    print(f"  Throughput/GPU:      {results['throughput_per_gpu']:.2f} tok/s/GPU")
    print(f"  Avg Step Time:       {results['avg_step_time_ms']:.4f} ms")
    print(f"  MTP Accept Rate:     {results['mtp_accept_rate']:.4f}")
    print(f"  Concurrency:         {results['concurrency']}")
    print(f"  Iterations Used:     {results['num_iterations']}")
    print("=" * 60)

    output_file = args.output or (logs_dir / 'gen_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
