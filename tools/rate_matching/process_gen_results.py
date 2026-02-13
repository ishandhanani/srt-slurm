#!/usr/bin/env python3
"""
Process GEN-only benchmark results from srt-slurm per-iteration logs.

Aligned with the original rate-matching methodology in:
  /data/users/nlevin/rate-matching/bench-trtllm-disagg/process_data/process_gen_iterlog_withctx.py

ENGINE-SPECIFIC: TRT-LLM per-iteration log format.

Methodology (original, exactly replicated):
    1. Parse per-iteration logs from decode worker (*_decode_w*.out or *_agg_w*.out)
    2. Filter for iterations where num_ctx_tokens == 0 (pure decode)
    3. Merge duplicate rows by (iter, global_rank) keeping last
    4. Remove first 50 and last 10 iterations (warmup/cooldown)
    5. Filter by exact concurrency match:
       - TEP: num_scheduled_requests == concurrency
              num_generation_tokens == concurrency * (mtp_num + 1)
       - DEP: num_scheduled_requests == concurrency / ep_rank
              num_generation_tokens == concurrency / ep_rank * (mtp_num + 1)
    6. Filter outliers using median +/- 20% of prev_device_step_time
    7. Calculate:
         elapsed_time_avg = mean(prev_device_step_time)  (in SECONDS, ms/1000)
         throughput_per_user = (1 / elapsed_time_avg) * mtp_accept_rate
         tpot = elapsed_time_avg / mtp_accept_rate  (in seconds)
         output_throughput = throughput_per_user * concurrency

MTP accept rates (hardcoded from original rate-matching repo measurements):
    1k/1k random: {1: 1.8, 2: 2.28, 3: 2.56}
    8k/1k random: {1: 1.84, 2: 2.38, 3: 2.76}
    32k/1k random: {1: 1.97, 2: 2.39, 3: 2.56}
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# MTP accept rates from original rate-matching repo
# These represent the effective tokens per decode step per user.
# For STP (mtp=0), the rate is 1.0 (one token per step per user).
# For MTP-3 at 1k/1k, the rate is 2.56 (each step produces ~2.56 tokens).
# ---------------------------------------------------------------------------
MTP_ACCEPT_RATES = {
    # ISL -> {mtp_num -> effective_tokens_per_step}
    1024: {0: 1.0, 1: 1.8, 2: 2.28, 3: 2.56},    # 1k/1k random
    8192: {0: 1.0, 1: 1.84, 2: 2.38, 3: 2.76},    # 8k/1k random
    32768: {0: 1.0, 1: 1.97, 2: 2.39, 3: 2.56},   # 32k/1k random
}

# Default fallback (uses 1k/1k values)
_DEFAULT_ACCEPT_RATES = {0: 1.0, 1: 1.8, 2: 2.28, 3: 2.56}


def get_mtp_accept_rate(isl: int, mtp_num: int) -> float:
    """Get the MTP accept rate for a given ISL and MTP level.

    Returns the effective tokens per decode step per user.
    For STP (mtp_num=0), returns 1.0.
    For MTP, returns a value > 1 (e.g., 2.56 for MTP-3 at 1k/1k).
    """
    rates = MTP_ACCEPT_RATES.get(isl, _DEFAULT_ACCEPT_RATES)
    return rates.get(mtp_num, 1.0)


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
    parser.add_argument('--ep-rank', type=int, default=None,
                        help='Expert parallel rank (DEP). Defaults to tp for DEP.')
    parser.add_argument('--mtp', type=int, default=0, help='MTP layers (0=STP)')
    parser.add_argument('--isl', type=int, default=1024,
                        help='ISL (for MTP accept rate lookup)')
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
    """Parse per-iteration log entries from a decode log file.

    Values are stored in their log units:
      - host_step_time_ms: milliseconds (as logged)
      - prev_device_step_time_ms: milliseconds (as logged)

    The original rate-matching repo converts ms -> seconds during processing.
    We do this conversion in process_gen_data().
    """
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
    ep_rank: Optional[int] = None,
    mtp: int = 0,
    isl: int = 1024,
    num_gpus: int = 8,
    verbose: bool = False,
) -> dict:
    """Process GEN log data using the original rate-matching methodology.

    CRITICAL: This function replicates the EXACT filtering and calculation
    from process_gen_iterlog_withctx.py in the original rate-matching repo.

    Steps:
        1. Filter num_ctx_tokens == 0 (pure decode)
        2. Merge duplicates by (iter, global_rank) keeping last
        3. Skip first 50 / last 10 iterations
        4. Filter by exact concurrency match on num_scheduled_requests
           and num_generation_tokens
        5. Outlier filter: median ±20%
        6. Calculate metrics using elapsed_time_avg (in seconds)

    Args:
        data: Parsed log entries from parse_log_file()
        concurrency: Benchmark concurrency for this measurement
        mode: 'tep' or 'dep'
        tp: Tensor parallelism
        ep_rank: Expert parallel rank (for DEP). Defaults to tp.
        mtp: MTP layers (0 = STP). Note: mtp_num in original code is the
             number of MTP layers, and the accept rate lookup uses this.
        isl: Input sequence length (for MTP accept rate lookup)
        num_gpus: GPUs used for this decode worker
        verbose: Print debug info

    Returns:
        Dict with tpot_ms, throughput_per_user, output_throughput, etc.
    """
    if not data:
        return {'error': 'No data to process'}

    if ep_rank is None:
        ep_rank = tp  # Default: ep_rank = rank_num = tp

    df = pd.DataFrame(data)

    # Remove rows where prev_device_step_time is None (first iteration)
    df = df[df['prev_device_step_time_ms'].notna()].copy()
    if verbose:
        print(f"After removing N/A: {len(df)} entries")

    # Step 1: Filter for pure decode iterations (num_ctx_tokens == 0)
    df = df[df['num_ctx_tokens'] == 0].copy()
    if verbose:
        print(f"After filtering num_ctx_tokens == 0: {len(df)} entries")

    if df.empty:
        return {'error': 'No pure decode iterations found'}

    # Step 2: Merge duplicate rows by (iter, global_rank) keeping last
    # (aligned with original: df.groupby(['iter', 'global_rank']).last())
    df = df.sort_values(['iter', 'global_rank'])
    df = df.drop_duplicates(subset=['iter', 'global_rank'], keep='last').copy()
    if verbose:
        print(f"After dedup by (iter, global_rank): {len(df)} entries")

    # Step 3: Remove first 50 and last 10 iterations (warmup/cooldown)
    # (aligned with original: df.iloc[50:-10])
    if len(df) > 60:
        df = df.iloc[50:-10].copy()
        if verbose:
            print(f"After warmup/cooldown trim (50/10): {len(df)} entries")
    elif len(df) > 10:
        # Fewer iterations than expected, still trim some
        skip = min(50, len(df) // 3)
        df = df.iloc[skip:-max(1, len(df) // 10)].copy()
        if verbose:
            print(f"After reduced trim ({skip}): {len(df)} entries")

    if len(df) < 5:
        return {'error': f'Insufficient data after warmup trim: {len(df)} entries'}

    # Step 4: Filter by exact concurrency match
    # This is the critical step that isolates iterations for a specific
    # concurrency when multiple concurrencies ran in the same job.
    mtp_num = mtp  # mtp_num as used in original code
    if mode == 'tep':
        expected_scheduled = concurrency
        expected_gen_tokens = concurrency * (mtp_num + 1)
    else:  # dep
        expected_scheduled = concurrency // ep_rank
        expected_gen_tokens = (concurrency // ep_rank) * (mtp_num + 1)

    before_conc = len(df)
    df = df[df['num_scheduled_requests'] == expected_scheduled].copy()
    if verbose:
        print(f"After num_scheduled_requests == {expected_scheduled}: "
              f"{len(df)} entries (filtered {before_conc - len(df)})")

    before_gen = len(df)
    df = df[df['num_generation_tokens'] == expected_gen_tokens].copy()
    if verbose:
        print(f"After num_generation_tokens == {expected_gen_tokens}: "
              f"{len(df)} entries (filtered {before_gen - len(df)})")

    if len(df) < 5:
        return {'error': (
            f'Insufficient data after concurrency filter: {len(df)} entries. '
            f'Expected: num_scheduled_requests={expected_scheduled}, '
            f'num_generation_tokens={expected_gen_tokens}'
        )}

    # Step 5: Filter outliers using median ±20%
    median_step = df['prev_device_step_time_ms'].median()
    lower = median_step * 0.8
    upper = median_step * 1.2
    before = len(df)
    df_filtered = df[(df['prev_device_step_time_ms'] >= lower) &
                     (df['prev_device_step_time_ms'] <= upper)].copy()
    if not df_filtered.empty:
        if verbose:
            print(f"Filtered {before - len(df_filtered)} outliers "
                  f"(median={median_step:.2f}ms ±20%), "
                  f"{len(df_filtered)} remaining")
        df = df_filtered

    if len(df) < 5:
        return {'error': f'Insufficient data after outlier filtering: {len(df)} entries'}

    # Step 6: Calculate metrics (aligned with original methodology)
    # Original uses elapsed_time in SECONDS: prev_device_step_time (ms) / 1000
    avg_step_time_ms = df['prev_device_step_time_ms'].mean()
    elapsed_time_avg = avg_step_time_ms / 1000.0  # Convert to seconds

    # MTP accept rate from hardcoded lookup table
    mtp_accept_rate = get_mtp_accept_rate(isl, mtp_num)

    # throughput_per_user = (1 / elapsed_time_avg) * mtp_accept_rate
    # This is tokens/s/user (interactivity)
    throughput_per_user = (1.0 / elapsed_time_avg) * mtp_accept_rate if elapsed_time_avg > 0 else 0

    # tpot = elapsed_time_avg / mtp_accept_rate (in seconds)
    tpot_s = elapsed_time_avg / mtp_accept_rate if mtp_accept_rate > 0 else elapsed_time_avg
    tpot_ms = tpot_s * 1000.0

    # output_throughput = throughput_per_user * concurrency (total tok/s)
    output_throughput = throughput_per_user * concurrency

    # output_throughput_per_gen_gpu = output_throughput / ep_rank
    output_throughput_per_gen_gpu = output_throughput / ep_rank

    return {
        'interactivity': round(throughput_per_user, 2),
        'throughput_per_gpu': round(output_throughput_per_gen_gpu, 2),
        'output_throughput': round(output_throughput, 2),
        'throughput_per_user': round(throughput_per_user, 3),
        'avg_step_time_ms': round(avg_step_time_ms, 4),
        'tpot_ms': round(tpot_ms, 4),
        'concurrency': concurrency,
        'mode': mode,
        'mtp': mtp_num,
        'mtp_accept_rate': round(mtp_accept_rate, 4),
        'ep_rank': ep_rank,
        'num_gpus': num_gpus,
        'num_iterations': len(df),
    }


def process_gen_data_all_concurrencies(
    data: list[dict],
    concurrency_list: list[int],
    mode: str = 'tep',
    tp: int = 8,
    ep_rank: Optional[int] = None,
    mtp: int = 0,
    isl: int = 1024,
    num_gpus: int = 8,
    verbose: bool = False,
) -> dict[int, dict]:
    """Process GEN log data for MULTIPLE concurrencies from a single decode log.

    When srt-slurm runs sa-bench with concurrencies "8x32x64", the decode
    worker log contains iterations for ALL concurrencies in sequence.
    The exact-match filter on num_scheduled_requests and num_generation_tokens
    (Step 4) naturally segments them.

    Returns:
        Dict mapping concurrency -> result dict.
    """
    results = {}
    for conc in concurrency_list:
        result = process_gen_data(
            data,
            concurrency=conc,
            mode=mode,
            tp=tp,
            ep_rank=ep_rank,
            mtp=mtp,
            isl=isl,
            num_gpus=num_gpus,
            verbose=verbose,
        )
        results[conc] = result
    return results


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

    ep_rank = args.ep_rank if args.ep_rank else args.tp
    results = process_gen_data(
        data,
        concurrency=args.concurrency,
        mode=args.mode,
        tp=args.tp,
        ep_rank=ep_rank,
        mtp=args.mtp,
        isl=args.isl,
        num_gpus=args.num_gpus,
        verbose=args.verbose,
    )

    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GEN-only Results (rate-matching methodology)")
    print("=" * 60)
    print(f"  TPOT:                {results['tpot_ms']:.4f} ms")
    print(f"  Interactivity:       {results['interactivity']:.2f} tok/s/user")
    print(f"  Output Throughput:   {results['output_throughput']:.2f} tok/s")
    print(f"  Throughput/GPU:      {results['throughput_per_gpu']:.2f} tok/s/GPU")
    print(f"  Avg Step Time:       {results['avg_step_time_ms']:.4f} ms")
    print(f"  MTP Accept Rate:     {results['mtp_accept_rate']:.4f}")
    print(f"  Concurrency:         {results['concurrency']}")
    print(f"  EP Rank:             {results['ep_rank']}")
    print(f"  Iterations Used:     {results['num_iterations']}")
    print("=" * 60)

    output_file = args.output or (logs_dir / 'gen_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
