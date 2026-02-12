#!/usr/bin/env python3
"""
Process CTX-only benchmark results from srt-slurm per-iteration logs.

This script parses prefill worker logs to extract CTX metrics using the same
methodology as the original rate-matching repo (get_ctx_throughput.py).

ENGINE-SPECIFIC: TRT-LLM
This log parser is specific to TRT-LLM's per-iteration log format.
For SGLang/vLLM support, either:
1. Create engine-specific parsers (SGLangLogParser, VLLMLogParser)
2. Fall back to SA-bench client-side metrics (request_throughput, median_ttft_ms)

See WORKLOG.md "Engine-Agnostic Refactoring Plan" for details.

Usage:
    python process_ctx_results.py -i /path/to/job/logs
    python process_ctx_results.py -i outputs/11059/logs --isl 8192

Methodology:
    1. Parse per-iteration logs from prefill worker (*_prefill_w*.out or *_agg_w*.out)
    2. Filter for iterations where num_generation_tokens == 0 (pure prefill)
    3. Skip first 2 and last 2 iterations (warmup/cooldown)
    4. Filter by num_ctx_requests >= threshold (2 for 8k ISL, 16 for 1k ISL)
    5. Filter outliers using median ±20%
    6. Calculate: ctx_throughput = num_ctx_tokens / prev_device_step_time
    7. Calculate: request_rate = sum(num_ctx_requests) / sum(prev_device_step_time)
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ENGINE-SPECIFIC: TRT-LLM per-iteration log format
# SGLang/vLLM will have different formats - create separate parsers
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process CTX-only benchmark results from srt-slurm logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to logs directory (e.g., outputs/11059/logs)')
    parser.add_argument('--isl', type=int, default=8192,
                        help='Input sequence length (used for filtering threshold)')
    parser.add_argument('--ctx-dep', action='store_true', default=True,
                        help='Whether CTX uses dep mode (affects request_rate calculation)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file path (default: <input>/ctx_results.json)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()


def get_threshold_for_isl(isl: int) -> int:
    """Get num_ctx_requests threshold based on ISL."""
    thresholds = {
        1024: 16,
        8192: 2,
        32768: 1,
    }
    if isl in thresholds:
        return thresholds[isl]
    # Default threshold based on ISL range
    if isl <= 2048:
        return 16
    elif isl <= 16384:
        return 2
    else:
        return 1


def find_prefill_log(logs_dir: Path) -> Optional[Path]:
    """Find prefill worker log file in the logs directory."""
    # Try different patterns
    patterns = [
        '*_prefill_w*.out',  # Disaggregated prefill worker
        '*_agg_w*.out',      # Aggregated worker
    ]
    
    for pattern in patterns:
        matches = list(logs_dir.glob(pattern))
        if matches:
            # Return the first match (usually only one)
            return matches[0]
    
    return None


def parse_log_file(log_file: Path, verbose: bool = False) -> list[dict]:
    """Parse per-iteration log entries from a log file."""
    data = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        matches = LOG_PATTERN.findall(content)
        
        if verbose:
            print(f"Found {len(matches)} iteration entries in {log_file.name}")
        
        for match in matches:
            iter_num = int(match[0])
            global_rank = int(match[1])
            rank = int(match[2])
            current_requests = int(match[3])
            total_requests = int(match[4])
            host_step_time_ms = float(match[5])
            
            # Handle N/A for prev_device_step_time (first iteration)
            prev_device_step_time_str = match[6]
            if prev_device_step_time_str == 'N/A':
                prev_device_step_time_ms = None
            else:
                prev_device_step_time_ms = float(prev_device_step_time_str)
            
            timestamp = match[7]
            num_scheduled_requests = int(match[8])
            num_ctx_requests = int(match[9])
            num_ctx_tokens = int(match[10])
            num_generation_tokens = int(match[11])
            
            data.append({
                'iter': iter_num,
                'global_rank': global_rank,
                'rank': rank,
                'current_requests': current_requests,
                'total_requests': total_requests,
                'host_step_time_ms': host_step_time_ms,
                'prev_device_step_time_ms': prev_device_step_time_ms,
                'timestamp': timestamp,
                'num_scheduled_requests': num_scheduled_requests,
                'num_ctx_requests': num_ctx_requests,
                'num_ctx_tokens': num_ctx_tokens,
                'num_generation_tokens': num_generation_tokens,
            })
    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return data


def process_ctx_data(data: list[dict], isl: int, ctx_dep: bool = True, verbose: bool = False) -> dict:
    """
    Process CTX log data using the rate-matching methodology.
    
    Returns metrics dict with ctx_throughput, request_rate, etc.
    """
    if not data:
        return {'error': 'No data to process'}
    
    df = pd.DataFrame(data)
    
    # Filter out rows where prev_device_step_time is None (first iteration)
    df = df[df['prev_device_step_time_ms'].notna()].copy()
    
    if verbose:
        print(f"After removing N/A prev_device_step_time: {len(df)} entries")
    
    # Step 1: Filter for pure prefill iterations (num_generation_tokens == 0)
    df = df[df['num_generation_tokens'] == 0].copy()
    
    if verbose:
        print(f"After filtering num_generation_tokens == 0: {len(df)} entries")
    
    if df.empty:
        return {'error': 'No pure prefill iterations found'}
    
    # Step 2: Get filtering threshold based on ISL
    threshold = get_threshold_for_isl(isl)
    
    # Step 3: Filter by num_ctx_requests threshold
    df_filtered = df[df['num_ctx_requests'] >= threshold].copy()
    
    if verbose:
        print(f"After filtering num_ctx_requests >= {threshold}: {len(df_filtered)} entries")
    
    # Step 4: Skip first 2 and last 2 iterations
    if len(df_filtered) > 4:
        df_filtered = df_filtered.iloc[2:-2].copy()
        if verbose:
            print(f"After removing first 2 and last 2 iterations: {len(df_filtered)} entries")
    
    if df_filtered.empty or len(df_filtered) < 5:
        return {'error': f'Insufficient data after filtering: {len(df_filtered)} entries'}
    
    # Step 5: Filter outliers using median ±20%
    prev_device_step_time_median = df_filtered['prev_device_step_time_ms'].median()
    lower_bound = prev_device_step_time_median * 0.8
    upper_bound = prev_device_step_time_median * 1.2
    
    before_count = len(df_filtered)
    df_filtered = df_filtered[
        (df_filtered['prev_device_step_time_ms'] >= lower_bound) &
        (df_filtered['prev_device_step_time_ms'] <= upper_bound)
    ].copy()
    after_count = len(df_filtered)
    
    if verbose:
        print(f"Filtered {before_count - after_count} outliers (median ±20%)")
        print(f"Final dataset: {len(df_filtered)} entries")
    
    if df_filtered.empty or len(df_filtered) < 5:
        return {'error': f'Insufficient data after outlier filtering: {len(df_filtered)} entries'}
    
    # Step 6: Calculate metrics
    # Convert ms to seconds for calculations
    df_filtered['prev_device_step_time_s'] = df_filtered['prev_device_step_time_ms'] / 1000.0
    
    # Per-iteration throughput
    df_filtered['ctx_throughput'] = df_filtered['num_ctx_tokens'] / df_filtered['prev_device_step_time_s']
    
    # Aggregate statistics
    avg_prev_device_step_time_s = df_filtered['prev_device_step_time_s'].mean()
    avg_prev_device_step_time_ms = df_filtered['prev_device_step_time_ms'].mean()
    avg_num_ctx_tokens = df_filtered['num_ctx_tokens'].mean()
    avg_ctx_throughput = df_filtered['ctx_throughput'].mean()
    avg_num_ctx_requests = df_filtered['num_ctx_requests'].mean()
    
    # Calculate request rate (requests per second)
    # Sum approach: total requests / total time
    sum_ctx_requests = df_filtered['num_ctx_requests'].sum()
    sum_prev_device_step_time_s = df_filtered['prev_device_step_time_s'].sum()
    request_rate = sum_ctx_requests / sum_prev_device_step_time_s if sum_prev_device_step_time_s > 0 else 0
    
    # Adjust for number of ranks if using dep mode
    num_ranks = df_filtered['global_rank'].nunique()
    if ctx_dep and num_ranks > 1:
        request_rate = request_rate * num_ranks
    
    return {
        'ctx_throughput_tokens_per_s': round(avg_ctx_throughput, 2),
        'request_rate_req_per_s': round(request_rate, 4),
        'avg_prev_device_step_time_ms': round(avg_prev_device_step_time_ms, 4),
        'avg_num_ctx_tokens': round(avg_num_ctx_tokens, 2),
        'avg_num_ctx_requests': round(avg_num_ctx_requests, 2),
        'num_iterations': len(df_filtered),
        'num_ranks': num_ranks,
        'isl': isl,
        'threshold_used': threshold,
    }


def main():
    args = parse_arguments()
    
    logs_dir = Path(args.input)
    if not logs_dir.exists():
        print(f"Error: Directory not found: {logs_dir}")
        sys.exit(1)
    
    # Find prefill log file
    log_file = find_prefill_log(logs_dir)
    if log_file is None:
        print(f"Error: No prefill worker log found in {logs_dir}")
        print("Expected patterns: *_prefill_w*.out or *_agg_w*.out")
        sys.exit(1)
    
    print(f"Processing: {log_file}")
    
    # Parse log file
    data = parse_log_file(log_file, verbose=args.verbose)
    
    if not data:
        print("Error: No iteration data found in log file")
        sys.exit(1)
    
    print(f"Parsed {len(data)} iteration entries")
    
    # Process data
    results = process_ctx_data(data, args.isl, args.ctx_dep, verbose=args.verbose)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Print results
    print("\n" + "=" * 60)
    print("CTX-only Results")
    print("=" * 60)
    print(f"  CTX Throughput:      {results['ctx_throughput_tokens_per_s']:,.2f} tokens/s")
    print(f"  Request Rate:        {results['request_rate_req_per_s']:.4f} req/s")
    print(f"  Avg Step Time:       {results['avg_prev_device_step_time_ms']:.4f} ms")
    print(f"  Avg CTX Tokens:      {results['avg_num_ctx_tokens']:.2f}")
    print(f"  Avg CTX Requests:    {results['avg_num_ctx_requests']:.2f}")
    print(f"  Iterations Used:     {results['num_iterations']}")
    print(f"  Ranks:               {results['num_ranks']}")
    print("=" * 60)
    
    # Save results to JSON
    output_file = args.output or (logs_dir / 'ctx_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
