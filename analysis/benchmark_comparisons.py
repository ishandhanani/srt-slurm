# %% [markdown]
# # Benchmark Comparison
# 
# Compare disaggregated vs aggregated mode with KV Router vs Round Robin routing.

# %%
import re
from pathlib import Path
import pandas as pd

# Default paths to benchmark.out files
DISAGG_KV_PATH = "../outputs/1573345_ctx2_dep8_gen2_dep8_batch32_8_nvfp4_router_trace/logs/benchmark.out"
DISAGG_RR_PATH = "../outputs/1588126_ctx2_dep8_gen2_dep8_batch32_8_nvfp4_trace_failed/logs/benchmark.out"
AGG_KV_PATH = "../outputs/1588128_agg4_dep8_batch8_nvfp4_router_trace/logs/benchmark.out"
AGG_RR_PATH = "../outputs/1577312_agg4_dep8_batch8_nvfp4_trace/logs/benchmark.out"
# AGG_RR_PATH = "../outputs/1582653_agg4_dep8_batch8_nvfp4_trace_trtllmkv/logs/benchmark.out"
# %%
def parse_benchmark_results(file_path: str) -> dict:
    """Parse the 'Serving Benchmark Result' section from a benchmark.out file."""
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f: 
        content = f.read()
    
    # Find the final "Serving Benchmark Result" section
    pattern = r'={12} Serving Benchmark Result ={12}(.+?)={50}'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        raise ValueError(f"Could not find 'Serving Benchmark Result' section in {file_path}")
    
    # Use the last match (final results)
    result_section = matches[-1].group(1)
    
    # Parse key-value pairs
    metrics = {}
    
    # Pattern for "Metric name:   value"
    kv_pattern = r'^([A-Za-z][A-Za-z0-9 ()-]+?):\s+([0-9.]+)\s*$'
    
    for line in result_section.split('\n'):
        line = line.strip()
        if not line or line.startswith('-'):
            continue
        
        match = re.match(kv_pattern, line)
        if match:
            key = match.group(1).strip()
            value = float(match.group(2))
            metrics[key] = value
    
    return metrics

# %%
def parse_aiperf_throughput(file_path: str) -> dict:
    """Parse Output Token Throughput from the final NVIDIA AIPerf LLM Metrics table."""
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find all NVIDIA AIPerf | LLM Metrics sections
    # Split by the header and take the last one
    sections = content.split('NVIDIA AIPerf | LLM Metrics')
    if len(sections) < 2:
        return {}
    
    final_section = sections[-1]
    
    metrics = {}
    
    # Parse Output Token Throughput (tokens/sec) - the system-wide one, not per-user
    # The table format has multiline rows. We look for the pattern:
    # │     Output │    633.71 │ ... (followed by lines with Token, Throughput, (tokens/s)
    # But NOT the "Per User" variant
    # Note: "Throughput" may be truncated to "Throughp…" in narrow tables
    
    lines = final_section.split('\n')
    for i, line in enumerate(lines):
        # Look for "Output" followed by a number, then check subsequent lines
        # to distinguish between "Per User" and system-wide throughput
        if '│' in line and 'Output' in line:
            # Check if this row is followed by "Token", "Throughput"/"Throughp", "(tokens/s" but NOT "Per User"
            context = '\n'.join(lines[i:i+5])
            if 'Per User' in context:
                continue
            # Handle both "Throughput" and truncated "Throughp…"
            if ('Throughput' in context or 'Throughp' in context) and '(tokens' in context:
                # Extract the avg value (second column after the metric name)
                parts = line.split('│')
                if len(parts) >= 3:
                    try:
                        val_str = parts[2].strip().replace(',', '')
                        if val_str and val_str != 'N/A':
                            metrics['Output Token Throughput (tok/s)'] = float(val_str)
                    except (ValueError, IndexError):
                        pass
    
    # Also try to get Request Throughput
    for i, line in enumerate(lines):
        if '│' in line and 'Request' in line:
            context = '\n'.join(lines[i:i+3])
            # Handle both "Throughput" and truncated "Throughp…"
            if ('Throughput' in context or 'Throughp' in context) and '(request' in context:
                parts = line.split('│')
                if len(parts) >= 3:
                    try:
                        val_str = parts[2].strip().replace(',', '')
                        if val_str and val_str != 'N/A':
                            metrics['Request Throughput (req/s)'] = float(val_str)
                    except (ValueError, IndexError):
                        pass
    
    return metrics

# %%
def load_all_benchmarks(disagg_kv=DISAGG_KV_PATH, disagg_rr=DISAGG_RR_PATH, 
                        agg_kv=AGG_KV_PATH, agg_rr=AGG_RR_PATH) -> dict:
    """Load all 4 benchmark results, merging standard metrics with AIPerf throughput."""
    paths = {
        'disagg_kv': disagg_kv,
        'disagg_rr': disagg_rr,
        'agg_kv': agg_kv,
        'agg_rr': agg_rr,
    }
    results = {}
    for name, path in paths.items():
        metrics = parse_benchmark_results(path)
        # Merge AIPerf throughput metrics
        metrics.update(parse_aiperf_throughput(path))
        results[name] = metrics
    return results

# %%
# Load all benchmarks
results = load_all_benchmarks()

# Show raw parsed data
for name, metrics in results.items():
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# %%
def create_comparison_table(results: dict, mode: str) -> pd.DataFrame:
    """Create a comparison table for either 'disagg' or 'agg' mode."""
    
    if mode == 'disagg':
        kv_data = results['disagg_kv']
        rr_data = results['disagg_rr']
        col_names = ['KV Router (disagg)', 'Round Robin (disagg)']
    else:
        kv_data = results['agg_kv']
        rr_data = results['agg_rr']
        col_names = ['KV Router (agg)', 'Round Robin (agg)']
    
    # Define metrics to display with friendly names
    metric_mapping = {
        'Mean TTFT (ms)': 'Mean TTFT (ms)',
        'Median TTFT (ms)': 'Median TTFT (ms)',
        'P99 TTFT (ms)': 'P99 TTFT (ms)',
        'Mean ITL (ms)': 'Mean ITL (ms)',
        'Median ITL (ms)': 'Median ITL (ms)',
        'P99 ITL (ms)': 'P99 ITL (ms)',
        'Mean E2EL (ms)': 'Mean E2EL (ms)',
        'Median E2EL (ms)': 'Median E2EL (ms)',
        'P99 E2EL (ms)': 'P99 E2EL (ms)',
        'Output token throughput (tok/s)': 'Throughput (tok/s)',
        'Request throughput (req/s)': 'Throughput (req/s)',
        'Successful requests': 'Successful Requests',
    }
    
    rows = []
    for src_name, display_name in metric_mapping.items():
        kv_val = kv_data.get(src_name, None)
        rr_val = rr_data.get(src_name, None)
        rows.append({
            'Metric': display_name,
            col_names[0]: kv_val,
            col_names[1]: rr_val,
        })
    
    df = pd.DataFrame(rows)
    df = df.set_index('Metric')
    return df

# %%
def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format numbers with thousands separators and 2 decimal places."""
    return df.apply(lambda x: x.map(lambda v: f"{v:,.2f}" if pd.notna(v) else "N/A"))

# %% [markdown]
# ## Disaggregated Mode Comparison

# %%
disagg_df = create_comparison_table(results, 'disagg')
format_table(disagg_df)

# %% [markdown]
# ## Aggregated Mode Comparison

# %%
agg_df = create_comparison_table(results, 'agg')
format_table(agg_df)

# %% [markdown]
# ## All Experiments Side-by-Side

# %%
def create_full_comparison_table(results: dict) -> pd.DataFrame:
    """Create a table with all 4 experiments."""
    
    col_names = ['Disagg + KV', 'Disagg + RR', 'Agg + KV', 'Agg + RR']
    data_keys = ['disagg_kv', 'disagg_rr', 'agg_kv', 'agg_rr']
    
    metric_mapping = {
        'Mean TTFT (ms)': 'TTFT Mean (ms)',
        'Median TTFT (ms)': 'TTFT Median (ms)',
        'P99 TTFT (ms)': 'TTFT P99 (ms)',
        'Mean ITL (ms)': 'ITL Mean (ms)',
        'Median ITL (ms)': 'ITL Median (ms)',
        'P99 ITL (ms)': 'ITL P99 (ms)',
        'Mean E2EL (ms)': 'E2EL Mean (ms)',
        'Median E2EL (ms)': 'E2EL Median (ms)',
        'P99 E2EL (ms)': 'E2EL P99 (ms)',
        'Output token throughput (tok/s)': 'Throughput (tok/s)',
        'Request throughput (req/s)': 'Throughput (req/s)',
        'Successful requests': 'Successful Requests',
    }
    
    rows = []
    for src_name, display_name in metric_mapping.items():
        row = {'Metric': display_name}
        for col_name, data_key in zip(col_names, data_keys):
            row[col_name] = results[data_key].get(src_name, None)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Metric')
    return df

full_df = create_full_comparison_table(results)
format_table(full_df)

# %% [markdown]
# ## Summary Statistics

# %%
# Key metrics comparison
summary_metrics = ['Mean TTFT (ms)', 'Mean ITL (ms)', 'Mean E2EL (ms)', 
                   'Output token throughput (tok/s)', 'Successful requests']

summary_data = []
for name, data in results.items():
    row = {'Experiment': name}
    for metric in summary_metrics:
        row[metric] = data.get(metric, None)
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data).set_index('Experiment')
format_table(summary_df)

# %%
# Calculate speedups
print("=== Speedup Analysis ===")
print()

# Agg: KV Router vs Round Robin
agg_kv_ttft = results['agg_kv']['Mean TTFT (ms)']
agg_rr_ttft = results['agg_rr']['Mean TTFT (ms)']
print(f"Agg Mode - KV Router vs Round Robin:")
print(f"  TTFT: {agg_rr_ttft/agg_kv_ttft:.1f}x lower with KV Router ({agg_kv_ttft:.0f}ms vs {agg_rr_ttft:.0f}ms)")

agg_kv_itl = results['agg_kv']['Mean ITL (ms)']
agg_rr_itl = results['agg_rr']['Mean ITL (ms)']
print(f"  ITL: {agg_rr_itl/agg_kv_itl:.1f}x lower with KV Router ({agg_kv_itl:.0f}ms vs {agg_rr_itl:.0f}ms)")
print()

# Disagg: KV Router vs Round Robin  
disagg_kv_ttft = results['disagg_kv']['Mean TTFT (ms)']
disagg_rr_ttft = results['disagg_rr']['Mean TTFT (ms)']
print(f"Disagg Mode - KV Router vs Round Robin:")
print(f"  TTFT: {disagg_rr_ttft/disagg_kv_ttft:.2f}x lower with KV Router ({disagg_kv_ttft:.0f}ms vs {disagg_rr_ttft:.0f}ms)")

disagg_kv_itl = results['disagg_kv']['Mean ITL (ms)']
disagg_rr_itl = results['disagg_rr']['Mean ITL (ms)']
print(f"  ITL: {disagg_rr_itl/disagg_kv_itl:.2f}x lower with Round Robin ({disagg_rr_itl:.0f}ms vs {disagg_kv_itl:.0f}ms)")


# KV router: Disagg vs agg
print(f"\n\nKV Router - Disagg vs Agg:")
disagg_kv_ttft = results['disagg_kv']['Mean TTFT (ms)']
agg_kv_ttft = results['agg_kv']['Mean TTFT (ms)']
print(f"  TTFT: {agg_kv_ttft/disagg_kv_ttft:.1f}x lower with Disagg ({agg_kv_ttft:.0f}ms vs {disagg_kv_ttft:.0f}ms)")

disagg_kv_itl = results['disagg_kv']['Mean ITL (ms)']
agg_kv_itl = results['agg_kv']['Mean ITL (ms)']
print(f"  ITL: {agg_kv_itl/disagg_kv_itl:.1f}x lower with Disagg ({disagg_kv_itl:.0f}ms vs {agg_kv_itl:.0f}ms)")
print()

# %%
# Print formatted comparison table
def print_comparison_table(results: dict):
    """Print a nicely formatted table with experiments as columns and metrics as rows.
    
    Best values are bolded using ANSI escape codes.
    """
    
    col_names = ['Disagg+KV', 'Disagg+RR', 'Agg+KV', 'Agg+RR']
    data_keys = ['disagg_kv', 'disagg_rr', 'agg_kv', 'agg_rr']
    
    # (source_name, display_name, lower_is_better)
    metrics = [
        ('Mean TTFT (ms)', 'TTFT Mean (ms)', True),
        ('Median TTFT (ms)', 'TTFT Median (ms)', True),
        ('P99 TTFT (ms)', 'TTFT P99 (ms)', True),
        ('Mean ITL (ms)', 'ITL Mean (ms)', True),
        ('Median ITL (ms)', 'ITL Median (ms)', True),
        ('P99 ITL (ms)', 'ITL P99 (ms)', True),
        ('Mean E2EL (ms)', 'E2EL Mean (ms)', True),
        ('Median E2EL (ms)', 'E2EL Median (ms)', True),
        ('P99 E2EL (ms)', 'E2EL P99 (ms)', True),
        ('Output Token Throughput (tok/s)', 'Throughput (tok/s)', False),  # higher is better
        ('Request Throughput (req/s)', 'Throughput (req/s)', False),       # higher is better
        ('Successful requests', 'Successful Reqs', False),  # higher is better
        ('Benchmark duration (s)', 'Duration (s)', True),   # lower is better
    ]
    
    # ANSI codes for bold
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Calculate column widths
    metric_width = max(len(m[1]) for m in metrics)
    col_width = 12
    
    # Header
    header = f"{'Metric':<{metric_width}}"
    for col in col_names:
        header += f" | {col:>{col_width}}"
    
    separator = "-" * len(header)
    
    print("\n" + "=" * len(header))
    print("BENCHMARK COMPARISON TABLE (best in bold)")
    print("=" * len(header))
    print(header)
    print(separator)
    
    # Rows
    for src_name, display_name, lower_is_better in metrics:
        # Get all values for this metric
        values = [results[dk].get(src_name, None) for dk in data_keys]
        
        # Find best value (min or max depending on metric)
        valid_values = [v for v in values if v is not None]
        if valid_values:
            best_val = min(valid_values) if lower_is_better else max(valid_values)
        else:
            best_val = None
        
        row = f"{display_name:<{metric_width}}"
        for val in values:
            if val is not None:
                formatted = f"{val:>{col_width},.2f}"
                # Bold if this is the best value
                if val == best_val:
                    formatted = f"{BOLD}{formatted}{RESET}"
                row += f" | {formatted}"
            else:
                row += f" | {'N/A':>{col_width}}"
        print(row)
    
    print("=" * len(header) + "\n")

print_comparison_table(results)