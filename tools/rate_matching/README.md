# Rate-Matching Sweep Tool

Automated pipeline for finding optimal CTX/GEN GPU allocations in disaggregated LLM inference deployments on SLURM clusters.

## What It Does

Rate matching determines the ideal ratio of prefill (CTX) GPUs to decode (GEN) GPUs for a given model + workload. The tool:

1. **Measures CTX speed-of-light (SOL)** — how fast can prefill go at maximum batch?
2. **Measures GEN SOL** — sweeps decode throughput across concurrency levels, modes (TEP/DEP), and MTP configurations
3. **Computes rate-matching** — finds the CTX:GEN ratio where prefill exactly feeds decode without bottlenecking either side
4. **Extracts Pareto frontier** — identifies optimal configs balancing interactivity (TPOT) vs throughput
5. **Validates via E2E benchmarks** — runs full prefill+decode benchmarks at each Pareto point to verify SOL predictions hold
6. **Generates dashboards** — interactive Plotly charts and CSV/JSON exports

## Glossary

| Term | Meaning |
|---|---|
| **SOL** | Speed-of-light — the theoretical maximum throughput of a component (CTX or GEN) running in isolation on dedicated hardware. SOL benchmarks measure what's achievable *without* cross-component interference. |
| **CTX** | Context / prefill stage — processes the input prompt. GPU-compute-bound. |
| **GEN** | Generation / decode stage — produces output tokens one step at a time. Memory-bandwidth-bound. |
| **TEP** | Tensor Expert Parallel — each decode worker sees all requests. Lower latency, limited by memory. |
| **DEP** | Disaggregated Expert Parallel — requests are distributed across workers. Higher throughput, higher latency. |
| **MTP** | Multi-Token Prediction — model predicts multiple tokens per decode step. `mtp_accept_rate` is effective tokens per step per user (model-dependent, must be measured). |
| **TPOT** | Time Per Output Token — key interactivity metric (lower = more responsive). |
| **TTFT** | Time To First Token — latency from request to first output token. |
| **Pareto frontier** | Set of non-dominated configurations: you can't improve throughput without sacrificing interactivity, and vice versa. Ranked by throughput/GPU. |
| **Concurrency multiplier** | Scaling factor applied to SOL-predicted concurrency for E2E validation. `1.0x` = exact match, `1.05x` = 5% headroom, `0.95x` = conservative. |

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Sweep YAML Config                                    │
│  (model, workload, resources, gen_sweep groups, settings)                   │
└──────────────┬──────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  Phase 1: Generate       │  Produces srt-slurm YAML configs for each job.
│  Configs                 │  One CTX config + one GEN config per concurrency.
└──────────────┬───────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌───────────────┐ ┌───────────────────────────────────────────────────────┐
│ Phase 2: CTX  │ │ Phase 3: GEN SOL (one SLURM job per concurrency)     │
│ SOL (1 job)   │ │   TEP c1, c2, c4, c8, c16, c32, c64, c128           │
│               │ │   DEP c64, c128, c256, c512, c1024, ...              │
│ Prefill-only  │ │   Each with correct TLLM_BENCHMARK_REQ_QUEUES_SIZE   │
│ benchmark     │ │                                                       │
└───────┬───────┘ └──────────────────────┬────────────────────────────────┘
        │                                │
        └────────────┬───────────────────┘
                     ▼
        ┌────────────────────────┐
        │ Phase 4: Rate-Matching │  For each GEN config: compute CTX:GEN ratio,
        │ Math                   │  total GPUs, throughput/GPU, interactivity.
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────┐
        │ Phase 5: Pareto        │  Extract non-dominated set: best throughput
        │ Frontier               │  for each interactivity level.
        └────────────┬───────────┘
                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │ Phase 6: E2E Validation                                    │
        │   For each Pareto point × multiplier (e.g. 0.95x, 1.0x,  │
        │   1.05x), run a full prefill+decode benchmark with the    │
        │   computed GPU allocation. Compare TPOT & throughput vs   │
        │   SOL predictions.                                         │
        └────────────┬───────────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │ Phase 7: Dashboard     │  Plotly charts, CSV/JSON exports,
        │ & Summary              │  Pareto frontier visualization.
        └────────────────────────┘
```

## Prerequisites

- **Python 3.10+**
- **srt-slurm** installed (`pip install -e .` from repo root)
- **SLURM access** — must be able to run `sbatch`, `squeue`, `sacct` from the login node
- **`srtslurm.yaml`** configured for your cluster (model paths, partitions, GPU types)
- **Model already built** — TRT-LLM engines must be pre-built and referenced in `srtslurm.yaml`
- **Plotly** (installed automatically with srt-slurm)

## Quick Start

```bash
# Install (from srt-slurm repo root)
pip install -e .

# Validate config without submitting (generates configs in ./sweeps/)
srtctl-rate-match dry-run -f tools/rate_matching/h200_1k1k_mtp_sweep.yaml

# Run inside tmux (sweeps take hours!)
tmux new -s sweep
srtctl-rate-match run -f tools/rate_matching/h200_1k1k_mtp_sweep.yaml

# Monitor progress (from another terminal)
srtctl-rate-match status -o ./sweeps/dsr1_1k1k_mtp_20260213_043718
srtctl-rate-match status -o ./sweeps/dsr1_1k1k_mtp_20260213_043718 --live

# Add more E2E multipliers to a completed sweep
srtctl-rate-match add-e2e -o ./sweeps/dsr1_1k1k_mtp_20260213_043718 --multipliers 0.90 1.10

# Re-process with updated config (no resubmission)
srtctl-rate-match reprocess -o ./sweeps/dsr1_1k1k_mtp_20260213_043718 -f updated.yaml

# Cancel all jobs
srtctl-rate-match cancel -o ./sweeps/dsr1_1k1k_mtp_20260213_043718
```

## End-to-End Walkthrough

This walks through a typical workflow from scratch.

### Step 1: Create a sweep config

Copy the example and edit for your model/workload:

```bash
cp tools/rate_matching/h200_1k1k_mtp_sweep.yaml my_sweep.yaml
```

Key things to set:
- `model.path` — model alias in your `srtslurm.yaml`
- `workload.isl` / `workload.osl` — your target input/output sequence lengths
- `workload.mtp_accept_rates` — measured MTP accept rates for your model (skip if not using MTP)
- `resources.max_total_gpus` — budget cap for GPU allocation search
- `gen_sweep` groups — configure which modes (TEP/DEP), concurrencies, batch sizes, and MTP levels to sweep

### Step 2: Dry run to validate

```bash
srtctl-rate-match dry-run -f my_sweep.yaml
```

This generates all SLURM configs in `./sweeps/<name>_<timestamp>/configs/` without submitting anything. Inspect them to verify they look correct.

### Step 3: Run the sweep

Always run inside `tmux` or `screen` — sweeps typically take 2-6 hours:

```bash
tmux new -s sweep
srtctl-rate-match run -f my_sweep.yaml
```

The orchestrator will:
1. Submit a CTX SOL job and wait for it to complete
2. Submit GEN SOL jobs (in parallel by default) and wait
3. Compute rate-matching math and extract the Pareto frontier
4. Submit E2E validation jobs for each Pareto point × multiplier
5. Generate dashboards and a summary

### Step 4: Monitor progress

From another terminal:

```bash
srtctl-rate-match status -o ./sweeps/my_sweep_20260213_120000
```

Or with auto-refresh:

```bash
srtctl-rate-match status -o ./sweeps/my_sweep_20260213_120000 --live
```

### Step 5: Interpret results

When the sweep completes, look at:

```
sweeps/my_sweep_20260213_120000/
├── dashboard/
│   ├── *_pareto_frontier.html    ← Open this first
│   ├── *_sol_vs_e2e.html         ← Then this
│   └── *_ttft_analysis.html      ← And this
└── results/
    ├── *_frontier.csv            ← Pareto points as a table
    └── *_sol_vs_e2e.csv          ← SOL vs E2E comparison
```

### Step 6: Iterate

Want to test additional concurrency multipliers?

```bash
srtctl-rate-match add-e2e -o ./sweeps/my_sweep_20260213_120000 --multipliers 0.90 1.10
```

Changed MTP accept rates or random_ratio?

```bash
srtctl-rate-match reprocess -o ./sweeps/my_sweep_20260213_120000 -f updated_config.yaml
```

## Interpreting Results

### Pareto Frontier Chart (`*_pareto_frontier.html`)

- **X-axis**: Interactivity (tok/s/user) — higher means more responsive per-user experience
- **Y-axis**: Output throughput per GPU (tok/s/GPU) — higher means better hardware utilization
- **Grey dots**: All rate-matching configurations (non-Pareto)
- **Blue stars + dashed line**: Pareto frontier — the optimal tradeoff curve
- **Labels**: Config names like `tep_c32_mtp3` (mode, concurrency, MTP level)

**How to read it**: Points on the frontier are the "best" configs. Pick the one that matches your priority:
- Need maximum throughput? → rightmost frontier point (usually DEP with high concurrency)
- Need low latency? → leftmost frontier point (usually TEP with low concurrency)
- Need a balance? → points in the middle of the curve

### SOL vs E2E Chart (`*_sol_vs_e2e.html`)

- **Circles**: SOL predictions (from isolated CTX + GEN benchmarks)
- **Diamonds**: E2E measurements (from full prefill+decode benchmarks)
- **Grey dotted lines**: Connect SOL prediction to corresponding E2E result
- **Grouped by multiplier**: Different colors for 1.0x, 1.05x, etc.

**How to read it**: If diamonds are close to circles, SOL predictions are accurate. Large gaps mean the system behaves differently under end-to-end load (queue contention, memory pressure, etc.).

Below the chart is an HTML table showing exact numbers with pass/fail verdicts:
- **TPOT pass**: E2E TPOT within `tpot_tolerance_pct` of SOL prediction
- **Throughput pass**: E2E throughput within `throughput_tolerance_pct` of SOL prediction
- **TTFT pass**: Median TTFT below `ttft_constraint_ms` (if configured)

### TTFT Analysis Chart (`*_ttft_analysis.html`)

- **Bar chart**: Median time-to-first-token for each configuration
- **Red dashed line**: TTFT constraint threshold (if configured)
- **Grouped by multiplier**

### Result CSVs

`*_frontier.csv` contains one row per Pareto point with columns:

| Column | Meaning |
|---|---|
| `pareto_rank` | 1 = highest interactivity |
| `config_name` | e.g. `tep_c32_b128_mtp3` |
| `mode` | `tep` or `dep` |
| `concurrency` | Per-worker concurrency |
| `interactivity` | tok/s/user |
| `tpot_ms` | Time per output token (ms) |
| `output_tput_per_gpu` | Output throughput per GPU |
| `ratio_str` | CTX:GEN allocation (e.g. `4:1`) |
| `total_gpus` | Total GPUs used |
| `estimate_e2e_latency_s` | Estimated TTFT + decode time |

`*_sol_vs_e2e.csv` contains one row per (Pareto point, multiplier) with SOL predictions, E2E measurements, percentage differences, and pass/fail verdicts.

### What to do when results look wrong

| Symptom | Likely cause | Action |
|---|---|---|
| E2E TPOT much worse than SOL | Queue contention at high multiplier | Try lower multiplier (0.95x) or more GEN GPUs |
| E2E throughput much lower than SOL | Prefill bottleneck or memory pressure | Check CTX GPU count, consider more CTX instances |
| All E2E TTFT fails | CTX instances overwhelmed | Increase `max_total_gpus` or reduce concurrency multiplier |
| Pareto frontier has very few points | GEN sweep didn't cover enough concurrencies | Add more concurrency levels to `gen_sweep` |
| GEN job stuck at PENDING | SLURM partition full | Check `squeue -u $USER`, wait or cancel |

## CLI Commands

| Command | Description |
|---|---|
| `run -f config.yaml` | Run full sweep (submit SLURM jobs) |
| `run --resume -o dir` | Resume interrupted sweep from checkpoint |
| `run --detach` | Run in background (nohup-style) |
| `dry-run -f config.yaml` | Validate and generate configs only |
| `status -o dir [--live]` | Show sweep progress dashboard |
| `cancel -o dir [-y]` | Cancel all SLURM jobs for a sweep |
| `add-e2e -o dir --multipliers 0.95` | Add E2E jobs to existing sweep |
| `reprocess -o dir [-f new.yaml]` | Re-derive metrics from logs |

### When to Use `--resume` vs `reprocess` vs `add-e2e`

These three commands serve different purposes. Use this decision guide:

```
What happened?
│
├─ Orchestrator was interrupted (SSH drop, Ctrl-C, crash)
│  └─ Use: srtctl-rate-match run --resume -o <dir> -f config.yaml
│     Picks up from the last saved phase. Reconciles stale "running"
│     jobs against what's actually on disk. Continues the pipeline.
│
├─ Sweep finished, but I changed config parameters (mtp_accept_rates,
│  random_ratio, tolerances) and want to re-derive metrics
│  └─ Use: srtctl-rate-match reprocess -o <dir> -f updated.yaml
│     Re-parses all logs from disk. Recomputes rate-matching, Pareto,
│     and SOL-vs-E2E. Regenerates dashboards. No SLURM jobs submitted.
│
├─ Sweep finished, but I want to test additional E2E multipliers
│  (e.g. add 0.90x and 1.10x alongside existing 0.95x, 1.0x, 1.05x)
│  └─ Use: srtctl-rate-match add-e2e -o <dir> --multipliers 0.90 1.10
│     Adds new E2E jobs without touching SOL phases. Creates a state
│     backup. Only submits jobs for new (pareto_rank, multiplier) pairs.
│
└─ Sweep finished, but I want to change the GEN sweep parameters
   (new concurrency levels, batch sizes, modes)
   └─ Not currently supported incrementally. Run a new sweep.
      (See "Current Limitations" below.)
```

## Sweep Config Format

See [`h200_1k1k_mtp_sweep.yaml`](h200_1k1k_mtp_sweep.yaml) for a complete example.

### Required Sections

```yaml
name: my_sweep          # Used for output directory naming

model:
  path: dsr1             # Model alias (resolved via srtslurm.yaml)
  container: "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1.post1"
  precision: fp8

workload:
  isl: 1024              # Input sequence length
  osl: 1024              # Output sequence length
  random_ratio: 0.8      # Controls ISL/OSL variance; avg = (ratio + 1) / 2
  mtp_accept_rates:      # Required when using MTP (model-dependent)
    1: 1.8
    2: 2.28
    3: 2.56

resources:
  gpu_type: h200
  gpus_per_node: 8
  ctx_gpus_per_instance: 8
  gen_gpus_per_instance: 8
  max_total_gpus: 64     # Upper bound for rate-matching GPU search
```

### CTX Config (Optional)

```yaml
ctx_config:
  benchmark_concurrency: 64   # sa-bench concurrency for CTX SOL benchmark
```

If omitted, a sensible default is used.

### GEN Sweep Groups

Each named group expands into one or more `GenSweepItem`s via `zip` or `grid` expansion:

```yaml
gen_sweep:
  mtp3_tep:
    expansion: zip       # zip: pair parameters 1:1; grid: full cartesian product
    parameters:
      concurrency: [[1, 2, 4, 8, 16, 32, 64, 128]]
      batch_size: [128]
      max_num_tokens: [512]
      gpu_memory_fraction: [0.9]
    defaults:
      mode: tep           # tep or dep
      tp_size: 8
      mtp_num: 3          # 0 = STP (single-token prediction)
```

**Expansion modes**:
- `zip` — pairs elements positionally. All parameter lists must have the same length (or length 1 for broadcasting). Use this when you have one set of parameters per concurrency level.
- `grid` — full cartesian product of all parameter lists. Use this to sweep multiple dimensions simultaneously (e.g., all combinations of batch_size × concurrency).

### Settings

```yaml
settings:
  poll_interval: 180          # Seconds between SLURM status checks
  max_retries: 2              # Retry failed jobs up to N times
  max_poll_time: 14400        # Timeout per job (seconds, default 4h)
  run_e2e_validation: true    # Set false to skip E2E phase
  parallel_submissions: true  # Submit all jobs at once vs one-at-a-time

  e2e_validation:
    concurrency_multipliers: [0.95, 1.0, 1.05]  # System concurrency = SOL × multiplier
    tpot_tolerance_pct: 15.0      # Max acceptable TPOT degradation vs SOL
    throughput_tolerance_pct: 20.0 # Max acceptable throughput degradation vs SOL
    ttft_constraint_ms: 5000.0    # TTFT pass/fail threshold (optional)
```

### Engine Type (Optional)

```yaml
engine_type: trtllm    # Default. Only trtllm is currently supported.
```

This controls which log parser is used. When vLLM or SGLang support is added, set this to `vllm` or `sglang`.

### Retry Semantics

When a SLURM job fails (non-zero exit, timeout, or SLURM failure):
1. The job's `status` is set to `"failed"` and `job_id` is cleared
2. The failure is recorded in `retry_history`
3. The job is resubmitted (up to `max_retries` times)
4. If all retries are exhausted, the job stays `"failed"` and the sweep continues with remaining jobs

You cannot retry a single job — the orchestrator handles retries automatically. If a specific job keeps failing, check its SLURM output logs (in the job's `output_dir`).

## Output Structure

```
sweeps/my_sweep_20260213_043718/
├── sweep_state.json              # Central state file (auto-saved, atomic writes)
├── sweep_state.json.bak.*        # Timestamped backups (before add-e2e mutations)
├── configs/                      # Generated srt-slurm YAML configs
│   ├── ctx_sol.yaml
│   ├── gen_sol_tep_c1_mtp3.yaml
│   ├── gen_sol_tep_c2_mtp3.yaml
│   └── ...
├── e2e_pareto_configs/           # E2E validation configs (one per Pareto × multiplier)
├── results/                      # CSV/JSON exports
│   ├── *_all.csv                 # All rate-matching results
│   ├── *_frontier.csv            # Pareto frontier only
│   ├── *_sol_vs_e2e.csv          # SOL vs E2E comparison
│   └── *.json                    # JSON equivalents
├── dashboard/                    # Interactive Plotly HTML charts
│   ├── *_pareto_frontier.html
│   ├── *_sol_vs_e2e.html
│   └── *_ttft_analysis.html
├── orchestrator.log              # If run with --detach
└── orchestrator.pid              # If run with --detach
```

## Rate-Matching Math

Given a model, ISL, and OSL, prefill (CTX) and decode (GEN) have different throughput characteristics. Rate matching finds the CTX:GEN GPU ratio where:

```
CTX request rate = GEN decode capacity
```

The core formulas:

```
avg_random_ratio   = (random_ratio + 1) / 2
gen_req_rate       = output_throughput / (osl × avg_random_ratio)
ctx_gen_inst_ratio = gen_req_rate / ctx_request_rate
output_tput_per_gpu = output_throughput / (ctx_gpus × ctx_gen_ratio + gen_gpus)
```

Too many CTX GPUs → prefill generates requests faster than decode can process them (queue buildup).
Too many GEN GPUs → decode workers are idle waiting for prefill.

The tool searches for the integer allocation (`ctx_instances`, `gen_instances`) that maximizes `output_tput_per_gpu` within the `max_total_gpus` budget.

## Resilience Features

### SSH Disconnect Protection

Sweeps run for hours. The orchestrator is designed to survive interruptions:

- **Signal handling**: `SIGHUP` (SSH disconnect), `SIGTERM`, `SIGINT` all save state before exiting
- **Atomic state saves**: `sweep_state.json` is written via temp-file-then-rename (never corrupt)
- **Per-job persistence**: state is saved after each individual job submission, not just at phase boundaries
- **`--resume`**: continues from the last saved checkpoint, reconciling stale "running" jobs against disk
- **`--detach`**: backgrounds the orchestrator (nohup-style) so it survives terminal close
- **`add-e2e`**: adds new multipliers without re-running SOL phases or risking state corruption
- **State backup**: `add-e2e` creates a timestamped backup before any mutation
- **Overwrite guard**: `run` refuses to overwrite existing state without `--resume`

### Reconciliation

If the orchestrator is killed, SLURM jobs keep running. On `--resume` or `reprocess`:
- CTX jobs: checked for logs on disk
- GEN jobs: checked for logs on disk
- E2E jobs: checked for sa-bench result JSON on disk
- Stale "running"/"submitted"/"pending" statuses are promoted to "completed" if results exist

## Troubleshooting

### Common Issues

| Problem | Diagnosis | Fix |
|---|---|---|
| `RuntimeError: sweep_state.json already exists` | Tried `run` on an existing sweep without `--resume` | Use `--resume -o <dir>` to continue, or choose a different output dir |
| Jobs stuck at "PENDING" in SLURM | Partition full or quota exceeded | `squeue -u $USER` to check; wait, cancel other jobs, or change partition |
| "Insufficient data after filtering" during CTX/GEN processing | Job completed but had too few iterations after warmup/cooldown trimming | Job may have been too short. Check logs, consider longer benchmarks |
| "No parser registered for engine X" | `engine_type` in config doesn't match any registered parser | Check spelling. Only `trtllm` is currently supported |
| E2E TPOT much higher than SOL | Normal — E2E includes real queue contention | Try lower multiplier. Large gaps (>30%) may indicate memory pressure |
| `add-e2e` says "No Pareto frontier found" | Sweep didn't reach the Pareto phase | Run the sweep to completion first |
| Reprocess gives different results | Config parameters changed (mtp_accept_rates, random_ratio) | Expected — reprocess uses the new config to re-derive metrics |
| State file corrupted (rare) | Process killed during pre-atomic-save era | Restore from `sweep_state.json.bak.*` if available |

### Recovering from State Corruption

If `sweep_state.json` is corrupt or lost:

1. Check for backups: `ls sweep_state.json.bak.*`
2. If a backup exists: `cp sweep_state.json.bak.<timestamp> sweep_state.json`
3. Then: `srtctl-rate-match reprocess -o <dir> -f config.yaml`

Reprocess will re-read all logs from disk and reconstruct results. SLURM job IDs and submission history will be preserved from the state file.

## Module Architecture

```
cli.py                  CLI entry point (argparse subcommands)
schema.py               Pydantic config schema (YAML → Python objects)
run_sweep.py            Main orchestrator (phases 1–7, add-e2e, reprocess)
state.py                SweepState class (persistence, backup, job records)
generate_configs.py     Config generation (CTX SOL, GEN SOL, E2E)
slurm_helpers.py        SLURM interaction (submit, poll, retry)
export.py               Result loading and CSV/JSON export
parser_base.py          Abstract parser base classes + registry
process_ctx_results.py  TRT-LLM CTX (prefill) log parser
process_gen_results.py  TRT-LLM GEN (decode) log parser
metrics.py              Rate-matching math, SOL vs E2E comparison
pareto.py               Pareto frontier extraction
dashboard_export.py     Plotly chart generation
sweep_status.py         Status dashboard (reads sweep_state.json)
```

### Parser Registry

Engine-specific parsers register themselves at import time:

```python
# In process_ctx_results.py
@register_ctx_parser("trtllm")
class TrtllmCTXLogParser(CTXLogParser):
    ...
```

The orchestrator resolves parsers via:
```python
ctx_parser = get_ctx_parser(cfg.engine_type)  # "trtllm" → TrtllmCTXLogParser
gen_parser = get_gen_parser(cfg.engine_type)
```

## Extending: Adding a New Engine

The parser registry makes it straightforward to add support for vLLM, SGLang, or other inference engines. Each engine needs:

1. **CTX parser** — subclass `CTXLogParser` from `parser_base.py`
2. **GEN parser** — subclass `GENLogParser` from `parser_base.py`
3. **Register** via `@register_ctx_parser("engine_name")` / `@register_gen_parser("engine_name")`
4. **Import** in `run_sweep.py` so the decorator runs at startup
5. **Set** `engine_type: "engine_name"` in the sweep YAML

### What each parser must implement

| Method | Description |
|---|---|
| `find_log(logs_dir)` | Locate the relevant log file in a job's output directory |
| `parse(log_file)` | Parse raw log lines into a list of iteration dicts |
| `process(data, ...)` | Filter, aggregate, and compute metrics from parsed data |

### Return contracts

CTX parsers must return a dict with these fields (or `{"error": "description"}` on failure):

| Field | Type |
|---|---|
| `ctx_throughput_tokens_per_s` | float |
| `request_rate_req_per_s` | float |
| `avg_prev_device_step_time_ms` | float |
| `num_iterations` | int |
| `num_ranks` | int |
| `isl` | int |

GEN parsers must return a dict with these fields (or `{"error": "description"}` on failure):

| Field | Type |
|---|---|
| `interactivity` | float |
| `throughput_per_gpu` | float |
| `output_throughput` | float |
| `tpot_ms` | float |
| `avg_step_time_ms` | float |
| `concurrency` | int |
| `mode` | str |
| `mtp` | int |
| `mtp_accept_rate` | float |
| `num_gpus` | int |

### Example: Adding vLLM support

```python
# process_ctx_results_vllm.py
from parser_base import CTXLogParser, register_ctx_parser

@register_ctx_parser("vllm")
class VllmCTXLogParser(CTXLogParser):
    def find_log(self, logs_dir):
        # vLLM writes to a different log format
        for f in logs_dir.glob("vllm_prefill_*.log"):
            return f
        return None

    def parse(self, log_file, verbose=False):
        # Parse vLLM's log format into iteration dicts
        ...

    def process(self, data, isl, *, verbose=False, max_batch_size=None):
        # Apply filtering, compute request_rate, throughput
        ...
```

Then in `run_sweep.py`:
```python
import process_ctx_results_vllm as _vllm_ctx_mod  # noqa: F401
```

And in the sweep YAML:
```yaml
engine_type: vllm  # instead of trtllm
```

## Current Limitations

- **TRT-LLM only**: Only the TRT-LLM engine has parser implementations. vLLM and SGLang parsers need to be written.
- **TP=8 only in practice**: Multi-node decode workers (TP>8) require multiple nodes per worker, which srt-slurm doesn't currently support.
- **Single-model sweeps**: Each sweep handles one model × one workload (ISL/OSL). Comparing across models requires separate sweeps.
- **No automatic MTP accept rate measurement**: `mtp_accept_rates` must be provided in the config. A future enhancement could auto-measure these.
- **No cross-sweep comparison tool**: Comparing results across different sweeps (e.g. different models or ISL/OSL) requires manual effort.
- **Sequential CTX benchmark**: Only one CTX SOL job is run. Multi-batch-size CTX sweeps could improve accuracy.
- **No incremental GEN additions**: Can't add new concurrency levels or modes to an existing sweep — only E2E additions are supported via `add-e2e`.

## Future Improvements

- **vLLM / SGLang parsers**: Implement parser classes for additional inference engines using the existing registry pattern.
- **Multi-node TP support**: Enable TP>8 with multi-node decode workers when srt-slurm supports it.
- **Auto MTP accept rate**: Benchmark MTP accept rates as a preliminary phase before the main sweep.
- **Cross-sweep comparison**: Dashboard tool to overlay results from multiple sweeps (different models, ISL/OSL pairs).
- **TTFT-aware Pareto**: Include time-to-first-token as a Pareto dimension alongside TPOT and throughput.
- **Cost-aware optimization**: Factor in GPU-hours and cost when ranking Pareto points.
- **Incremental GEN additions**: Support adding new concurrency levels or modes to an existing sweep (currently only E2E additions are supported via `add-e2e`).
