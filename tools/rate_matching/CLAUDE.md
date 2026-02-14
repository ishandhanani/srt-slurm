# CLAUDE.md — Rate-Matching Module

Development guide for the `tools/rate_matching/` module. This module is a self-contained Python package within the srt-slurm repo that orchestrates multi-phase GPU benchmarking sweeps on SLURM clusters.

## Quick Reference

```bash
# Run rate-matching tests (uses module-local .venv, NOT the repo-level uv)
cd /path/to/srt-slurm
.venv/bin/python -m pytest tests/test_rate_matching.py -v --tb=short

# Run a single test class
.venv/bin/python -m pytest tests/test_rate_matching.py::TestAddE2E -v

# Dry-run a sweep config (no SLURM, just validates + generates configs)
srtctl-rate-match dry-run -f tools/rate_matching/h200_1k1k_mtp_sweep.yaml

# Check CLI help
srtctl-rate-match --help
srtctl-rate-match run --help
```

## Architecture Overview

The module implements a 7-phase pipeline orchestrated by `run_sweep.py`:

```
Phase 1 (init→ctx):     Generate SLURM configs from sweep YAML
Phase 2 (ctx→gen):      Submit CTX SOL benchmark, parse results
Phase 3 (gen→rate_match): Submit GEN SOL benchmarks, parse results
Phase 4 (rate_match→pareto): Compute rate-matching math
Phase 5 (pareto→e2e):   Extract Pareto frontier
Phase 6 (e2e→complete): Submit E2E validation, compare SOL vs E2E
Phase 7:                Generate dashboards and exports
```

State is tracked in `sweep_state.json` via `SweepState` (in `state.py`). Phase transitions are guarded by `if state.phase == "X"` checks in `run_sweep()`.

### File Responsibilities

| File | Role |
|---|---|
| `cli.py` | CLI entry point (`srtctl-rate-match`). Argparse subcommands: run, dry-run, status, cancel, add-e2e, reprocess |
| `schema.py` | Pydantic config schema. YAML → `RateMatchingSweepConfig` object |
| `run_sweep.py` | Main orchestrator. All phase functions, `run_sweep()`, `reprocess_sweep()`, `add_e2e_jobs()`, signal handling, reconciliation |
| `state.py` | `SweepState` class. Atomic JSON persistence, backup, job record TypedDicts |
| `generate_configs.py` | Generates srt-slurm YAML configs for CTX SOL, GEN SOL, and E2E jobs |
| `slurm_helpers.py` | SLURM interaction: `_submit_and_poll()` (serial), `_submit_poll_parallel()` (parallel). Submit via sbatch, poll via squeue/sacct |
| `parser_base.py` | Abstract base classes (`CTXLogParser`, `GENLogParser`) + decorator-based registry |
| `process_ctx_results.py` | TRT-LLM CTX log parser. Registered as `@register_ctx_parser("trtllm")` |
| `process_gen_results.py` | TRT-LLM GEN log parser. Registered as `@register_gen_parser("trtllm")` |
| `metrics.py` | Rate-matching math (`compute_rate_matching`), SOL vs E2E comparison (`compare_sol_vs_e2e`) |
| `pareto.py` | Pareto frontier extraction (`extract_pareto_frontier`) |
| `export.py` | CSV/JSON export of results |
| `dashboard_export.py` | Plotly chart generation (Pareto frontier, SOL vs E2E, TTFT analysis) |
| `sweep_status.py` | Status dashboard (reads `sweep_state.json`, shows progress) |

### Dependency Flow

```
cli.py → run_sweep.py → {schema, state, generate_configs, slurm_helpers,
                          parser_base, process_ctx_results, process_gen_results,
                          metrics, pareto, export, dashboard_export}
```

`run_sweep.py` is the hub. Other modules are mostly leaves. Parser modules self-register via decorators on import.

## Key Patterns

### Import Convention

The module uses `sys.path` manipulation, NOT package imports:

```python
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from schema import RateMatchingSweepConfig  # bare name, not tools.rate_matching.schema
```

### Parser Registry

Parsers register via decorators that fire on import:

```python
@register_ctx_parser("trtllm")
class TrtllmCTXLogParser(CTXLogParser): ...
```

`run_sweep.py` force-imports parser modules so decorators fire:

```python
import process_ctx_results as _ctx_mod  # noqa: F401
```

### State Persistence

- `SweepState.save()` — atomic writes (temp file + `os.replace`)
- `SweepState.save_backup()` — timestamped copy before destructive mutations
- `slurm_helpers` calls `state.save()` after EACH job status change (not just phase boundaries)
- Signal handlers (`SIGHUP`, `SIGTERM`, `SIGINT`) save state before exiting

### Job Status Lifecycle

```
pending → submitted → running → completed
                              → failed (retried up to max_retries, then stays failed)
```

Stale statuses reconciled on `--resume` or `reprocess`: `running`/`submitted`/`pending` → `completed` if results exist on disk.

## Data Structures

### CTXResult (success)

```python
{"ctx_throughput_tokens_per_s": float, "request_rate_req_per_s": float,
 "avg_prev_device_step_time_ms": float, "num_iterations": int,
 "num_ranks": int, "isl": int}
```

### GENResult (success)

```python
{"interactivity": float, "throughput_per_gpu": float, "output_throughput": float,
 "tpot_ms": float, "avg_step_time_ms": float, "concurrency": int, "mode": str,
 "mtp": int, "mtp_accept_rate": float, "num_gpus": int}
```

### Rate-matching result

Key fields: `config_name`, `mode`, `concurrency`, `mtp_num`, `interactivity`, `tpot_ms`, `output_tput_per_gpu`, `ctx_gen_inst_ratio`, `ratio_str`, `ctx_instances`, `gen_instances`, `total_gpus`.

### Pareto entry

Rate-matching result + `pareto_rank` (int, 1 = highest interactivity) + `is_pareto_optimal` (True).

### SOL vs E2E entry

```python
{"pareto_rank": int, "multiplier": float, "sol": {...}, "e2e": {...},
 "diff_pct": {"tpot": float, "throughput": float},
 "pass": {"tpot": bool, "throughput": bool, "overall": bool}}
```

## Testing

### Test Location

All tests: `tests/test_rate_matching.py`

### How to Run

```bash
# Full suite
.venv/bin/python -m pytest tests/test_rate_matching.py -v --tb=short

# Single class
.venv/bin/python -m pytest tests/test_rate_matching.py::TestCTXProcessing -v

# Single test
.venv/bin/python -m pytest tests/test_rate_matching.py::TestAddE2E::test_add_new_multiplier_dry_run -v
```

**Important**: Use `.venv/bin/python -m pytest`, NOT `uv run pytest`. The rate-matching module has its own virtualenv.

### Test Patterns

- **Processing tests**: Generate synthetic log data → `parser.parse()` → `parser.process()` → assert fields
- **Config tests**: Build `RateMatchingSweepConfig` → `generate_*_config()` → assert YAML content
- **State tests**: `SweepState()` → `save()` / `load()` / `save_backup()` → verify JSON + atomicity
- **Signal tests**: Mock `signal.signal` → `_install_signal_handlers()` → verify registration
- **Reconciliation tests**: Create stale jobs → mock `find_log()` → `_reconcile_stale_jobs()` → verify status changes
- **add-e2e tests**: Create completed state with Pareto → `add_e2e_jobs(dry_run=True)` → verify new configs/jobs
- **SLURM tests**: Mock `subprocess.run` for sbatch/squeue/sacct

Use `tmp_path` fixture for any filesystem tests. Mock SLURM interactions via `unittest.mock.patch`.

### When to Add Tests

Always add tests for:
- New phase functions or modifications to existing phases
- New CLI subcommands
- Changes to state persistence logic
- New parser implementations
- Changes to rate-matching math or Pareto extraction

## Common Tasks

### Adding a New Engine Parser

1. Create `process_ctx_results_<engine>.py` with `@register_ctx_parser("<engine>")`
2. Create `process_gen_results_<engine>.py` with `@register_gen_parser("<engine>")`
3. Add `import process_ctx_results_<engine> as _mod  # noqa: F401` in `run_sweep.py`
4. Add tests in `test_rate_matching.py`
5. Document in README.md

### Adding a New CLI Subcommand

1. Add `cmd_<name>(args)` function in `cli.py`
2. Add argparse subparser in `build_parser()` at the bottom of `cli.py`
3. Wire up in the `cmd_map` dict
4. Add corresponding logic in `run_sweep.py` if needed
5. Add tests

### Adding a New Phase

1. Add phase function in `run_sweep.py` following `phaseN_xxx(cfg, state, ...)` pattern
2. Add phase transition guard in `run_sweep()` main function
3. Add corresponding logic in `reprocess_sweep()` if applicable
4. Update `state.phase` comment in `state.py` to include new phase name
5. Add tests

### Modifying Rate-Matching Math

All formulas are in `metrics.py`. Key function: `compute_rate_matching()`. Changes here affect:
- `state.rate_matching_results`
- `state.pareto_frontier` (derived from rate-matching results)
- CSV/JSON exports
- Dashboard charts

After changes, run the full test suite and visually verify dashboard output with a `reprocess`.

## Gotchas

1. **sys.path imports** — all imports are bare module names. Don't use `from tools.rate_matching.X import Y`.
2. **Parser imports are side effects** — if you add a parser file but forget to import it in `run_sweep.py`, the registry won't have it and `get_*_parser()` raises `KeyError`.
3. **slurm_helpers mutates job dicts in place** — it receives references to dicts inside `state.gen_jobs` / `state.e2e_jobs` and modifies them directly, then calls `state.save()`.
4. **E2E identity is (pareto_rank, multiplier)** — `add_e2e_jobs()` uses this pair to detect duplicates.
5. **Reprocess vs resume** — `reprocess` re-parses all logs and recomputes everything. `--resume` continues from the last saved phase, submitting remaining SLURM jobs. Don't confuse them.
6. **Overwrite guard** — `run_sweep()` refuses to overwrite existing `sweep_state.json` without `--resume`. This is intentional to prevent data loss.
7. **GEN jobs may span multiple concurrencies** — one SLURM job can benchmark several concurrency levels. `process_all_concurrencies()` returns results keyed by concurrency.

## Code Style

- Python 3.10+ (`|` unions, match statements OK)
- Pydantic for config schemas
- TypedDict for data contracts
- No hardcoded paths in comments (see `.cursor/rules/comment-standards.mdc`)
- Line length: 120 chars
- Follow the parent repo's CLAUDE.md for general Python patterns
