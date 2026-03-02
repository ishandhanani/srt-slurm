#!/usr/bin/env python3
"""
Rate-matching sweep orchestrator.

Drives the full pipeline:
  Phase 1: Generate configs (CTX SOL + GEN SOL)
  Phase 2: Submit & process CTX-only SOL
  Phase 3: Submit & process GEN-only SOL jobs
  Phase 4: Compute rate-matching metrics
  Phase 5: Extract Pareto frontier
  Phase 6: Generate & run E2E validation (multiple concurrency variants)
  Phase 7: Dashboard export & summary

State is persisted to sweep_state.json after each phase for resume support.

Usage (via CLI):
    srtctl-rate-match run -c sweep.yaml -o ./sweeps/my_sweep
    srtctl-rate-match run -c sweep.yaml -o ./sweeps/my_sweep --resume
"""

from __future__ import annotations

import json
import signal
import sys
from datetime import datetime
from pathlib import Path

# Force line-buffered stdout so output appears immediately in non-TTY contexts
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Ensure this directory is on the path for sibling imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from export import _export_results, _load_sa_bench_result
from generate_configs import (
    generate_ctx_sol_config,
    generate_ctx_sol_from_base,
    generate_e2e_config,
    generate_e2e_configs_from_pareto,
    generate_gen_sol_config,
    generate_gen_sol_override_config,
    get_recipe_filename,
)
from metrics import compare_sol_vs_e2e, compute_rate_matching
from pareto import extract_pareto_frontier
from parser_base import get_ctx_parser, get_gen_parser
# Import parser modules so they self-register via decorators.
import process_ctx_results as _ctx_mod  # noqa: F401
import process_gen_results as _gen_mod  # noqa: F401
import process_ctx_results_vllm as _vllm_ctx_mod  # noqa: F401
import process_gen_results_vllm as _vllm_gen_mod  # noqa: F401
import process_ctx_results_sglang as _sglang_ctx_mod  # noqa: F401
import process_gen_results_sglang as _sglang_gen_mod  # noqa: F401
from schema import RateMatchingSweepConfig, load_sweep_config
from slurm_helpers import (
    _submit_and_poll,
    _submit_poll_parallel,
    get_job_output_dir,
)
from state import SweepState


def _get_parsers(cfg: RateMatchingSweepConfig):
    """Return (ctx_parser, gen_parser) for the configured engine."""
    return get_ctx_parser(cfg.engine_type), get_gen_parser(cfg.engine_type)


# ---------------------------------------------------------------------------
# Signal handling — save state on SIGHUP / SIGTERM / SIGINT
# ---------------------------------------------------------------------------

# Module-level reference set by run_sweep() so signal handlers can save state.
_active_state: SweepState | None = None


def _graceful_shutdown(signum: int, frame) -> None:  # noqa: ANN001
    """Save sweep state and exit cleanly on termination signals.

    SIGHUP is sent when an SSH session disconnects.  SIGTERM is the default
    ``kill`` signal.  SIGINT is Ctrl-C.  In all cases we persist the current
    state so ``--resume`` can pick up where we left off.
    """
    sig_name = signal.Signals(signum).name
    state = _active_state
    if state is not None:
        try:
            state.save()
            print(
                f"\n[{sig_name}] State saved to {state.output_dir}/sweep_state.json",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"\n[{sig_name}] Failed to save state: {exc}", file=sys.stderr)
    else:
        print(f"\n[{sig_name}] No active state to save.", file=sys.stderr)

    print(
        f"[{sig_name}] Orchestrator interrupted.  SLURM jobs already submitted "
        "will continue running.\nResume with:  srtctl-rate-match run --resume "
        "-o <output_dir>",
        file=sys.stderr,
    )
    sys.exit(128 + signum)


def _install_signal_handlers() -> None:
    """Register graceful shutdown handlers for common termination signals."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _graceful_shutdown)
    # SIGHUP is not available on Windows
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_shutdown)


# ---------------------------------------------------------------------------
# Stale-job reconciliation
# ---------------------------------------------------------------------------

_STALE_STATUSES = ("running", "submitted", "pending")


def _reconcile_stale_jobs(
    state: SweepState,
    ctx_parser=None,
    gen_parser=None,
    verbose: bool = True,
) -> int:
    """Reconcile jobs stuck at stale statuses by checking for results on disk.

    When the orchestrator is interrupted (e.g. SSH disconnect), SLURM jobs
    continue running and produce results, but ``sweep_state.json`` still shows
    them as ``"running"``.  This function checks the output directory of each
    stale job and promotes it to ``"completed"`` if results exist on disk.

    Args:
        state: The ``SweepState`` to reconcile — mutated in place.
        ctx_parser: CTX log parser (needed to verify CTX logs exist).
        gen_parser: GEN log parser (needed to verify GEN logs exist).
        verbose: Print reconciliation details.

    Returns:
        Number of jobs reconciled.
    """
    reconciled = 0

    # --- CTX job ---
    ctx = state.ctx_job
    if ctx.get("status") in _STALE_STATUSES and ctx.get("job_id"):
        out_dir = ctx.get(
            "output_dir",
            get_job_output_dir(str(ctx["job_id"]), srtctl_root=state.srtctl_root),
        )
        logs_dir = Path(out_dir) / "logs"
        has_log = ctx_parser and ctx_parser.find_log(logs_dir) is not None
        if has_log:
            ctx["status"] = "completed"
            ctx["output_dir"] = out_dir
            reconciled += 1
            if verbose:
                print(f"  Reconciled CTX job {ctx['job_id']} (logs found on disk)")

    # --- GEN jobs ---
    for gj in state.gen_jobs:
        if gj.get("status") in _STALE_STATUSES and gj.get("job_id"):
            out_dir = gj.get(
                "output_dir",
                get_job_output_dir(str(gj["job_id"]), srtctl_root=state.srtctl_root),
            )
            logs_dir = Path(out_dir) / "logs"
            has_log = gen_parser and gen_parser.find_log(logs_dir) is not None
            if has_log:
                gj["status"] = "completed"
                gj["output_dir"] = out_dir
                reconciled += 1
                if verbose:
                    print(f"  Reconciled GEN job {gj['job_id']} (logs found on disk)")

    # --- E2E jobs ---
    for ej in state.e2e_jobs:
        if ej.get("status") in _STALE_STATUSES and ej.get("job_id"):
            out_dir = ej.get(
                "output_dir",
                get_job_output_dir(str(ej["job_id"]), srtctl_root=state.srtctl_root),
            )
            sa_result = _load_sa_bench_result(out_dir)
            if sa_result is not None:
                ej["status"] = "completed"
                ej["output_dir"] = out_dir
                reconciled += 1
                if verbose:
                    print(f"  Reconciled E2E job {ej['job_id']} (results found on disk)")

    if reconciled:
        state.save()
    return reconciled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_submit_path(job_dict: dict) -> str:
    """Build the submit path for a job, appending :selector if present.

    In override mode, each gen_job dict has a ``selector`` key (e.g.
    "base" or "override_c16") that selects a variant within the
    base/override YAML. The submit path becomes ``config_path:selector``.
    """
    path = job_dict["config_path"]
    selector = job_dict.get("selector")
    if selector:
        return f"{path}:{selector}"
    return path


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase1_generate_configs(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 1: Generate CTX SOL and GEN SOL configs.

    Two modes:
      - Legacy: all configs computed from sweep fields (original path)
      - Override: ctx_sol_base / gen_sol_base produce human-readable YAML
        with base/override_* format for srt-slurm's config override system
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: Generating configs")
        print("=" * 60)

    configs_dir = Path(state.output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # --- CTX SOL ---
    ctx_path = str(configs_dir / "ctx_sol.yaml")
    if cfg.ctx_sol_base is not None:
        # Override mode: use user-provided base directly
        ctx_config_dict = generate_ctx_sol_from_base(cfg, output_path=ctx_path)
        if verbose:
            print(f"  CTX SOL config (from ctx_sol_base): {ctx_path}")
    else:
        # Legacy mode: compute from sweep fields
        ctx_config_dict = generate_ctx_sol_config(cfg, output_path=ctx_path)
        if verbose:
            print(f"  CTX SOL config: {ctx_path}")

    # Extract the effective max_batch_size from the generated config.
    # This is the ground truth (resolves YAML overrides + workload defaults)
    # and is used later as the num_ctx_requests threshold when processing
    # CTX logs, ensuring the filter is GPU-agnostic.
    if cfg.engine_type in ("sglang", "vllm"):
        ctx_max_batch_size = 1
    elif cfg.engine_type == "trtllm" and "trtllm_config" in ctx_config_dict.get("backend", {}):
        ctx_max_batch_size = ctx_config_dict["backend"]["trtllm_config"]["prefill"]["max_batch_size"]
    else:
        # Fallback for base-mode configs where structure may differ
        ctx_max_batch_size = (
            ctx_config_dict.get("backend", {})
            .get("trtllm_config", {})
            .get("prefill", {})
            .get("max_batch_size", 8)
        )

    state.ctx_job = {
        "config_path": ctx_path,
        "status": "pending",
        "max_batch_size": ctx_max_batch_size,
    }
    if verbose:
        print(f"  CTX max_batch_size (threshold): {ctx_max_batch_size}")

    # --- GEN SOL ---
    if cfg.gen_sol_base is not None and "base" in cfg.gen_sol_base:
        # New mode: gen_sol_base is in base/override format → write as-is
        _generate_gen_sol_direct_override(cfg, state, configs_dir, verbose)
    elif cfg.gen_sol_base is not None:
        # Legacy override mode: gen_sol_base (flat) + gen_sweep groups → generate overrides
        _generate_gen_sol_override_mode(cfg, state, configs_dir, verbose)
    else:
        # Legacy mode: compute configs from sweep fields
        _generate_gen_sol_legacy_mode(cfg, state, configs_dir, verbose)

    if verbose:
        print(f"\n  Total: 1 CTX + {len(state.gen_jobs)} GEN = {1 + len(state.gen_jobs)} configs")

    state.phase = "ctx"
    state.save()


def _generate_gen_sol_legacy_mode(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    configs_dir: Path,
    verbose: bool,
) -> None:
    """Generate GEN SOL configs using the legacy computed path.

    One SLURM job per concurrency level (original behaviour).
    """
    from schema import GenSweepItem as _GSI

    state.gen_jobs = []
    for i, gen_item in enumerate(cfg.gen_sweep):
        conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]
        mtp_suffix = f"_mtp{gen_item.mtp_num}" if gen_item.mtp_num > 0 else ""

        for conc in conc_list:
            single_item = _GSI(
                mode=gen_item.mode,
                batch_size=gen_item.batch_size,
                concurrency=conc,
                tp_size=gen_item.tp_size,
                mtp_num=gen_item.mtp_num,
                max_num_tokens=gen_item.max_num_tokens,
                gpu_memory_fraction=gen_item.gpu_memory_fraction,
                eplb_num_slots=gen_item.eplb_num_slots,
                decode_overrides=gen_item.decode_overrides,
                prefill_overrides=gen_item.prefill_overrides,
            )
            fname = f"gen_sol_{gen_item.mode}{gen_item.tp_size}_c{conc}{mtp_suffix}.yaml"
            gen_path = str(configs_dir / fname)
            generate_gen_sol_config(cfg, single_item, output_path=gen_path)
            state.gen_jobs.append({
                "config_path": gen_path,
                "status": "pending",
                "gen_item_index": i,
                "concurrency": conc,
            })
            if verbose:
                print(f"  GEN SOL config: {fname}  (queue_size={conc})")


def _generate_gen_sol_direct_override(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    configs_dir: Path,
    verbose: bool,
) -> None:
    """Write gen_sol_base (base/override format) as-is to one YAML file.

    gen_sol_base already has `base:` + `override_*:` keys.  We write it
    directly and create one gen_job per variant (base + each override).
    gen_sweep items are auto-generated by the schema validator.
    """
    import yaml

    gen_path = str(configs_dir / "gen_sol.yaml")
    Path(gen_path).parent.mkdir(parents=True, exist_ok=True)
    with open(gen_path, "w") as f:
        yaml.dump(cfg.gen_sol_base, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    override_keys = sorted(k for k in cfg.gen_sol_base if k.startswith("override_"))
    all_variants = ["base", *override_keys]

    if verbose:
        print(f"  GEN SOL config (direct override): gen_sol.yaml  (base + {len(override_keys)} overrides)")

    state.gen_jobs = []
    for i, variant_key in enumerate(all_variants):
        # Extract concurrency from the resolved config
        conc = cfg.gen_sweep[i].concurrency if i < len(cfg.gen_sweep) else 0

        state.gen_jobs.append({
            "config_path": gen_path,
            "selector": variant_key,
            "status": "pending",
            "gen_item_index": i,
            "concurrency": conc if isinstance(conc, int) else conc[0],
        })
        if verbose:
            print(f"    → {variant_key} (c={conc})")


def _generate_gen_sol_override_mode(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    configs_dir: Path,
    verbose: bool,
) -> None:
    """Generate GEN SOL configs using gen_sol_base (flat) + gen_sweep groups.

    Each named group in gen_sweep produces one YAML file with base: + override_c*:.
    Each concurrency level becomes a separate gen_job record with a `selector`
    field that tells slurm_helpers which override variant to submit.
    """
    from schema import GenSweepGroup, GenSweepItem as _GSI

    raw_groups = cfg.raw_gen_sweep_groups
    if raw_groups is None:
        raise ValueError("gen_sol_base requires gen_sweep to be a dict of named groups")

    state.gen_jobs = []
    gen_item_offset = 0  # Track position in the flattened cfg.gen_sweep list

    for group_name, group_data in raw_groups.items():
        # Expand group to get the items
        if isinstance(group_data, dict) and "parameters" in group_data:
            group = GenSweepGroup(**group_data)
            group_items = group.expand()
        else:
            group_items = [_GSI(**group_data)]

        # Generate the base/override YAML for this group
        mtp_suffix = f"_mtp{group_items[0].mtp_num}" if group_items[0].mtp_num > 0 else ""
        fname = f"gen_sol_{group_name}{mtp_suffix}.yaml"
        gen_path = str(configs_dir / fname)
        override_config = generate_gen_sol_override_config(
            cfg, group_name, group_items, output_path=gen_path,
        )

        if verbose:
            override_keys = [k for k in override_config if k.startswith("override_")]
            print(f"  GEN SOL config (override): {fname}  (base + {len(override_keys)} overrides)")

        # Create one gen_job per concurrency level.
        # Base concurrency (first item) uses selector="base".
        # Other concurrencies use selector="override_c{N}".
        all_concurrencies: list[int] = []
        for item in group_items:
            conc_list = item.concurrency if isinstance(item.concurrency, list) else [item.concurrency]
            for c in conc_list:
                if c not in all_concurrencies:
                    all_concurrencies.append(c)

        first_conc = all_concurrencies[0]
        for conc in sorted(all_concurrencies):
            if conc == first_conc:
                selector = "base"
            else:
                selector = f"override_c{conc}"

            # Find the gen_item_index in the flattened list for this concurrency
            gen_item_index = gen_item_offset
            for j, item in enumerate(group_items):
                item_concs = item.concurrency if isinstance(item.concurrency, list) else [item.concurrency]
                if conc in item_concs:
                    gen_item_index = gen_item_offset + j
                    break

            state.gen_jobs.append({
                "config_path": gen_path,
                "selector": selector,
                "status": "pending",
                "gen_item_index": gen_item_index,
                "concurrency": conc,
            })
            if verbose:
                print(f"    → c{conc} (selector={selector})")

        gen_item_offset += len(group_items)


def phase2_ctx_sol(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 2: Submit CTX-only SOL, poll, process results."""
    ctx_parser, _ = _get_parsers(cfg)

    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: Running CTX-only SOL benchmark")
        print("=" * 60)

    ctx_job = state.ctx_job

    if ctx_job.get("status") == "completed" and state.ctx_result:
        if verbose:
            print("  CTX SOL already completed, skipping.")
        return

    if dry_run:
        if verbose:
            print(f"  [DRY-RUN] Would submit: {ctx_job['config_path']}")
        return

    # Submit, poll, and retry on failure
    _submit_and_poll(
        job_dict=ctx_job,
        config_path=ctx_job["config_path"],
        poll_interval=cfg.settings.poll_interval,
        max_retries=cfg.settings.max_retries,
        state=state,
        verbose=verbose,
        max_poll_time=cfg.settings.max_poll_time,
    )

    # Process results
    if verbose:
        print("  Processing CTX results...")
    logs_dir = Path(ctx_job["output_dir"]) / "logs"
    log_file = ctx_parser.find_log(logs_dir)
    if log_file is None:
        raise RuntimeError(f"No prefill log found in {logs_dir}")

    data = ctx_parser.parse(log_file, verbose=False)
    if not data:
        raise RuntimeError(f"No data parsed from {log_file}")

    # Pass the effective max_batch_size from the generated config so the
    # num_ctx_requests threshold matches the actual deployed batch size
    # (GPU-agnostic -- works for H200, GB200, etc.).
    ctx_mbs = ctx_job.get("max_batch_size")
    ctx_result = ctx_parser.process(
        data, isl=cfg.workload.isl, verbose=False, max_batch_size=ctx_mbs,
    )
    if "error" in ctx_result:
        raise RuntimeError(f"CTX processing error: {ctx_result['error']}")

    state.ctx_result = ctx_result
    if verbose:
        print(f"  CTX request rate: {ctx_result['request_rate_req_per_s']:.4f} req/s")
        print(f"  CTX throughput:   {ctx_result['ctx_throughput_tokens_per_s']:,.0f} tok/s")
        print(f"  Threshold used:   num_ctx_requests >= {ctx_result.get('threshold_used', '?')}")

    state.phase = "gen"
    state.save()


def phase3_gen_sol(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 3: Submit GEN-only SOL jobs, poll, process results."""
    _, gen_parser = _get_parsers(cfg)

    if verbose:
        parallel = cfg.settings.parallel_submissions
        mode = "parallel" if parallel else "serialised"
        print(f"\n{'=' * 60}")
        print(f"PHASE 3: Running GEN-only SOL benchmarks ({mode})")
        print(f"{'=' * 60}")

    if dry_run:
        for gj in state.gen_jobs:
            submit_path = _resolve_submit_path(gj)
            if verbose:
                print(f"  [DRY-RUN] Would submit: {submit_path}")
        return

    # Submit all (or one at a time)
    pending = [gj for gj in state.gen_jobs if gj.get("status") not in ("completed",)]
    if not pending:
        if verbose:
            print("  All GEN SOL jobs already completed.")
        state.phase = "rate_match"
        state.save()
        return

    max_retries = cfg.settings.max_retries

    if cfg.settings.parallel_submissions:
        _submit_poll_parallel(
            job_dicts=pending,
            poll_interval=cfg.settings.poll_interval,
            max_retries=max_retries,
            max_poll_time=cfg.settings.max_poll_time,
            state=state,
            verbose=verbose,
            label="GEN",
        )
    else:
        # Serial: submit, poll, retry per job
        for gj in pending:
            try:
                _submit_and_poll(
                    job_dict=gj,
                    config_path=_resolve_submit_path(gj),
                    poll_interval=cfg.settings.poll_interval,
                    max_retries=max_retries,
                    state=state,
                    verbose=verbose,
                    max_poll_time=cfg.settings.max_poll_time,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ERROR: GEN job exhausted retries: {exc}")
                # Continue to other GEN jobs -- the failed one stays "failed"

    # Process results for all completed GEN jobs
    # METHODOLOGY: Always use decode worker logs (prev_device_step_time), NOT
    # sa-bench client-side JSONs.  Decode logs give the true device step time,
    # which combined with hardcoded MTP accept rates yields SOL throughput.
    # sa-bench JSONs capture E2E client metrics (including network overhead)
    # and are NOT used for GEN SOL.
    #
    # Multi-concurrency handling: when sa-bench runs concurrencies "8x32x64"
    # sequentially, the decode log is continuous.  The exact-match filter on
    # num_scheduled_requests (Step 4) naturally segments iterations by
    # concurrency.
    state.gen_results = []
    isl = cfg.workload.isl
    mtp_overrides = getattr(cfg.workload, 'mtp_accept_rates', None)
    for idx, gj in enumerate(state.gen_jobs):
        if gj["status"] != "completed":
            if verbose:
                print(f"  WARNING: GEN job {gj.get('job_id', '?')} not completed, skipping.")
            continue

        gen_item = cfg.gen_sweep[gj.get("gen_item_index", idx)]
        # If the job was split per-concurrency, use the stored single value.
        # Otherwise, use the sweep item's full concurrency list.
        if "concurrency" in gj:
            conc_list = [gj["concurrency"]]
        else:
            conc_list = gen_item.concurrency if isinstance(gen_item.concurrency, list) else [gen_item.concurrency]
        num_gpus = cfg.resources.gen_gpus_per_instance

        # Always parse the decode worker log (not sa-bench JSONs)
        logs_dir = Path(gj.get("output_dir", "")) / "logs"
        log_file = gen_parser.find_log(logs_dir)
        if log_file is None:
            if verbose:
                print(f"  WARNING: No decode log for job {gj.get('job_id', '?')}")
            continue

        data = gen_parser.parse(log_file, verbose=False)
        if not data:
            if verbose:
                print(f"  WARNING: No iteration data from {log_file}")
            continue

        if verbose:
            print(f"  Parsed {len(data)} iterations from {log_file.name}")

        # Determine ep_rank for DEP mode
        ep_rank = gen_item.tp_size  # ep_rank = rank_num = tp

        # Process each concurrency from the decode log.
        # When split per-concurrency, there's exactly one.
        # When multi-concurrency, the exact-match filter on
        # num_scheduled_requests isolates each level.
        gj["results"] = []
        gen_process_kwargs: dict = {}
        if cfg.engine_type == "vllm":
            gen_process_kwargs["aggregated_log"] = cfg.backend.vllm_aggregated_log

        for conc in conc_list:
            result = gen_parser.process(
                data,
                concurrency=conc,
                mode=gen_item.mode,
                tp=gen_item.tp_size,
                ep_rank=ep_rank,
                mtp=gen_item.mtp_num,
                isl=isl,
                num_gpus=num_gpus,
                verbose=verbose,
                mtp_accept_rate_overrides=mtp_overrides,
                **gen_process_kwargs,
            )
            if "error" in result:
                if verbose:
                    print(f"  WARNING: GEN c{conc} processing error: {result['error']}")
                continue

            # Attach config metadata to result
            result["batch_size"] = gen_item.batch_size
            result["max_num_tokens"] = gen_item.max_num_tokens
            result["gpu_memory_fraction"] = gen_item.gpu_memory_fraction
            result["eplb_num_slots"] = gen_item.eplb_num_slots
            result["tp_size"] = gen_item.tp_size
            # Carry per-item overrides so E2E configs use the same backend settings
            if gen_item.decode_overrides:
                result["decode_overrides"] = gen_item.decode_overrides
            if gen_item.prefill_overrides:
                result["prefill_overrides"] = gen_item.prefill_overrides

            gj["results"].append(result)
            state.gen_results.append(result)
            if verbose:
                print(
                    f"  {gen_item.mode.upper()} c{conc} mtp{gen_item.mtp_num}: "
                    f"step={result['avg_step_time_ms']:.2f}ms "
                    f"TPOT={result['tpot_ms']:.2f}ms "
                    f"tput/gpu={result.get('throughput_per_gpu', 0):.1f} "
                    f"accept_rate={result['mtp_accept_rate']}"
                )

    state.save()

    if verbose:
        print(f"\n  Processed {len(state.gen_results)}/{len(state.gen_jobs)} GEN results")

    state.phase = "rate_match"
    state.save()


def phase4_rate_matching(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 4: Compute rate-matching metrics for each GEN result."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("PHASE 4: Computing rate-matching allocations")
        print(f"{'=' * 60}")

    osl = cfg.workload.osl
    isl = cfg.workload.isl
    random_ratio = getattr(cfg.workload, 'random_ratio', 1.0)
    state.rate_matching_results = []

    for gen_result in state.gen_results:
        rm = compute_rate_matching(
            ctx_result=state.ctx_result,
            gen_result=gen_result,
            osl=osl,
            random_ratio=random_ratio,
            gpus_per_ctx_instance=cfg.resources.ctx_gpus_per_instance,
            gpus_per_gen_instance=cfg.resources.gen_gpus_per_instance,
            max_total_gpus=getattr(cfg.resources, 'max_total_gpus', 64),
            isl=isl,
        )

        # Carry forward extra fields
        rm["max_num_tokens"] = gen_result.get("max_num_tokens")
        rm["gpu_memory_fraction"] = gen_result.get("gpu_memory_fraction")
        rm["eplb_num_slots"] = gen_result.get("eplb_num_slots", 0)
        # Per-item overrides flow through to E2E config generation
        if gen_result.get("decode_overrides"):
            rm["decode_overrides"] = gen_result["decode_overrides"]
        if gen_result.get("prefill_overrides"):
            rm["prefill_overrides"] = gen_result["prefill_overrides"]

        state.rate_matching_results.append(rm)

        if verbose:
            print(
                f"  {rm['config_name']}: "
                f"ratio={rm['ratio_str']} "
                f"gen_req_rate={rm['gen_req_rate']:.2f} req/s "
                f"tput/gpu={rm['output_tput_per_gpu']:.1f} "
                f"ctx:gen={rm['ctx_instances']}:{rm['gen_instances']}"
            )

    state.phase = "pareto"
    state.save()


def phase5_pareto(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 5: Extract Pareto frontier."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("PHASE 5: Extracting Pareto frontier")
        print(f"{'=' * 60}")

    frontier = extract_pareto_frontier(state.rate_matching_results)
    state.pareto_frontier = frontier

    if verbose:
        print(f"  {len(frontier)} Pareto-optimal points from {len(state.rate_matching_results)} configs:")
        for p in frontier:
            print(
                f"    Rank {p['pareto_rank']}: c{p['concurrency']} {p['mode'].upper()} "
                f"mtp{p.get('mtp_num', 0)} "
                f"ratio={p['ratio_str']} "
                f"inter={p['interactivity']:.1f} "
                f"tput/gpu={p['output_tput_per_gpu']:.1f}"
            )

    # Export results
    _export_results(cfg, state, verbose=verbose)

    state.phase = "e2e" if cfg.settings.run_e2e_validation else "complete"
    state.save()


def phase6_e2e_validation(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Phase 6: Generate E2E configs, submit, poll, process, compare SOL vs E2E."""
    if verbose:
        parallel = cfg.settings.parallel_submissions
        mode_str = "parallel" if parallel else "serialised"
        mults = cfg.settings.e2e_validation.concurrency_multipliers
        print(f"\n{'=' * 60}")
        print(f"PHASE 6: E2E validation ({mode_str})")
        print(f"  Concurrency multipliers: {mults}")
        print(f"  Pareto points: {len(state.pareto_frontier)}")
        print(f"  Total E2E jobs: {len(state.pareto_frontier) * len(mults)}")
        print(f"{'=' * 60}")

    # Generate E2E configs if not already done
    if not state.e2e_configs:
        e2e_dir = Path(state.output_dir) / "e2e_pareto_configs"
        state.e2e_configs = generate_e2e_configs_from_pareto(
            cfg, state.pareto_frontier, str(e2e_dir),
        )
        state.save()
        if verbose:
            for ec in state.e2e_configs:
                print(f"  Config: {Path(ec['config_path']).name} "
                      f"(rank {ec['pareto_rank']}, {ec['multiplier']}x, "
                      f"sys_conc={ec['system_concurrency']})")

    if dry_run:
        if verbose:
            print(f"\n  [DRY-RUN] Would submit {len(state.e2e_configs)} E2E jobs")
        return

    # Submit E2E jobs
    if not state.e2e_jobs:
        state.e2e_jobs = [
            {"config_path": ec["config_path"], "status": "pending",
             "pareto_rank": ec["pareto_rank"], "multiplier": ec["multiplier"],
             "per_worker_concurrency": ec["per_worker_concurrency"],
             "system_concurrency": ec["system_concurrency"],
             "config_name": ec["config_name"]}
            for ec in state.e2e_configs
        ]
        state.save()

    pending = [ej for ej in state.e2e_jobs if ej.get("status") not in ("completed",)]
    if not pending and state.sol_vs_e2e:
        if verbose:
            print("  All E2E jobs already completed and processed.")
        return

    max_retries = cfg.settings.max_retries

    if cfg.settings.parallel_submissions:
        _submit_poll_parallel(
            job_dicts=pending,
            poll_interval=cfg.settings.poll_interval,
            max_retries=max_retries,
            max_poll_time=cfg.settings.max_poll_time,
            state=state,
            verbose=verbose,
            label="E2E",
        )
    else:
        # Serial: submit, poll, retry per job
        for ej in pending:
            try:
                _submit_and_poll(
                    job_dict=ej,
                    config_path=ej["config_path"],
                    poll_interval=cfg.settings.poll_interval,
                    max_retries=max_retries,
                    state=state,
                    verbose=verbose,
                    max_poll_time=cfg.settings.max_poll_time,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ERROR: E2E job exhausted retries: {exc}")
                # Continue to other E2E jobs

    # Process E2E results
    _process_e2e_results(cfg, state, verbose=verbose)

    state.phase = "complete"
    state.save()


def _process_e2e_results(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Process E2E job results and compare against SOL predictions."""
    tolerances = {
        "tpot_tolerance_pct": cfg.settings.e2e_validation.tpot_tolerance_pct,
        "throughput_tolerance_pct": cfg.settings.e2e_validation.throughput_tolerance_pct,
    }
    ttft_constraint = cfg.settings.e2e_validation.ttft_constraint_ms

    # Build pareto map for quick lookup
    pareto_map = {p["pareto_rank"]: p for p in state.pareto_frontier}

    state.e2e_results = []
    state.sol_vs_e2e = []

    for ej in state.e2e_jobs:
        if ej.get("status") != "completed":
            continue

        if not ej.get("result"):
            # Load sa-bench result from output dir
            output_dir = ej.get("output_dir", get_job_output_dir(ej.get("job_id", ""), srtctl_root=state.srtctl_root))
            sa_result = _load_sa_bench_result(output_dir)
            if sa_result is None:
                if verbose:
                    print(f"  WARNING: No sa-bench result for E2E job {ej.get('job_id', '?')}")
                continue
            ej["result"] = sa_result

        sa_result = ej["result"]
        pareto_rank = ej["pareto_rank"]
        multiplier = ej["multiplier"]
        pp = pareto_map.get(pareto_rank)
        if pp is None:
            continue

        gen_instances = pp["gen_instances"]
        ctx_instances = pp["ctx_instances"]
        total_gpus = pp["total_gpus"]
        per_worker_conc = pp["concurrency"]
        system_conc = ej["system_concurrency"]

        if verbose:
            print(f"\n  Processing E2E (rank {pareto_rank}, {multiplier}x)...")
            print(f"    System concurrency: {system_conc} "
                  f"(per-worker {per_worker_conc} x {gen_instances} workers x {multiplier})")

        comparison = compare_sol_vs_e2e(
            sol_prediction=pp,
            e2e_result=sa_result,
            total_gpus=total_gpus,
            gen_instances=gen_instances,
            per_worker_conc=per_worker_conc,
            tolerances=tolerances,
        )

        # TTFT
        e2e_ttft_ms = sa_result.get("median_ttft_ms", 0)
        if e2e_ttft_ms == 0:
            ttft_percentiles = sa_result.get("ttft_percentiles", {})
            e2e_ttft_ms = ttft_percentiles.get("p50", 0)
        ttft_pass = True
        if ttft_constraint is not None and e2e_ttft_ms > 0:
            ttft_pass = bool(e2e_ttft_ms <= ttft_constraint)

        entry = {
            "pareto_rank": pareto_rank,
            "multiplier": multiplier,
            "per_worker_concurrency": per_worker_conc,
            "system_concurrency": system_conc,
            "e2e_ttft_ms": e2e_ttft_ms,
            "ttft_constraint_ms": ttft_constraint,
            "ttft_pass": ttft_pass,
            "config_name": ej.get("config_name", ""),
            **comparison,
        }
        state.sol_vs_e2e.append(entry)
        state.e2e_results.append({"pareto_rank": pareto_rank, "multiplier": multiplier, **sa_result})

        if verbose:
            print(f"    TPOT diff: {comparison['diff_pct']['tpot']:+.1f}%")
            print(f"    Throughput diff: {comparison['diff_pct']['output_tput_per_gpu']:+.1f}%")
            if ttft_constraint is not None:
                status = "PASS" if ttft_pass else "FAIL"
                print(f"    TTFT: {e2e_ttft_ms:.0f}ms ({status}, constraint: {ttft_constraint}ms)")

    state.save()




# ---------------------------------------------------------------------------
# Phase 7: Dashboard / summary
# ---------------------------------------------------------------------------

def phase7_summary(
    cfg: RateMatchingSweepConfig,
    state: SweepState,
    verbose: bool = True,
) -> None:
    """Phase 7: Generate charts, print final summary."""
    if verbose:
        print(f"\n{'=' * 60}")
        print("SWEEP COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {state.output_dir}")
        print(f"  - Configs: {state.output_dir}/configs/")
        print(f"  - Results: {state.output_dir}/results/")
        if state.e2e_configs:
            print(f"  - E2E Pareto: {state.output_dir}/e2e_pareto_configs/")
        print(f"  - State: {state.output_dir}/sweep_state.json")

    # Generate charts
    try:
        from dashboard_export import generate_charts, export_rate_matching_results

        dashboard_dir = Path(state.output_dir) / "dashboard"
        export_state = {
            "sweep_name": state.sweep_name,
            "rate_matching_results": state.rate_matching_results,
            "pareto_frontier": state.pareto_frontier,
            "sol_vs_e2e": state.sol_vs_e2e,
            "ctx_result": state.ctx_result,
        }
        export_rate_matching_results(export_state, str(dashboard_dir))
        charts = generate_charts(export_state, str(dashboard_dir))
        if charts and verbose:
            print(f"\nInteractive charts (open in browser):")
            for chart_path in charts:
                print(f"  file://{chart_path}")
    except Exception as e:
        if verbose:
            print(f"\n  (Dashboard export skipped: {e})")

    # TTFT summary
    if state.sol_vs_e2e and verbose:
        ttft_constraint = cfg.settings.e2e_validation.ttft_constraint_ms
        if ttft_constraint:
            ttft_results = [e for e in state.sol_vs_e2e if e.get("e2e_ttft_ms", 0) > 0]
            ttft_pass = sum(1 for e in ttft_results if e.get("ttft_pass", False))
            print(f"\nTTFT CONSTRAINT CHECK (soft limit: {ttft_constraint}ms)")
            if ttft_pass == len(ttft_results):
                print(f"  All {len(ttft_results)} E2E points meet the TTFT constraint.")
            else:
                print(f"  {ttft_pass}/{len(ttft_results)} E2E points meet the TTFT constraint.")


# ---------------------------------------------------------------------------
# Reprocess (re-derive everything from existing logs, no SLURM submission)
# ---------------------------------------------------------------------------

def add_e2e_jobs(
    output_dir: str,
    multipliers: list[float],
    config_path: str | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> SweepState:
    """Add new E2E validation jobs to a completed sweep.

    Generates and submits E2E configs for new ``(pareto_rank, multiplier)``
    pairs that don't already exist in the sweep state.  Existing results are
    preserved — this is a purely additive operation.

    Typical usage::

        srtctl-rate-match add-e2e -o ./sweeps/my_sweep --multipliers 0.95
        srtctl-rate-match add-e2e -o ./sweeps/my_sweep --multipliers 0.90 0.95 1.10

    Args:
        output_dir: Sweep output directory containing sweep_state.json.
        multipliers: New concurrency multipliers to add.
        config_path: Optional path to an updated sweep YAML config.  If None,
            the original config from the sweep state is used.
        dry_run: If True, show what would be added without submitting.
        verbose: Print progress.

    Returns:
        Updated SweepState.

    Raises:
        FileNotFoundError: If the sweep state or config is missing.
        RuntimeError: If the sweep has not reached the Pareto phase yet.
    """
    global _active_state  # noqa: PLW0603

    # --- Load state ---
    state_path = Path(output_dir) / "sweep_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No sweep state found at {state_path}")
    state = SweepState.load(str(state_path))

    # Must have a Pareto frontier to generate E2E configs against
    if not state.pareto_frontier:
        raise RuntimeError(
            "Cannot add E2E jobs: sweep has no Pareto frontier yet.  "
            "Run the sweep through at least Phase 5 (Pareto extraction) first."
        )

    # --- Load config ---
    cfg_path = config_path or state.sweep_config_path
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Sweep config not found: {cfg_path}")
    cfg = load_sweep_config(cfg_path)

    # --- Back up state before mutation ---
    backup_path = state.save_backup()
    if verbose:
        print(f"State backed up to: {backup_path}")

    # --- Install signal handlers ---
    _active_state = state
    if not dry_run:
        _install_signal_handlers()

    # --- Determine which (pareto_rank, multiplier) pairs already exist ---
    existing_pairs: set[tuple[int, float]] = set()
    for ec in state.e2e_configs:
        existing_pairs.add((ec["pareto_rank"], ec["multiplier"]))
    for ej in state.e2e_jobs:
        existing_pairs.add((ej["pareto_rank"], ej["multiplier"]))

    # Compute new pairs
    new_multipliers = []
    for m in multipliers:
        # Check if ALL Pareto points already have this multiplier
        all_exist = all(
            (pp.get("pareto_rank", 0), m) in existing_pairs
            for pp in state.pareto_frontier
        )
        if not all_exist:
            new_multipliers.append(m)

    if not new_multipliers:
        if verbose:
            print("All requested multipliers already exist in the sweep. Nothing to add.")
        return state

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ADD E2E: {state.sweep_name}")
        print(f"{'=' * 60}")
        print(f"  Output dir:           {output_dir}")
        print(f"  Pareto points:        {len(state.pareto_frontier)}")
        print(f"  Existing multipliers: {sorted({m for _, m in existing_pairs})}")
        print(f"  New multipliers:      {new_multipliers}")
        n_new = sum(
            1 for pp in state.pareto_frontier for m in new_multipliers
            if (pp.get("pareto_rank", 0), m) not in existing_pairs
        )
        print(f"  New E2E jobs:         {n_new}")

    # --- Generate new E2E configs ---
    e2e_dir = Path(state.output_dir) / "e2e_pareto_configs"
    new_configs = []
    for pp in state.pareto_frontier:
        for m in new_multipliers:
            pair = (pp.get("pareto_rank", 0), m)
            if pair in existing_pairs:
                continue
            recipe_name = get_recipe_filename(
                pp["concurrency"], pp["ctx_instances"], pp["gen_instances"],
                pp["mode"], pp.get("tp_size", cfg.resources.gen_gpus_per_instance),
                pp["batch_size"], pp.get("mtp_num", 0),
                pp.get("eplb_num_slots", 0), multiplier=m,
            )
            config_path_e2e = str(e2e_dir / f"{recipe_name}.yaml")
            generate_e2e_config(
                cfg, pp, concurrency_multiplier=m, output_path=config_path_e2e,
            )
            gen_instances = pp["gen_instances"]
            pw_conc = pp["concurrency"]
            sys_conc = int(pw_conc * gen_instances * m)

            ec = {
                "config_path": config_path_e2e,
                "pareto_rank": pp.get("pareto_rank", 0),
                "multiplier": m,
                "per_worker_concurrency": pw_conc,
                "system_concurrency": sys_conc,
                "config_name": recipe_name,
            }
            new_configs.append(ec)
            existing_pairs.add(pair)  # prevent duplicates within this call
            if verbose:
                print(f"  Config: {recipe_name}.yaml "
                      f"(rank {ec['pareto_rank']}, {m}x, sys_conc={sys_conc})")

    # Append to state
    state.e2e_configs.extend(new_configs)

    new_jobs = [
        {"config_path": ec["config_path"], "status": "pending",
         "pareto_rank": ec["pareto_rank"], "multiplier": ec["multiplier"],
         "per_worker_concurrency": ec["per_worker_concurrency"],
         "system_concurrency": ec["system_concurrency"],
         "config_name": ec["config_name"]}
        for ec in new_configs
    ]
    state.e2e_jobs.extend(new_jobs)
    state.save()

    if dry_run:
        if verbose:
            print(f"\n  [DRY-RUN] Would submit {len(new_jobs)} new E2E jobs")
        return state

    # --- Submit new jobs ---
    if verbose:
        print(f"\nSubmitting {len(new_jobs)} new E2E jobs...")

    if cfg.settings.parallel_submissions:
        _submit_poll_parallel(
            job_dicts=new_jobs,
            poll_interval=cfg.settings.poll_interval,
            max_retries=cfg.settings.max_retries,
            max_poll_time=cfg.settings.max_poll_time,
            state=state,
            verbose=verbose,
            label="E2E",
        )
    else:
        for ej in new_jobs:
            try:
                _submit_and_poll(
                    job_dict=ej,
                    config_path=ej["config_path"],
                    poll_interval=cfg.settings.poll_interval,
                    max_retries=cfg.settings.max_retries,
                    state=state,
                    verbose=verbose,
                    max_poll_time=cfg.settings.max_poll_time,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ERROR: E2E job exhausted retries: {exc}")

    # --- Re-process ALL E2E results (old + new) and regenerate dashboard ---
    _process_e2e_results(cfg, state, verbose=verbose)

    state.phase = "complete"
    state.save()

    phase7_summary(cfg, state, verbose=verbose)

    if verbose:
        print(f"\nAdd-E2E complete. {len(new_jobs)} new jobs added.")
        print(f"Results in: {state.output_dir}")

    return state


def reprocess_sweep(
    output_dir: str,
    config_path: str | None = None,
    skip_e2e: bool = False,
    verbose: bool = True,
) -> SweepState:
    """Re-process an existing sweep from log files without resubmitting jobs.

    This is useful when:
      - You've changed parameters in the sweep YAML (e.g., mtp_accept_rates,
        random_ratio) and want to re-derive metrics.
      - You've fixed a bug in processing logic and need to recompute.
      - You want to regenerate dashboards/exports.

    The function re-reads raw TRT-LLM logs from each job's output_dir and
    re-runs phases 2 (CTX processing) -> 3 (GEN processing) -> 4 (rate-match)
    -> 5 (Pareto) -> 6 (E2E comparison, if applicable) -> 7 (dashboard).

    No SLURM jobs are submitted.

    Args:
        output_dir: Sweep output directory containing sweep_state.json.
        config_path: Optional path to an updated sweep YAML config. If None,
            the original config from the sweep state is used.
        skip_e2e: If True, skip E2E re-processing even if E2E results exist.
        verbose: Print progress.

    Returns:
        Updated SweepState.
    """
    # Load state
    state_path = Path(output_dir) / "sweep_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No sweep state found at {state_path}")
    state = SweepState.load(str(state_path))

    # Load config (allow override)
    cfg_path = config_path or state.sweep_config_path
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Sweep config not found: {cfg_path}")
    cfg = load_sweep_config(cfg_path)
    ctx_parser, gen_parser = _get_parsers(cfg)

    if verbose:
        print(f"{'=' * 60}")
        print(f"REPROCESS: {state.sweep_name}")
        print(f"{'=' * 60}")
        print(f"  Output dir: {output_dir}")
        print(f"  Config:     {cfg_path}")
        if config_path:
            print(f"  (using updated config)")

    # Reconcile stale jobs (orchestrator may have been interrupted)
    n = _reconcile_stale_jobs(
        state, ctx_parser=ctx_parser, gen_parser=gen_parser, verbose=verbose,
    )
    if n and verbose:
        print(f"  Total reconciled: {n} job(s)")

    # --- Re-run Phase 2: CTX processing (no submission) ---
    if verbose:
        print(f"\n{'=' * 60}")
        print("REPROCESS Phase 2: Re-processing CTX results from logs")
        print(f"{'=' * 60}")

    ctx_job = state.ctx_job
    if ctx_job.get("status") == "completed" and ctx_job.get("output_dir"):
        logs_dir = Path(ctx_job["output_dir"]) / "logs"
        log_file = ctx_parser.find_log(logs_dir)
        if log_file is None:
            raise RuntimeError(f"No prefill log found in {logs_dir}")

        data = ctx_parser.parse(log_file, verbose=False)
        if not data:
            raise RuntimeError(f"No data parsed from {log_file}")

        ctx_mbs = ctx_job.get("max_batch_size")
        ctx_result = ctx_parser.process(
            data, isl=cfg.workload.isl, verbose=False, max_batch_size=ctx_mbs,
        )
        if "error" in ctx_result:
            raise RuntimeError(f"CTX processing error: {ctx_result['error']}")

        state.ctx_result = ctx_result
        if verbose:
            print(f"  CTX request rate: {ctx_result['request_rate_req_per_s']:.4f} req/s")
            print(f"  CTX throughput:   {ctx_result['ctx_throughput_tokens_per_s']:,.0f} tok/s")
            print(f"  Threshold used:   num_ctx_requests >= {ctx_result.get('threshold_used', '?')}")
    else:
        if verbose:
            print("  WARNING: CTX job not completed, keeping existing ctx_result")

    # --- Re-run Phase 3: GEN processing (no submission) ---
    if verbose:
        print(f"\n{'=' * 60}")
        print("REPROCESS Phase 3: Re-processing GEN results from logs")
        print(f"{'=' * 60}")

    isl = cfg.workload.isl
    mtp_overrides = getattr(cfg.workload, 'mtp_accept_rates', None)
    num_gpus = cfg.resources.gen_gpus_per_instance
    state.gen_results = []

    for idx, gj in enumerate(state.gen_jobs):
        if gj["status"] != "completed":
            if verbose:
                print(f"  WARNING: GEN job {gj.get('job_id', '?')} not completed, skipping.")
            continue

        # Determine concurrency list: prefer job-level overrides, then config
        if "concurrency" in gj:
            conc_list = [gj["concurrency"]]
        elif "concurrencies" in gj:
            conc_list = gj["concurrencies"]
        else:
            gi_idx = gj.get("gen_item_index", idx)
            if gi_idx < len(cfg.gen_sweep):
                gi = cfg.gen_sweep[gi_idx]
                conc_list = gi.concurrency if isinstance(gi.concurrency, list) else [gi.concurrency]
            else:
                if verbose:
                    print(f"  WARNING: gen_item_index {gi_idx} out of range, skipping GEN job {idx}")
                continue

        # Resolve mode/mtp/batch/tp from job dict first, then fall back to config
        gi_idx = gj.get("gen_item_index", idx)
        if gi_idx < len(cfg.gen_sweep):
            gen_item = cfg.gen_sweep[gi_idx]
            mode = gj.get("mode", gen_item.mode)
            tp_size = gj.get("tp_size", gen_item.tp_size)
            mtp_num = gj.get("mtp_num", gen_item.mtp_num)
            batch_size = gj.get("batch_size", gen_item.batch_size)
            max_num_tokens = gj.get("max_num_tokens", gen_item.max_num_tokens)
            gpu_mem_frac = gj.get("gpu_memory_fraction", gen_item.gpu_memory_fraction)
            eplb_slots = gj.get("eplb_num_slots", gen_item.eplb_num_slots)
        else:
            # Config gen_sweep doesn't match (e.g., after refactor) - use job dict
            mode = gj.get("mode", "tep")
            tp_size = gj.get("tp_size", 8)
            mtp_num = gj.get("mtp_num", 0)
            batch_size = gj.get("batch_size", 128)
            max_num_tokens = gj.get("max_num_tokens", 512)
            gpu_mem_frac = gj.get("gpu_memory_fraction", 0.9)
            eplb_slots = gj.get("eplb_num_slots", 0)

        logs_dir = Path(gj.get("output_dir", "")) / "logs"
        log_file = gen_parser.find_log(logs_dir)
        if log_file is None:
            if verbose:
                print(f"  WARNING: No decode log in {logs_dir}, skipping GEN job {idx}")
            continue

        data = gen_parser.parse(log_file, verbose=False)
        if not data:
            if verbose:
                print(f"  WARNING: No data parsed from {log_file}")
            continue

        if verbose:
            print(f"  Parsed {len(data)} iterations from {log_file.name}")

        ep_rank = tp_size  # ep_rank = rank_num = tp

        gen_process_kwargs_reparse: dict = {}
        if cfg.engine_type == "vllm":
            gen_process_kwargs_reparse["aggregated_log"] = cfg.backend.vllm_aggregated_log

        for conc in conc_list:
            result = gen_parser.process(
                data,
                concurrency=conc,
                mode=mode,
                tp=tp_size,
                ep_rank=ep_rank,
                mtp=mtp_num,
                isl=isl,
                num_gpus=num_gpus,
                verbose=False,
                mtp_accept_rate_overrides=mtp_overrides,
                **gen_process_kwargs_reparse,
            )
            if "error" in result:
                if verbose:
                    print(f"  WARNING: GEN c{conc} processing error: {result['error']}")
                continue

            result["batch_size"] = batch_size
            result["max_num_tokens"] = max_num_tokens
            result["gpu_memory_fraction"] = gpu_mem_frac
            result["eplb_num_slots"] = eplb_slots
            result["tp_size"] = tp_size

            state.gen_results.append(result)
            if verbose:
                print(
                    f"  {mode.upper()} c{conc} mtp{mtp_num}: "
                    f"step={result['avg_step_time_ms']:.2f}ms "
                    f"TPOT={result['tpot_ms']:.2f}ms "
                    f"tput/gpu={result.get('throughput_per_gpu', 0):.1f} "
                    f"accept_rate={result['mtp_accept_rate']}"
                )

    state.save()
    if verbose:
        print(f"\n  Re-processed {len(state.gen_results)} GEN results")

    # --- Re-run Phase 4: Rate matching ---
    phase4_rate_matching(cfg, state, verbose=verbose)

    # --- Re-run Phase 5: Pareto ---
    phase5_pareto(cfg, state, verbose=verbose)

    # Override phase transition from phase5 (which sets "e2e" or "complete").
    # If reprocess crashes after this point, a subsequent `run_sweep --resume`
    # should NOT try to submit new E2E jobs.  Setting to "complete" is safe
    # because run_sweep's phase checks use `==` (not `in`), and "complete"
    # only triggers phase7_summary (always called anyway).
    state.phase = "complete"
    state.save()

    # --- Re-run Phase 6: E2E comparison (from existing results, no submission) ---
    if not skip_e2e and state.e2e_jobs:
        # Stale E2E jobs were already reconciled by _reconcile_stale_jobs above.
        completed_e2e = [ej for ej in state.e2e_jobs if ej.get("status") == "completed"]
        if completed_e2e:
            if verbose:
                print(f"\n{'=' * 60}")
                print("REPROCESS Phase 6: Re-processing E2E results")
                print(f"{'=' * 60}")
            _process_e2e_results(cfg, state, verbose=verbose)

    # --- Re-run Phase 7: Dashboard ---
    phase7_summary(cfg, state, verbose=verbose)

    state.phase = "complete"
    state.save()

    if verbose:
        print(f"\nReprocessing complete. State saved to: {state_path}")

    return state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(
    config_path: str,
    output_dir: str,
    resume: bool = False,
    skip_e2e: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> SweepState:
    """Run the full rate-matching sweep pipeline.

    Args:
        config_path: Path to sweep YAML config
        output_dir: Output directory for configs, results, state
        resume: If True, resume from saved state
        skip_e2e: If True, skip E2E validation phase
        dry_run: If True, generate configs only (no SLURM submission)
        verbose: Print progress

    Returns:
        Final SweepState
    """
    global _active_state  # noqa: PLW0603

    # Load config
    cfg = load_sweep_config(config_path)

    # Create or resume state
    state_path = Path(output_dir) / "sweep_state.json"
    if resume and state_path.exists():
        state = SweepState.load(str(state_path))
        if verbose:
            print(f"Resuming sweep from phase: {state.phase}")
    elif state_path.exists() and not dry_run:
        raise RuntimeError(
            f"Sweep state already exists at {state_path}.\n"
            "  To continue an interrupted sweep:  --resume\n"
            "  To add E2E multipliers:            srtctl-rate-match add-e2e\n"
            "  To re-derive metrics from logs:    srtctl-rate-match reprocess\n"
            "  To start fresh, choose a different -o directory."
        )
    else:
        state = SweepState()
        state.sweep_name = cfg.name
        state.sweep_config_path = str(Path(config_path).resolve())
        state.output_dir = str(Path(output_dir).resolve())
        state.created_at = datetime.now().isoformat()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Install signal handlers so state is saved on SIGHUP / SIGTERM / SIGINT
    _active_state = state
    if not dry_run:
        _install_signal_handlers()

    # On resume, reconcile jobs that completed while the orchestrator was down
    if resume and not dry_run:
        ctx_parser, gen_parser = _get_parsers(cfg)
        n = _reconcile_stale_jobs(
            state, ctx_parser=ctx_parser, gen_parser=gen_parser, verbose=verbose,
        )
        if n and verbose:
            print(f"  Reconciled {n} stale job(s) from previous run")

    # Run phases in order, skipping completed ones
    if state.phase in ("init", "ctx"):
        if state.phase == "init":
            phase1_generate_configs(cfg, state, verbose=verbose)

    if state.phase == "ctx":
        phase2_ctx_sol(cfg, state, dry_run=dry_run, verbose=verbose)

    if state.phase == "gen":
        phase3_gen_sol(cfg, state, dry_run=dry_run, verbose=verbose)

    if state.phase == "rate_match":
        phase4_rate_matching(cfg, state, verbose=verbose)

    if state.phase == "pareto":
        phase5_pareto(cfg, state, verbose=verbose)

    if state.phase in ("e2e",) and not skip_e2e and cfg.settings.run_e2e_validation:
        phase6_e2e_validation(cfg, state, dry_run=dry_run, verbose=verbose)

    phase7_summary(cfg, state, verbose=verbose)

    if not dry_run:
        state.phase = "complete"
        state.save()

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rate-matching sweep orchestrator")
    parser.add_argument("-c", "--config", required=True, help="Sweep config YAML")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E validation")
    parser.add_argument("--dry-run", action="store_true", help="Config generation only")
    args = parser.parse_args()

    run_sweep(
        config_path=args.config,
        output_dir=args.output,
        resume=args.resume,
        skip_e2e=args.skip_e2e,
        dry_run=args.dry_run,
    )
