#!/usr/bin/env python3
"""
SGLang GEN (decode) log parser for rate-matching.

Parses SGLang decode worker logs to extract GEN SOL metrics.
Implements the GENLogParser interface from parser_base.py.

Status: STUB — not yet implemented. Log format analysis and metric
extraction need to be written once SGLang SOL benchmarks are available.

Usage (pipeline):
    from process_gen_results_sglang import SglangGENLogParser
    parser = SglangGENLogParser()
    log_file = parser.find_log(logs_dir)
    data = parser.parse(log_file)
    result = parser.process(data, concurrency=32, mode='tep', tp=8)

Implementation guide:
    The parser must extract the same metrics as the TRT-LLM parser
    (see process_gen_results.py) but from SGLang's log format:

    1. find_log(): Locate the SGLang decode worker log file.
       - TRT-LLM uses: *_decode_w*.out, *_agg_w*.out
       - SGLang uses per-process srun launching — check the SGLang backend
         in src/srtctl/backends/ for log file naming.

    2. parse(): Extract per-iteration entries from the log.
       - SGLang's decode loop may log differently from TRT-LLM.
       - Each entry should capture:
         * iteration/step number
         * step time (device or wall-clock)
         * number of generation tokens
         * number of scheduled/active requests
         * rank (for DEP: expert parallel rank)
       - Return a list of dicts with engine-specific keys.

    3. process(): Filter and aggregate for ONE concurrency level.
       - Filter for pure decode iterations (no prefill tokens)
       - Apply warmup/cooldown trimming
       - Filter for exact concurrency match (TEP vs DEP logic)
       - Filter outliers
       - Compute throughput_per_user, tpot_ms, output_throughput
       - Return GENResult dict

    4. get_mtp_accept_rate(): Return MTP accept rate for SGLang.
       - SGLang has its own speculative decoding / MTP implementation.
       - Accept rates are model-dependent; use overrides from sweep YAML.

    SGLang-specific notes:
        - SGLang uses RadixAttention for KV cache management, which may
          affect decode scheduling and batch composition.
        - SGLang supports both TP (tensor parallel) and EP (expert parallel)
          via its router-based disaggregated architecture.
        - The srt-slurm SGLang backend already supports prefill/decode/
          aggregated modes with per-process srun launching.
        - SGLang's continuous batching may not have clean "iteration"
          boundaries — consider using time-windowed aggregation instead.

    Reference: process_gen_results.py for the complete TRT-LLM implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parser_base import GENLogParser, GENResult, register_gen_parser


# ---------------------------------------------------------------------------
# MTP accept rate defaults for SGLang
#
# SGLang's speculative decoding may yield different accept rates than TRT-LLM.
# These must be measured empirically for each model + ISL combination.
#
# Format: SGLANG_MTP_ACCEPT_RATES[isl][mtp_num] = tokens_per_step_per_user
# ---------------------------------------------------------------------------
SGLANG_MTP_ACCEPT_RATES: dict[int, dict[int, float]] = {
    # TODO: Populate with SGLang-specific MTP accept rates once measured.
    # Example structure (using TRT-LLM DSR1 values as placeholders):
    # 1024: {1: 1.8, 2: 2.28, 3: 2.56},
    # 8192: {1: 1.84, 2: 2.38, 3: 2.76},
}


@register_gen_parser("sglang")
class SglangGENLogParser(GENLogParser):
    """SGLang decode log parser.

    TODO: Implement once SGLang SOL benchmark log format is characterised.

    SGLang architecture notes:
        - RadixAttention for KV cache management
        - Per-process srun launching via srt-slurm SGLang backend
        - Router-based disaggregated serving (prefill/decode separation)
        - Continuous batching — may not have clean iteration boundaries
        - May expose metrics via Prometheus endpoint (:9090/metrics)
    """

    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the SGLang decode worker log file.

        TODO: Determine SGLang's log file naming convention in srt-slurm.
        The SGLang backend uses per-process srun launching, so log files
        may be named differently from TRT-LLM's MPI-style output.

        Candidate patterns (update based on actual SGLang backend output):
            - *_decode_*.log
            - *_sglang_*.out
            - sglang_worker_*.out
        """
        # TODO: Update glob patterns based on actual SGLang srt-slurm output
        patterns = [
            "*_decode_*.log",
            "*_sglang_decode_*.out",
            "*_decode_w*.out",
        ]
        for pattern in patterns:
            matches = list(logs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse per-iteration entries from an SGLang decode log.

        TODO: Implement SGLang log parsing. Each entry dict should contain
        at minimum:
            - iter: int (iteration/step number)
            - step_time_ms: float (device step time in milliseconds)
            - num_generation_tokens: int (decode tokens this iteration)
            - num_scheduled_requests: int (active decodes)
            - num_ctx_tokens: int (prefill tokens — should be 0 for pure decode)
            - rank: int (worker rank, 0-indexed)
            - global_rank: int (global rank across all workers)

        SGLang-specific considerations:
            - Continuous batching means iteration boundaries may be fluid.
              Consider using fixed time windows or token-count thresholds.
            - SGLang's RadixAttention may cause variable batch compositions.
            - Check if SGLang logs per-step stats to stdout or if you need
              to parse structured JSON output.

        These keys are consumed by process() below; adjust both together.
        """
        raise NotImplementedError(
            "SGLang GEN log parsing is not yet implemented. "
            "See process_gen_results.py (TRT-LLM) for reference implementation. "
            "Contributions welcome — implement parse() to extract per-iteration "
            "decode metrics from SGLang worker logs."
        )

    def process(
        self,
        data: list[dict],
        concurrency: int,
        mode: str,
        *,
        tp: int = 8,
        ep_rank: int | None = None,
        mtp: int = 0,
        isl: int = 1024,
        num_gpus: int = 8,
        verbose: bool = False,
        mtp_accept_rate_overrides: dict[int, float] | None = None,
    ) -> GENResult:
        """Process parsed SGLang decode entries into GEN SOL metrics.

        TODO: Implement filtering and aggregation for one concurrency level:
            1. Filter for pure decode iterations (no prefill tokens)
            2. Merge duplicate rows by (iter, rank) if applicable
            3. Remove warmup/cooldown iterations
            4. Filter by exact concurrency match (TEP vs DEP logic)
            5. Filter outliers
            6. Compute metrics:
                elapsed_time_avg = mean(step_time) in seconds
                mtp_rate = self.get_mtp_accept_rate(isl, mtp, overrides)
                throughput_per_user = (1 / elapsed_time_avg) * mtp_rate
                tpot_ms = (elapsed_time_avg / mtp_rate) * 1000
                output_throughput = throughput_per_user * concurrency

        Must return a GENResult dict on success (see parser_base.py):
            {
                "interactivity": throughput_per_user,
                "throughput_per_gpu": output_throughput / num_gpus,
                "output_throughput": output_throughput,
                "throughput_per_user": throughput_per_user,
                "avg_step_time_ms": elapsed_time_avg * 1000,
                "tpot_ms": tpot_ms,
                "concurrency": concurrency,
                "mode": mode,
                "mtp": mtp,
                "mtp_accept_rate": mtp_rate,
                "ep_rank": ep_rank or tp,
                "num_gpus": num_gpus,
                "num_iterations": len(filtered_data),
            }

        Or {"error": "description"} on failure.
        """
        raise NotImplementedError(
            "SGLang GEN result processing is not yet implemented. "
            "See process_gen_results.py (TRT-LLM) for reference implementation."
        )

    def get_mtp_accept_rate(
        self,
        isl: int,
        mtp_num: int,
        overrides: dict[int, float] | None = None,
    ) -> float:
        """Return the MTP accept rate for SGLang at given ISL and MTP level.

        For STP (mtp_num=0), always returns 1.0.
        For MTP/speculative decoding, checks overrides first, then the
        SGLANG_MTP_ACCEPT_RATES table.

        Note: SGLang's speculative decoding implementation may differ from
        TRT-LLM's MTP. Accept rates are model-dependent and must be measured.
        Until SGLang MTP benchmarks are run, use overrides from the sweep YAML.
        """
        if mtp_num == 0:
            return 1.0

        # Check overrides from sweep YAML first
        if overrides and mtp_num in overrides:
            return overrides[mtp_num]

        # Check built-in table
        if isl in SGLANG_MTP_ACCEPT_RATES and mtp_num in SGLANG_MTP_ACCEPT_RATES[isl]:
            return SGLANG_MTP_ACCEPT_RATES[isl][mtp_num]

        raise ValueError(
            f"No SGLang MTP accept rate for ISL={isl}, mtp_num={mtp_num}. "
            f"Provide mtp_accept_rates in the sweep YAML config. "
            f"Built-in table has ISLs: {sorted(SGLANG_MTP_ACCEPT_RATES.keys()) or '(empty)'}."
        )
