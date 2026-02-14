#!/usr/bin/env python3
"""
vLLM GEN (decode) log parser for rate-matching.

Parses vLLM decode worker logs to extract GEN SOL metrics.
Implements the GENLogParser interface from parser_base.py.

Status: STUB — not yet implemented. Log format analysis and metric
extraction need to be written once vLLM SOL benchmarks are available.

Usage (pipeline):
    from process_gen_results_vllm import VllmGENLogParser
    parser = VllmGENLogParser()
    log_file = parser.find_log(logs_dir)
    data = parser.parse(log_file)
    result = parser.process(data, concurrency=32, mode='tep', tp=8)

Implementation guide:
    The parser must extract the same metrics as the TRT-LLM parser
    (see process_gen_results.py) but from vLLM's log format:

    1. find_log(): Locate the vLLM decode worker log file.
       - TRT-LLM uses: *_decode_w*.out, *_agg_w*.out
       - vLLM naming depends on srt-slurm's vLLM backend worker setup.

    2. parse(): Extract per-iteration entries from the log.
       - Need to identify vLLM's per-iteration decode logging format.
       - Each entry should capture at minimum:
         * iteration number
         * step time (device or wall-clock)
         * number of generation tokens
         * number of scheduled requests (active decodes)
         * rank (for DEP: expert parallel rank)
       - Return a list of dicts with engine-specific keys.

    3. process(): Filter and aggregate for ONE concurrency level.
       - Filter for pure decode iterations (no prefill tokens)
       - Apply warmup/cooldown trimming
       - Filter for exact concurrency match:
         * TEP: num_requests == concurrency
         * DEP: num_requests == concurrency / ep_rank
       - Filter outliers
       - Compute:
         * elapsed_time_avg = mean(step_time) in seconds
         * throughput_per_user = (1 / elapsed_time_avg) * mtp_accept_rate
         * tpot = elapsed_time_avg / mtp_accept_rate
         * output_throughput = throughput_per_user * concurrency
       - Return GENResult dict

    4. get_mtp_accept_rate(): Return MTP accept rate for vLLM.
       - vLLM may have different MTP semantics than TRT-LLM.
       - For now, can reuse the same override-based approach.

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
# MTP accept rate defaults for vLLM
#
# These may differ from TRT-LLM values. Update once vLLM MTP benchmarks
# are available. For now, these are placeholders based on TRT-LLM measurements.
#
# Format: VLLM_MTP_ACCEPT_RATES[isl][mtp_num] = tokens_per_step_per_user
# ---------------------------------------------------------------------------
VLLM_MTP_ACCEPT_RATES: dict[int, dict[int, float]] = {
    # TODO: Populate with vLLM-specific MTP accept rates once measured.
    # Example structure (using TRT-LLM DSR1 values as placeholders):
    # 1024: {1: 1.8, 2: 2.28, 3: 2.56},
    # 8192: {1: 1.84, 2: 2.38, 3: 2.76},
}


@register_gen_parser("vllm")
class VllmGENLogParser(GENLogParser):
    """vLLM decode log parser.

    TODO: Implement once vLLM SOL benchmark log format is characterised.

    Expected log patterns to look for (update once known):
        - vLLM logs decode iteration stats when ``--log-stats`` is enabled
        - The srt-slurm vLLM backend may produce structured output
        - Check vLLM's engine output for per-step token counts
    """

    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the vLLM decode worker log file.

        TODO: Determine vLLM's log file naming convention in srt-slurm.
        Candidate patterns (update based on actual vLLM backend output):
            - *_decode_*.log
            - vllm_worker_*.out
            - *_vllm_*.out
        """
        # TODO: Update glob patterns based on actual vLLM srt-slurm output
        patterns = [
            "*_decode_*.log",
            "*_vllm_decode_*.out",
            "*_decode_w*.out",
        ]
        for pattern in patterns:
            matches = list(logs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse per-iteration entries from a vLLM decode log.

        TODO: Implement vLLM log parsing. Each entry dict should contain
        at minimum:
            - iter: int (iteration number)
            - step_time_ms: float (device step time in milliseconds)
            - num_generation_tokens: int (decode tokens this iteration)
            - num_scheduled_requests: int (active decodes)
            - num_ctx_tokens: int (prefill tokens — should be 0 for pure decode)
            - rank: int (worker rank, 0-indexed)
            - global_rank: int (global rank across all workers)

        These keys are consumed by process() below; adjust both together.
        """
        raise NotImplementedError(
            "vLLM GEN log parsing is not yet implemented. "
            "See process_gen_results.py (TRT-LLM) for reference implementation. "
            "Contributions welcome — implement parse() to extract per-iteration "
            "decode metrics from vLLM worker logs."
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
        """Process parsed vLLM decode entries into GEN SOL metrics.

        TODO: Implement filtering and aggregation for one concurrency level:
            1. Filter for pure decode iterations (no prefill tokens)
            2. Merge duplicate rows by (iter, rank)
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
            "vLLM GEN result processing is not yet implemented. "
            "See process_gen_results.py (TRT-LLM) for reference implementation."
        )

    def get_mtp_accept_rate(
        self,
        isl: int,
        mtp_num: int,
        overrides: dict[int, float] | None = None,
    ) -> float:
        """Return the MTP accept rate for vLLM at given ISL and MTP level.

        For STP (mtp_num=0), always returns 1.0.
        For MTP, checks overrides first, then VLLM_MTP_ACCEPT_RATES table.

        Note: vLLM's MTP implementation may differ from TRT-LLM's. The
        accept rates are model-dependent and must be measured empirically.
        Until vLLM MTP benchmarks are run, use overrides from the sweep YAML.
        """
        if mtp_num == 0:
            return 1.0

        # Check overrides from sweep YAML first
        if overrides and mtp_num in overrides:
            return overrides[mtp_num]

        # Check built-in table
        if isl in VLLM_MTP_ACCEPT_RATES and mtp_num in VLLM_MTP_ACCEPT_RATES[isl]:
            return VLLM_MTP_ACCEPT_RATES[isl][mtp_num]

        raise ValueError(
            f"No vLLM MTP accept rate for ISL={isl}, mtp_num={mtp_num}. "
            f"Provide mtp_accept_rates in the sweep YAML config. "
            f"Built-in table has ISLs: {sorted(VLLM_MTP_ACCEPT_RATES.keys()) or '(empty)'}."
        )
