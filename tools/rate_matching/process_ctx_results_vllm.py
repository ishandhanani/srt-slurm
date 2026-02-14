#!/usr/bin/env python3
"""
vLLM CTX (prefill) log parser for rate-matching.

Parses vLLM prefill worker logs to extract CTX SOL metrics.
Implements the CTXLogParser interface from parser_base.py.

Status: STUB — not yet implemented. Log format analysis and metric
extraction need to be written once vLLM SOL benchmarks are available.

Usage (pipeline):
    from process_ctx_results_vllm import VllmCTXLogParser
    parser = VllmCTXLogParser()
    log_file = parser.find_log(logs_dir)
    data = parser.parse(log_file)
    result = parser.process(data, isl=1024, max_batch_size=8)

Implementation guide:
    The parser must extract the same metrics as the TRT-LLM parser
    (see process_ctx_results.py) but from vLLM's log format:

    1. find_log(): Locate the vLLM prefill worker log file.
       - TRT-LLM uses: *_prefill_w*.out, *_agg_w*.out
       - vLLM likely uses a different naming convention depending on
         how srt-slurm launches vLLM workers. Check the vLLM backend
         in src/srtctl/backends/ for log file naming.

    2. parse(): Extract per-iteration entries from the log.
       - Need to identify vLLM's per-iteration logging format.
       - Each entry should capture at minimum:
         * iteration number
         * step time (device or wall-clock)
         * number of prefill tokens processed
         * number of requests in the batch
       - Return a list of dicts with engine-specific keys.

    3. process(): Filter and aggregate parsed entries into CTXResult.
       - Apply warmup/cooldown trimming (skip first/last N iterations)
       - Filter for fully-packed batches (num_requests >= max_batch_size)
       - Filter outliers (median ± threshold)
       - Compute:
         * ctx_throughput_tokens_per_s = num_tokens / step_time
         * request_rate_req_per_s = num_requests / total_time
       - Return CTXResult dict (see parser_base.py for field definitions)

    Reference: process_ctx_results.py for the complete TRT-LLM implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parser_base import CTXLogParser, CTXResult, register_ctx_parser


@register_ctx_parser("vllm")
class VllmCTXLogParser(CTXLogParser):
    """vLLM prefill log parser.

    TODO: Implement once vLLM SOL benchmark log format is characterised.

    Expected log patterns to look for (update once known):
        - vLLM typically logs to stderr with iteration-level stats
        - The srt-slurm vLLM backend may produce structured JSON logs
        - Check vLLM's ``--log-stats`` output format
    """

    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the vLLM prefill worker log file.

        TODO: Determine vLLM's log file naming convention in srt-slurm.
        Candidate patterns (update based on actual vLLM backend output):
            - *_prefill_*.log
            - vllm_worker_*.out
            - *_vllm_*.out
        """
        # TODO: Update glob patterns based on actual vLLM srt-slurm output
        patterns = [
            "*_prefill_*.log",
            "*_vllm_prefill_*.out",
            "*_prefill_w*.out",
        ]
        for pattern in patterns:
            matches = list(logs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse per-iteration entries from a vLLM prefill log.

        TODO: Implement vLLM log parsing. Each entry dict should contain
        at minimum:
            - iter: int (iteration number)
            - step_time_ms: float (device step time in milliseconds)
            - num_prefill_tokens: int (tokens processed this iteration)
            - num_requests: int (requests in the batch)
            - rank: int (worker rank, 0-indexed)

        These keys are consumed by process() below; adjust both together.
        """
        raise NotImplementedError(
            "vLLM CTX log parsing is not yet implemented. "
            "See process_ctx_results.py (TRT-LLM) for reference implementation. "
            "Contributions welcome — implement parse() to extract per-iteration "
            "prefill metrics from vLLM worker logs."
        )

    def process(
        self,
        data: list[dict],
        isl: int,
        *,
        verbose: bool = False,
        max_batch_size: int | None = None,
    ) -> CTXResult:
        """Process parsed vLLM prefill entries into CTX SOL metrics.

        TODO: Implement filtering and aggregation:
            1. Remove warmup/cooldown iterations
            2. Filter for fully-packed batches (if max_batch_size set)
            3. Filter outliers
            4. Compute ctx_throughput_tokens_per_s and request_rate_req_per_s

        Must return a CTXResult dict on success (see parser_base.py):
            {
                "ctx_throughput_tokens_per_s": float,
                "request_rate_req_per_s": float,
                "avg_prev_device_step_time_ms": float,
                "avg_num_ctx_tokens": float,
                "avg_num_ctx_requests": float,
                "num_iterations": int,
                "num_ranks": int,
                "isl": isl,
                "threshold_used": int,
            }

        Or {"error": "description"} on failure.
        """
        raise NotImplementedError(
            "vLLM CTX result processing is not yet implemented. "
            "See process_ctx_results.py (TRT-LLM) for reference implementation."
        )
