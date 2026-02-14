#!/usr/bin/env python3
"""
SGLang CTX (prefill) log parser for rate-matching.

Parses SGLang prefill worker logs to extract CTX SOL metrics.
Implements the CTXLogParser interface from parser_base.py.

Status: STUB — not yet implemented. Log format analysis and metric
extraction need to be written once SGLang SOL benchmarks are available.

Usage (pipeline):
    from process_ctx_results_sglang import SglangCTXLogParser
    parser = SglangCTXLogParser()
    log_file = parser.find_log(logs_dir)
    data = parser.parse(log_file)
    result = parser.process(data, isl=1024, max_batch_size=8)

Implementation guide:
    The parser must extract the same metrics as the TRT-LLM parser
    (see process_ctx_results.py) but from SGLang's log format:

    1. find_log(): Locate the SGLang prefill worker log file.
       - TRT-LLM uses: *_prefill_w*.out, *_agg_w*.out
       - SGLang uses per-process srun launching (see SGLang backend in
         src/srtctl/backends/). Log files may follow a different naming
         convention than TRT-LLM's MPI-style output.

    2. parse(): Extract per-iteration entries from the log.
       - SGLang's RadixAttention engine has its own scheduling loop.
       - Look for per-batch or per-step metrics in SGLang's output.
       - Each entry should capture:
         * iteration/step number
         * step time (prefill processing time)
         * number of prefill tokens processed
         * number of requests in the batch
       - Return a list of dicts with engine-specific keys.

    3. process(): Filter and aggregate parsed entries into CTXResult.
       - Apply warmup/cooldown trimming
       - Filter for fully-packed batches if max_batch_size set
       - Filter outliers
       - Compute ctx_throughput_tokens_per_s and request_rate_req_per_s
       - Return CTXResult dict (see parser_base.py for field definitions)

    SGLang-specific notes:
        - SGLang uses a different scheduling algorithm (RadixAttention)
          which may affect prefill batching behaviour.
        - The srt-slurm SGLang backend already supports prefill/decode/
          aggregated modes — check the backend for worker log paths.
        - SGLang may log metrics via its built-in metrics server
          (Prometheus format) rather than per-iteration stdout. Consider
          whether to parse stdout logs or query the metrics endpoint.

    Reference: process_ctx_results.py for the complete TRT-LLM implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parser_base import CTXLogParser, CTXResult, register_ctx_parser


@register_ctx_parser("sglang")
class SglangCTXLogParser(CTXLogParser):
    """SGLang prefill log parser.

    TODO: Implement once SGLang SOL benchmark log format is characterised.

    SGLang architecture notes:
        - Uses RadixAttention for KV cache management
        - Per-process srun launching (unlike TRT-LLM's MPI style)
        - May expose metrics via Prometheus endpoint (:9090/metrics)
        - The srt-slurm SGLang backend handles prefill/decode/aggregated modes
    """

    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the SGLang prefill worker log file.

        TODO: Determine SGLang's log file naming convention in srt-slurm.
        The SGLang backend uses per-process srun launching, so log files
        may be named differently from TRT-LLM's MPI-style output.

        Candidate patterns (update based on actual SGLang backend output):
            - *_prefill_*.log
            - *_sglang_*.out
            - sglang_worker_*.out
        """
        # TODO: Update glob patterns based on actual SGLang srt-slurm output
        patterns = [
            "*_prefill_*.log",
            "*_sglang_prefill_*.out",
            "*_prefill_w*.out",
        ]
        for pattern in patterns:
            matches = list(logs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse per-iteration entries from an SGLang prefill log.

        TODO: Implement SGLang log parsing. Each entry dict should contain
        at minimum:
            - iter: int (iteration/step number)
            - step_time_ms: float (prefill step time in milliseconds)
            - num_prefill_tokens: int (tokens processed this iteration)
            - num_requests: int (requests in the batch)
            - rank: int (worker rank, 0-indexed)

        SGLang-specific considerations:
            - SGLang may use "batch" rather than "iteration" terminology
            - RadixAttention scheduling may produce variable batch sizes
            - Check if SGLang logs per-batch stats to stdout or only via
              the metrics endpoint

        These keys are consumed by process() below; adjust both together.
        """
        raise NotImplementedError(
            "SGLang CTX log parsing is not yet implemented. "
            "See process_ctx_results.py (TRT-LLM) for reference implementation. "
            "Contributions welcome — implement parse() to extract per-iteration "
            "prefill metrics from SGLang worker logs."
        )

    def process(
        self,
        data: list[dict],
        isl: int,
        *,
        verbose: bool = False,
        max_batch_size: int | None = None,
    ) -> CTXResult:
        """Process parsed SGLang prefill entries into CTX SOL metrics.

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
            "SGLang CTX result processing is not yet implemented. "
            "See process_ctx_results.py (TRT-LLM) for reference implementation."
        )
