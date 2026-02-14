"""Abstract base classes and registry for engine-specific log parsers.

The rate-matching pipeline needs two parsers per engine:
  - CTXLogParser: extracts prefill SOL metrics from worker logs
  - GENLogParser: extracts decode SOL metrics from worker logs

Each engine (TRT-LLM, vLLM, SGLang) produces logs in a different format,
but the pipeline expects the same result shape from all of them. The
TypedDicts below define those contracts.

Implementations:
  - TRT-LLM: process_ctx_results.TrtllmCTXLogParser
             process_gen_results.TrtllmGENLogParser

Adding a new engine parser
--------------------------
1. Create a new module, e.g. ``process_ctx_results_vllm.py``.
2. Subclass ``CTXLogParser`` and/or ``GENLogParser``.
3. Decorate each class with the appropriate registry decorator::

       from parser_base import CTXLogParser, CTXResult, register_ctx_parser

       @register_ctx_parser("vllm")
       class VllmCTXLogParser(CTXLogParser):
           def find_log(self, logs_dir): ...
           def parse(self, log_file, verbose=False): ...
           def process(self, data, isl, *, verbose=False, max_batch_size=None): ...

4. Import the new module in ``run_sweep.py`` (alongside the existing
   ``import process_ctx_results``) so the decorator runs at startup.
5. Set ``engine_type: "vllm"`` in the sweep YAML.  The orchestrator calls
   ``get_ctx_parser(cfg.engine_type)`` / ``get_gen_parser(cfg.engine_type)``
   to obtain the correct parser at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Result contracts
# ---------------------------------------------------------------------------

class CTXResult(TypedDict, total=False):
    """Result dict returned by any CTXLogParser.process().

    Required keys are produced on success. On failure, only 'error' is set.
    """

    # --- success fields ---
    ctx_throughput_tokens_per_s: float
    request_rate_req_per_s: float
    avg_prev_device_step_time_ms: float
    avg_num_ctx_tokens: float
    avg_num_ctx_requests: float
    num_iterations: int
    num_ranks: int
    isl: int
    threshold_used: int

    # --- error field (mutually exclusive with above) ---
    error: str


class GENResult(TypedDict, total=False):
    """Result dict returned by any GENLogParser.process().

    Required keys are produced on success. On failure, only 'error' is set.
    """

    # --- success fields ---
    interactivity: float
    throughput_per_gpu: float
    output_throughput: float
    throughput_per_user: float
    avg_step_time_ms: float
    tpot_ms: float
    concurrency: int
    mode: str
    mtp: int
    mtp_accept_rate: float
    ep_rank: int
    num_gpus: int
    num_iterations: int

    # --- error field (mutually exclusive with above) ---
    error: str


# ---------------------------------------------------------------------------
# Abstract parsers
# ---------------------------------------------------------------------------

class CTXLogParser(ABC):
    """Abstract parser for CTX (prefill) SOL logs.

    Subclasses must implement find_log, parse, and process.
    The pipeline calls them in sequence:

        parser = TrtllmCTXLogParser()    # or VllmCTXLogParser(), etc.
        log_file = parser.find_log(logs_dir)
        raw_entries = parser.parse(log_file)
        result = parser.process(raw_entries, isl=1024, max_batch_size=8)
    """

    @abstractmethod
    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the CTX worker log file within a job's logs directory.

        Returns None if no matching log is found.
        """

    @abstractmethod
    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse raw per-iteration entries from a CTX log file.

        Returns a list of dicts, one per parsed iteration/rank entry.
        The exact keys are engine-specific, but must be consumable by
        this parser's process() method.
        """

    @abstractmethod
    def process(
        self,
        data: list[dict],
        isl: int,
        *,
        verbose: bool = False,
        max_batch_size: int | None = None,
    ) -> CTXResult:
        """Process parsed CTX log entries into SOL metrics.

        Args:
            data: Output of self.parse().
            isl: Input sequence length.
            verbose: Print debug info.
            max_batch_size: If provided, only fully-packed iterations
                (num_ctx_requests >= max_batch_size) are kept.

        Returns:
            CTXResult on success, or {'error': '...'} on failure.
        """


class GENLogParser(ABC):
    """Abstract parser for GEN (decode) SOL logs.

    Subclasses must implement find_log, parse, process, and
    get_mtp_accept_rate.

    The pipeline calls them in sequence:

        parser = TrtllmGENLogParser()    # or VllmGENLogParser(), etc.
        log_file = parser.find_log(logs_dir)
        raw_entries = parser.parse(log_file)
        result = parser.process(raw_entries, concurrency=32, mode='tep', ...)
    """

    @abstractmethod
    def find_log(self, logs_dir: Path) -> Path | None:
        """Locate the decode worker log file within a job's logs directory.

        Returns None if no matching log is found.
        """

    @abstractmethod
    def parse(self, log_file: Path, verbose: bool = False) -> list[dict]:
        """Parse raw per-iteration entries from a GEN log file.

        Returns a list of dicts, one per parsed iteration/rank entry.
        The exact keys are engine-specific, but must be consumable by
        this parser's process() method.
        """

    @abstractmethod
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
        """Process parsed GEN log entries into SOL metrics for one concurrency.

        Args:
            data: Output of self.parse().
            concurrency: Target concurrency to extract metrics for.
            mode: 'tep' or 'dep'.
            tp: Tensor parallelism.
            ep_rank: Expert parallel rank (for DEP). Defaults to tp.
            mtp: MTP layers (0 = STP).
            isl: Input sequence length (for MTP accept rate lookup).
            num_gpus: GPUs used for this decode worker.
            verbose: Print debug info.
            mtp_accept_rate_overrides: Optional overrides for MTP accept
                rates from the sweep YAML.

        Returns:
            GENResult on success, or {'error': '...'} on failure.
        """

    @abstractmethod
    def get_mtp_accept_rate(
        self,
        isl: int,
        mtp_num: int,
        overrides: dict[int, float] | None = None,
    ) -> float:
        """Return the MTP accept rate for a given ISL and MTP level.

        For STP (mtp_num=0), always returns 1.0.
        For MTP, returns tokens_per_step_per_user (e.g. 2.56 for MTP-3).
        """

    def process_all_concurrencies(
        self,
        data: list[dict],
        concurrency_list: list[int],
        mode: str,
        *,
        tp: int = 8,
        ep_rank: int | None = None,
        mtp: int = 0,
        isl: int = 1024,
        num_gpus: int = 8,
        verbose: bool = False,
        mtp_accept_rate_overrides: dict[int, float] | None = None,
    ) -> dict[int, GENResult]:
        """Process GEN log data for MULTIPLE concurrencies from a single log.

        Default implementation loops over concurrency_list and calls
        self.process() for each. Subclasses can override for optimisation.

        Returns:
            Dict mapping concurrency -> GENResult.
        """
        results: dict[int, GENResult] = {}
        for conc in concurrency_list:
            results[conc] = self.process(
                data,
                concurrency=conc,
                mode=mode,
                tp=tp,
                ep_rank=ep_rank,
                mtp=mtp,
                isl=isl,
                num_gpus=num_gpus,
                verbose=verbose,
                mtp_accept_rate_overrides=mtp_accept_rate_overrides,
            )
        return results


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

_CTX_PARSERS: dict[str, type[CTXLogParser]] = {}
_GEN_PARSERS: dict[str, type[GENLogParser]] = {}


def register_ctx_parser(engine: str):
    """Class decorator: register a CTXLogParser implementation for *engine*."""
    def decorator(cls: type[CTXLogParser]) -> type[CTXLogParser]:
        _CTX_PARSERS[engine] = cls
        return cls
    return decorator


def register_gen_parser(engine: str):
    """Class decorator: register a GENLogParser implementation for *engine*."""
    def decorator(cls: type[GENLogParser]) -> type[GENLogParser]:
        _GEN_PARSERS[engine] = cls
        return cls
    return decorator


def get_ctx_parser(engine: str) -> CTXLogParser:
    """Instantiate the registered CTXLogParser for *engine*.

    Raises ``KeyError`` if no parser is registered for that engine.
    """
    try:
        return _CTX_PARSERS[engine]()
    except KeyError:
        available = ", ".join(sorted(_CTX_PARSERS)) or "(none)"
        raise KeyError(f"No CTX parser registered for engine '{engine}'. Available: {available}") from None


def get_gen_parser(engine: str) -> GENLogParser:
    """Instantiate the registered GENLogParser for *engine*.

    Raises ``KeyError`` if no parser is registered for that engine.
    """
    try:
        return _GEN_PARSERS[engine]()
    except KeyError:
        available = ", ".join(sorted(_GEN_PARSERS)) or "(none)"
        raise KeyError(f"No GEN parser registered for engine '{engine}'. Available: {available}") from None
