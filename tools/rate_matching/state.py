"""Sweep state management: persistence, serialisation, and job record types.

The ``SweepState`` class is the single mutable data structure that tracks
progress across all phases.  It is serialised to ``sweep_state.json`` after
every mutation so the orchestrator can resume from any point.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Job record TypedDicts
# ---------------------------------------------------------------------------

class JobRecord(TypedDict, total=False):
    """Base fields shared by all job records."""
    config_path: str
    status: str
    job_id: int | None
    output_dir: str
    submit_time: str
    complete_time: str
    retry_history: list[dict]


class CTXJobRecord(JobRecord, total=False):
    """CTX SOL job metadata."""
    max_batch_size: int


class GENJobRecord(JobRecord, total=False):
    """GEN SOL job metadata."""
    gen_item_index: int
    concurrency: int
    results: list[dict]


class E2EJobRecord(JobRecord, total=False):
    """E2E validation job metadata."""
    pareto_rank: int
    multiplier: float
    per_worker_concurrency: int
    system_concurrency: int
    config_name: str
    result: dict


# ---------------------------------------------------------------------------
# SweepState
# ---------------------------------------------------------------------------

class SweepState:
    """Mutable sweep state, serialised to JSON after each phase."""

    def __init__(self) -> None:
        self.sweep_name: str = ""
        self.sweep_config_path: str = ""
        self.output_dir: str = ""
        self.srtctl_root: str = str(Path(__file__).resolve().parent.parent.parent)
        self.created_at: str = ""
        self.last_updated: str = ""
        self.phase: str = "init"  # init, ctx, gen, rate_match, pareto, e2e, complete

        # Job tracking
        self.ctx_job: CTXJobRecord = {}  # type: ignore[assignment]
        self.gen_jobs: list[GENJobRecord] = []
        self.ctx_result: dict = {}
        self.gen_results: list[dict] = []

        # Analysis
        self.rate_matching_results: list[dict] = []
        self.pareto_frontier: list[dict] = []

        # E2E
        self.e2e_configs: list[dict] = []  # list of {config_path, pareto_rank, multiplier, ...}
        self.e2e_jobs: list[E2EJobRecord] = []
        self.e2e_results: list[dict] = []
        self.sol_vs_e2e: list[dict] = []

    def save(self, path: str | None = None) -> None:
        """Persist state to JSON atomically (write-tmp-then-rename).

        Writing to a temporary file first and then renaming avoids leaving
        a corrupt ``sweep_state.json`` if the process is killed mid-write.
        ``os.replace`` is atomic on POSIX when source and destination are on
        the same filesystem, which is guaranteed here because we create the
        temp file in the same directory as the target.
        """
        target = Path(path) if path else Path(self.output_dir) / "sweep_state.json"
        self.last_updated = datetime.now().isoformat()
        # Write to a temp file in the same directory, then atomically replace
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp", prefix=".sweep_state_",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.__dict__, f, indent=2, default=str)
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up the temp file on any failure (including signals)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_backup(self) -> str:
        """Create a timestamped backup of sweep_state.json.

        Called before destructive mutations (e.g. ``add-e2e``) so the user can
        recover if something goes wrong.

        Returns:
            Path to the backup file.
        """
        source = Path(self.output_dir) / "sweep_state.json"
        if not source.exists():
            raise FileNotFoundError(f"No state file to back up: {source}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = source.with_name(f"sweep_state.json.bak.{ts}")
        shutil.copy2(str(source), str(backup))
        return str(backup)

    @classmethod
    def load(cls, path: str) -> SweepState:
        state = cls()
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state
