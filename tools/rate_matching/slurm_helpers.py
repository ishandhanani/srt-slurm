"""SLURM job submission, polling, and retry helpers.

All interaction with the SLURM scheduler (``srtctl apply``, ``squeue``,
``sacct``) is contained here so the orchestrator does not need to import
``subprocess`` directly.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from state import SweepState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_submit_path(job_dict: dict) -> str:
    """Build the submit path for a job, appending :selector if present.

    Override-mode jobs have a ``selector`` key (e.g. "base" or
    "override_c16").  The submit path becomes ``config_path:selector``.
    """
    path = job_dict["config_path"]
    selector = job_dict.get("selector")
    if selector:
        return f"{path}:{selector}"
    return path


# ---------------------------------------------------------------------------
# Submit / poll primitives
# ---------------------------------------------------------------------------

def submit_job(config_path: str, verbose: bool = True) -> str:
    """Submit a job via ``srtctl apply -f <config>`` and return the SLURM job ID."""
    cmd = ["srtctl", "apply", "-f", config_path]
    if verbose:
        print(f"  Submitting: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(
            f"srtctl apply failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    # Parse job ID from output.  srtctl prints "Submitted job <ID>".
    # Strip ANSI escape codes first (Rich formatting can wrap the job ID).
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    clean_output = ansi_escape.sub("", result.stdout)
    for line in clean_output.strip().split("\n"):
        for token in line.split():
            if token.isdigit():
                return token

    # Fallback: check SLURM_JOB_ID in environment
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        return slurm_id

    raise RuntimeError(f"Could not parse job ID from srtctl output:\n{result.stdout}")


def poll_job(
    job_id: str,
    poll_interval: int = 300,
    verbose: bool = True,
    max_poll_time: int = 14400,
) -> str:
    """Poll SLURM job until completion. Returns final status.

    Args:
        job_id: SLURM job ID to monitor.
        poll_interval: Seconds between squeue polls.
        verbose: Print progress messages.
        max_poll_time: Maximum seconds to poll before returning TIMEOUT_POLL.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > max_poll_time:
            if verbose:
                print(f"  Job {job_id}: poll timeout after {elapsed:.0f}s")
            return "TIMEOUT_POLL"

        try:
            result = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True,
            )
            status = result.stdout.strip()
        except Exception:
            status = ""

        if not status:
            # Job no longer in queue -- check sacct for final status
            try:
                result = subprocess.run(
                    ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                    capture_output=True, text=True,
                )
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                if lines:
                    status = lines[0]
                else:
                    status = "UNKNOWN"
            except Exception:
                status = "UNKNOWN"
            break

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            break

        if verbose:
            print(f"  Job {job_id}: {status} (next check in {poll_interval}s)")
        time.sleep(poll_interval)

    if verbose:
        print(f"  Job {job_id}: {status}")
    return status


def get_job_output_dir(job_id: str, srtctl_root: str | None = None) -> str:
    """Get the output directory for a SLURM job.

    Args:
        job_id: SLURM job ID.
        srtctl_root: Root of the srt-slurm repo.  Falls back to deriving
            from ``__file__`` if not provided (original behaviour).
    """
    root = Path(srtctl_root) if srtctl_root else Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs" / str(job_id)
    return str(output_dir)


# ---------------------------------------------------------------------------
# Submit-poll-retry (serial)
# ---------------------------------------------------------------------------

def _submit_and_poll(
    job_dict: dict,
    config_path: str,
    poll_interval: int,
    max_retries: int,
    state: SweepState,
    verbose: bool = True,
    max_poll_time: int = 14400,
) -> str:
    """Submit a SLURM job, poll to completion, and retry on failure.

    This is the common submit -> poll -> retry loop used by CTX, GEN, and
    E2E phases.  On failure the job is resubmitted up to *max_retries* times
    (total attempts = 1 + max_retries).  Each attempt's metadata is appended
    to ``job_dict["retry_history"]``.

    Args:
        job_dict: Mutable dict from ``state.ctx_job``, ``state.gen_jobs[i]``,
            or ``state.e2e_jobs[i]``.  Updated in-place with ``job_id``,
            ``status``, ``output_dir``, timestamps, and ``retry_history``.
        config_path: Path to the srt-slurm YAML config to submit.
        poll_interval: Seconds between ``squeue`` polls.
        max_retries: Maximum number of *retries* (not including the first
            attempt).
        state: The parent ``SweepState`` -- ``state.save()`` is called after
            each status change so progress is durable.
        verbose: Print progress.
        max_poll_time: Maximum seconds to poll a single job.

    Returns:
        Final SLURM status string (e.g. ``"COMPLETED"``).

    Raises:
        RuntimeError: If the job fails after all retries are exhausted.
    """
    if "retry_history" not in job_dict:
        job_dict["retry_history"] = []

    attempts = 0
    max_attempts = 1 + max_retries

    while attempts < max_attempts:
        attempts += 1
        attempt_label = f"(attempt {attempts}/{max_attempts})" if max_attempts > 1 else ""

        # Submit (skip if already running from a prior resume)
        if not job_dict.get("job_id") or job_dict.get("status") in ("failed", "pending"):
            try:
                job_id = submit_job(config_path, verbose=verbose)
            except RuntimeError as exc:
                if verbose:
                    print(f"  Submit failed {attempt_label}: {exc}")
                job_dict["retry_history"].append({
                    "attempt": attempts,
                    "event": "submit_failed",
                    "error": str(exc),
                    "time": datetime.now().isoformat(),
                })
                state.save()
                if attempts < max_attempts:
                    if verbose:
                        print(f"  Retrying in {poll_interval}s...")
                    time.sleep(poll_interval)
                    continue
                raise RuntimeError(
                    f"Job submission failed after {attempts} attempt(s): {exc}"
                ) from exc

            job_dict["job_id"] = int(job_id)
            job_dict["submit_time"] = datetime.now().isoformat()
            job_dict["status"] = "running"
            job_dict["output_dir"] = get_job_output_dir(job_id, srtctl_root=state.srtctl_root)
            job_dict["retry_history"].append({
                "attempt": attempts,
                "event": "submitted",
                "job_id": int(job_id),
                "time": datetime.now().isoformat(),
            })
            state.save()

        # Poll
        status = poll_job(str(job_dict["job_id"]), poll_interval, verbose=verbose, max_poll_time=max_poll_time)
        is_success = "COMPLETED" in status.upper()
        job_dict["status"] = "completed" if is_success else "failed"
        job_dict["complete_time"] = datetime.now().isoformat()

        job_dict["retry_history"].append({
            "attempt": attempts,
            "event": "completed" if is_success else "failed",
            "job_id": job_dict["job_id"],
            "slurm_status": status,
            "time": datetime.now().isoformat(),
        })
        state.save()

        if is_success:
            return status

        # Failed -- retry?
        if attempts < max_attempts:
            if verbose:
                print(
                    f"  Job {job_dict['job_id']} failed ({status}) {attempt_label}. "
                    f"Retrying..."
                )
            # Clear job_id so the next iteration submits a fresh job
            job_dict["job_id"] = None
            job_dict["status"] = "pending"
            state.save()
        else:
            raise RuntimeError(
                f"Job failed with status {status} after {attempts} attempt(s). "
                f"Config: {config_path}"
            )

    # Should not reach here, but just in case
    return status  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Submit-poll-retry (parallel)
# ---------------------------------------------------------------------------

def _submit_poll_parallel(
    job_dicts: list[dict],
    poll_interval: int,
    max_retries: int,
    max_poll_time: int,
    state: SweepState,
    verbose: bool = True,
    label: str = "job",
) -> None:
    """Submit multiple SLURM jobs in parallel, poll all, then retry failures.

    This is the parallel counterpart to the serial ``_submit_and_poll`` loop.
    Each retry round re-submits all jobs that are not yet ``"completed"`` and
    polls them all before moving to the next round.

    Args:
        job_dicts: Mutable list of job record dicts — updated in-place.
        poll_interval: Seconds between squeue polls.
        max_retries: Maximum retry rounds (total attempts = 1 + max_retries).
        max_poll_time: Maximum seconds to poll a single job.
        state: Parent ``SweepState`` — ``state.save()`` called after mutations.
        verbose: Print progress.
        label: Human-readable label for log messages (e.g. "GEN", "E2E").
    """
    for attempt_round in range(1 + max_retries):
        to_submit = [j for j in job_dicts if j.get("status") not in ("completed",)]
        if not to_submit:
            break
        if attempt_round > 0 and verbose:
            print(f"  Retry round {attempt_round}/{max_retries} for {len(to_submit)} failed {label} job(s)")

        # Submit phase — save after each submission so that if the process
        # is killed mid-batch, already-submitted jobs are recorded and won't
        # be duplicated on resume.
        for j in to_submit:
            if not j.get("job_id") or j.get("status") in ("failed", "pending"):
                try:
                    # Support selector-aware paths (config_path:selector)
                    submit_path = _resolve_submit_path(j)
                    job_id = submit_job(submit_path, verbose=verbose)
                    j["job_id"] = int(job_id)
                    j["submit_time"] = datetime.now().isoformat()
                    j["status"] = "running"
                    j["output_dir"] = get_job_output_dir(job_id, srtctl_root=state.srtctl_root)
                    j.setdefault("retry_history", []).append({
                        "attempt": attempt_round + 1,
                        "event": "submitted",
                        "job_id": int(job_id),
                        "time": datetime.now().isoformat(),
                    })
                except RuntimeError as exc:
                    if verbose:
                        print(f"  {label} submit failed: {exc}")
                    j.setdefault("retry_history", []).append({
                        "attempt": attempt_round + 1,
                        "event": "submit_failed",
                        "error": str(exc),
                        "time": datetime.now().isoformat(),
                    })
                    j["status"] = "failed"
                state.save()

        # Poll phase
        for j in to_submit:
            if j.get("status") != "running":
                continue
            status = poll_job(str(j["job_id"]), poll_interval, verbose=verbose, max_poll_time=max_poll_time)
            is_ok = "COMPLETED" in status.upper()
            j["status"] = "completed" if is_ok else "failed"
            j["complete_time"] = datetime.now().isoformat()
            j.setdefault("retry_history", []).append({
                "attempt": attempt_round + 1,
                "event": "completed" if is_ok else "failed",
                "job_id": j["job_id"],
                "slurm_status": status,
                "time": datetime.now().isoformat(),
            })
            if not is_ok:
                j["job_id"] = None  # clear so next round resubmits
            state.save()
