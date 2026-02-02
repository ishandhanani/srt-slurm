# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fire-and-forget status reporter for external job tracking.

This module provides optional status reporting to an external API endpoint.
If the endpoint is not configured or unreachable, operations silently continue.
The API contract is defined in srtctl.contract.

Configuration (in srtslurm.yaml or recipe YAML):
    reporting:
      status:
        endpoint: "https://status.example.com"
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import requests

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ReportingConfig, SrtConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatusReporter:
    """Fire-and-forget status reporter.

    Reports job status to an external API if reporting.status.endpoint is configured.
    All operations are non-blocking and failures are silently logged.

    Usage:
        reporter = StatusReporter.from_config(config.reporting, job_id="12345")
        reporter.report(JobStatus.WORKERS_READY, stage=JobStage.WORKERS)
    """

    job_id: str
    api_endpoint: str | None = None
    timeout: float = 5.0

    @classmethod
    def from_config(cls, reporting: "ReportingConfig | None", job_id: str) -> "StatusReporter":
        """Create reporter from reporting config.

        Args:
            reporting: ReportingConfig from srtslurm.yaml or recipe
            job_id: SLURM job ID

        Returns:
            StatusReporter instance (disabled if no endpoint configured)
        """
        endpoint = None
        if reporting and reporting.status and reporting.status.endpoint:
            endpoint = reporting.status.endpoint.rstrip("/")
            logger.info("Status reporting enabled: %s", endpoint)

        return cls(job_id=job_id, api_endpoint=endpoint)

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return self.api_endpoint is not None

    def _now_iso(self) -> str:
        """Get current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def report(
        self,
        status: JobStatus,
        stage: JobStage | None = None,
        message: str | None = None,
    ) -> bool:
        """Report status update (fire-and-forget).

        Args:
            status: New job status
            stage: Current execution stage
            message: Optional human-readable message

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload = JobUpdatePayload(
                status=status.value,
                updated_at=self._now_iso(),
                stage=stage.value if stage else None,
                message=message,
            )

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload.model_dump(exclude_none=True), timeout=self.timeout)

            if response.status_code == 200:
                logger.debug("Status reported: %s", status.value)
                return True
            else:
                logger.debug("Status report failed: HTTP %d", response.status_code)
                return False

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False

    def report_started(self, config: "SrtConfig", runtime: "RuntimeContext") -> bool:
        """Report job started with initial metadata.

        Args:
            config: Job configuration
            runtime: Runtime context with computed values

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            metadata = {
                "model": {
                    "path": str(config.model.path),
                    "precision": config.model.precision,
                },
                "resources": {
                    "gpu_type": config.resources.gpu_type,
                    "gpus_per_node": config.resources.gpus_per_node,
                    "prefill_workers": config.resources.num_prefill,
                    "decode_workers": config.resources.num_decode,
                    "agg_workers": config.resources.num_agg,
                },
                "benchmark": {
                    "type": config.benchmark.type,
                },
                "backend_type": config.backend_type,
                "frontend_type": config.frontend.type,
                "head_node": runtime.nodes.head,
            }

            payload = JobUpdatePayload(
                status=JobStatus.STARTING.value,
                stage=JobStage.STARTING.value,
                message=f"Job started on {runtime.nodes.head}",
                started_at=self._now_iso(),
                updated_at=self._now_iso(),
                metadata=metadata,
            )

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload.model_dump(exclude_none=True), timeout=self.timeout)
            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False

    def report_completed(self, exit_code: int) -> bool:
        """Report job completed with exit code.

        Args:
            exit_code: Process exit code (0 = success)

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
            message = "Benchmark completed successfully" if exit_code == 0 else f"Job failed with exit code {exit_code}"

            payload = JobUpdatePayload(
                status=status.value,
                stage=JobStage.CLEANUP.value,
                message=message,
                completed_at=self._now_iso(),
                updated_at=self._now_iso(),
                exit_code=exit_code,
            )

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload.model_dump(exclude_none=True), timeout=self.timeout)
            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False


def create_job_record(
    reporting: "ReportingConfig | None",
    job_id: str,
    job_name: str,
    cluster: str | None = None,
    recipe: str | None = None,
    metadata: dict | None = None,
) -> bool:
    """Create initial job record in status API (called at submission time).

    This is a standalone function used by submit.py before the job starts.

    Args:
        reporting: ReportingConfig from srtslurm.yaml or recipe
        job_id: SLURM job ID
        job_name: Job/config name
        cluster: Cluster name (optional)
        recipe: Path to recipe file (optional)
        metadata: Job metadata dict (may include "tags" list)

    Returns:
        True if created successfully, False otherwise
    """
    if not reporting or not reporting.status or not reporting.status.endpoint:
        return False

    api_endpoint = reporting.status.endpoint.rstrip("/")

    try:
        payload = JobCreatePayload(
            job_id=job_id,
            job_name=job_name,
            submitted_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            cluster=cluster,
            recipe=recipe,
            metadata=metadata,
        )

        url = f"{api_endpoint}/api/jobs"
        response = requests.post(url, json=payload.model_dump(exclude_none=True), timeout=5.0)

        if response.status_code == 201:
            logger.debug("Job record created: %s", job_id)
            return True
        else:
            logger.debug("Job record creation failed: HTTP %d", response.status_code)
            return False

    except requests.exceptions.RequestException as e:
        logger.debug("Job record creation error (ignored): %s", e)
        return False
