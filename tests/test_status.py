# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for status reporting functionality."""

from unittest.mock import MagicMock, patch

import pytest

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload
from srtctl.core.schema import ReportingConfig, ReportingStatusConfig
from srtctl.core.status import (
    StatusReporter,
    create_job_record,
)


# ============================================================================
# StatusReporter Tests
# ============================================================================


class TestStatusReporterFromConfig:
    """Test StatusReporter.from_config() factory method."""

    def test_creates_enabled_reporter_with_endpoint(self):
        """Reporter is enabled when endpoint is configured."""
        reporting = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://status.example.com")
        )

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is True
        assert reporter.job_id == "12345"
        assert reporter.api_endpoint == "https://status.example.com"

    def test_creates_disabled_reporter_without_endpoint(self):
        """Reporter is disabled when no endpoint configured."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint=None))

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is False
        assert reporter.api_endpoint is None

    def test_creates_disabled_reporter_with_none_reporting(self):
        """Reporter is disabled when reporting config is None."""
        reporter = StatusReporter.from_config(None, job_id="12345")

        assert reporter.enabled is False
        assert reporter.api_endpoint is None

    def test_creates_disabled_reporter_with_none_status(self):
        """Reporter is disabled when status config is None."""
        reporting = ReportingConfig(status=None)

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.enabled is False

    def test_strips_trailing_slash_from_endpoint(self):
        """Trailing slash is removed from endpoint URL."""
        reporting = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://status.example.com/")
        )

        reporter = StatusReporter.from_config(reporting, job_id="12345")

        assert reporter.api_endpoint == "https://status.example.com"


class TestStatusReporterReport:
    """Test StatusReporter.report() method."""

    def test_returns_false_when_disabled(self):
        """Report returns False when reporter is disabled."""
        reporter = StatusReporter(job_id="12345", api_endpoint=None)

        result = reporter.report(JobStatus.STARTING)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_sends_put_request_to_correct_url(self, mock_put):
        """Report sends PUT request to /api/jobs/{job_id}."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        reporter.report(JobStatus.WORKERS, stage=JobStage.WORKERS)

        mock_put.assert_called_once()
        call_args = mock_put.call_args
        assert call_args[0][0] == "https://status.example.com/api/jobs/12345"

    @patch("srtctl.core.status.requests.put")
    def test_returns_true_on_success(self, mock_put):
        """Report returns True on HTTP 200."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        result = reporter.report(JobStatus.STARTING)

        assert result is True

    @patch("srtctl.core.status.requests.put")
    def test_returns_false_on_http_error(self, mock_put):
        """Report returns False on non-200 status."""
        mock_put.return_value = MagicMock(status_code=500)
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        result = reporter.report(JobStatus.STARTING)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_returns_false_on_request_exception(self, mock_put):
        """Report returns False on network error (fire-and-forget)."""
        import requests

        mock_put.side_effect = requests.exceptions.ConnectionError("Network error")
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        result = reporter.report(JobStatus.STARTING)

        assert result is False


class TestStatusReporterCompleted:
    """Test StatusReporter.report_completed() method."""

    def test_returns_false_when_disabled(self):
        """report_completed returns False when disabled."""
        reporter = StatusReporter(job_id="12345", api_endpoint=None)

        result = reporter.report_completed(exit_code=0)

        assert result is False

    @patch("srtctl.core.status.requests.put")
    def test_reports_completed_status_on_success(self, mock_put):
        """Exit code 0 reports COMPLETED status."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        reporter.report_completed(exit_code=0)

        call_args = mock_put.call_args
        payload = call_args[1]["json"]
        assert payload["status"] == "completed"
        assert payload["exit_code"] == 0

    @patch("srtctl.core.status.requests.put")
    def test_reports_failed_status_on_nonzero_exit(self, mock_put):
        """Non-zero exit code reports FAILED status."""
        mock_put.return_value = MagicMock(status_code=200)
        reporter = StatusReporter(job_id="12345", api_endpoint="https://status.example.com")

        reporter.report_completed(exit_code=1)

        call_args = mock_put.call_args
        payload = call_args[1]["json"]
        assert payload["status"] == "failed"
        assert payload["exit_code"] == 1


# ============================================================================
# Payload Model Tests
# ============================================================================


class TestJobCreatePayload:
    """Test JobCreatePayload Pydantic model."""

    def test_model_dump_includes_required_fields(self):
        """model_dump includes all required fields."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
        )

        result = payload.model_dump(exclude_none=True)

        assert result["job_id"] == "12345"
        assert result["job_name"] == "test-job"
        assert result["submitted_at"] == "2025-01-26T10:00:00Z"

    def test_model_dump_excludes_none_values(self):
        """model_dump(exclude_none=True) excludes fields with None values."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
            cluster=None,
            recipe=None,
            metadata=None,
        )

        result = payload.model_dump(exclude_none=True)

        assert "cluster" not in result
        assert "recipe" not in result
        assert "metadata" not in result

    def test_model_dump_includes_optional_fields_when_set(self):
        """model_dump includes optional fields when they have values."""
        payload = JobCreatePayload(
            job_id="12345",
            job_name="test-job",
            submitted_at="2025-01-26T10:00:00Z",
            cluster="gpu-cluster",
            recipe="configs/test.yaml",
            metadata={"key": "value"},
        )

        result = payload.model_dump(exclude_none=True)

        assert result["cluster"] == "gpu-cluster"
        assert result["recipe"] == "configs/test.yaml"
        assert result["metadata"] == {"key": "value"}


class TestJobUpdatePayload:
    """Test JobUpdatePayload Pydantic model."""

    def test_model_dump_includes_required_fields(self):
        """model_dump includes required fields."""
        payload = JobUpdatePayload(
            status="starting",
            updated_at="2025-01-26T10:00:00Z",
        )

        result = payload.model_dump(exclude_none=True)

        assert result["status"] == "starting"
        assert result["updated_at"] == "2025-01-26T10:00:00Z"

    def test_model_dump_excludes_none_values(self):
        """model_dump(exclude_none=True) excludes fields with None values."""
        payload = JobUpdatePayload(
            status="starting",
            updated_at="2025-01-26T10:00:00Z",
            stage=None,
            message=None,
        )

        result = payload.model_dump(exclude_none=True)

        assert "stage" not in result
        assert "message" not in result

    def test_model_dump_includes_exit_code_when_set(self):
        """model_dump includes exit_code when set."""
        payload = JobUpdatePayload(
            status="completed",
            updated_at="2025-01-26T10:00:00Z",
            exit_code=0,
        )

        result = payload.model_dump(exclude_none=True)

        assert result["exit_code"] == 0


# ============================================================================
# create_job_record Tests
# ============================================================================


class TestCreateJobRecord:
    """Test create_job_record() standalone function."""

    def test_returns_false_when_no_reporting_config(self):
        """Returns False when reporting config is None."""
        result = create_job_record(
            reporting=None,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False

    def test_returns_false_when_no_endpoint(self):
        """Returns False when no endpoint configured."""
        reporting = ReportingConfig(status=ReportingStatusConfig(endpoint=None))

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False

    @patch("srtctl.core.status.requests.post")
    def test_sends_post_request_to_correct_url(self, mock_post):
        """Sends POST request to /api/jobs."""
        mock_post.return_value = MagicMock(status_code=201)
        reporting = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://status.example.com")
        )

        create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://status.example.com/api/jobs"

    @patch("srtctl.core.status.requests.post")
    def test_returns_true_on_201_created(self, mock_post):
        """Returns True on HTTP 201."""
        mock_post.return_value = MagicMock(status_code=201)
        reporting = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://status.example.com")
        )

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is True

    @patch("srtctl.core.status.requests.post")
    def test_returns_false_on_request_exception(self, mock_post):
        """Returns False on network error (fire-and-forget)."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")
        reporting = ReportingConfig(
            status=ReportingStatusConfig(endpoint="https://status.example.com")
        )

        result = create_job_record(
            reporting=reporting,
            job_id="12345",
            job_name="test-job",
        )

        assert result is False


# ============================================================================
# Enum Tests
# ============================================================================


class TestJobStatusEnum:
    """Test JobStatus enum values match API spec."""

    def test_status_values(self):
        """Status values match API spec."""
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.STARTING.value == "starting"
        assert JobStatus.WORKERS.value == "workers"
        assert JobStatus.FRONTEND.value == "frontend"
        assert JobStatus.BENCHMARK.value == "benchmark"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.TIMEOUT.value == "timeout"


class TestJobStageEnum:
    """Test JobStage enum values match API spec."""

    def test_stage_values(self):
        """Stage values match API spec."""
        assert JobStage.STARTING.value == "starting"
        assert JobStage.HEAD_INFRASTRUCTURE.value == "head_infrastructure"
        assert JobStage.WORKERS.value == "workers"
        assert JobStage.FRONTEND.value == "frontend"
        assert JobStage.BENCHMARK.value == "benchmark"
        assert JobStage.CLEANUP.value == "cleanup"
