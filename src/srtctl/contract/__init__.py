# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared API contract between srtslurm (status reporter) and tripwire (status API).

This package defines the canonical Pydantic models and enums for the Status API.
It has zero internal imports — only depends on pydantic.

Usage (srtslurm side):
    from srtctl.contract import JobStatus, JobStage, JobCreatePayload, JobUpdatePayload

Usage (tripwire side):
    from srtctl.contract import JobStatus, JobStage, JobCreatePayload, JobUpdatePayload
    from srtctl.contract import JobResponse, JobSummary, JobDetail, JobListResponse
"""

from srtctl.contract.enums import JobStage, JobStatus
from srtctl.contract.requests import JobCreatePayload, JobUpdatePayload
from srtctl.contract.responses import JobDetail, JobListResponse, JobResponse, JobSummary

__all__ = [
    "JobStage",
    "JobStatus",
    "JobCreatePayload",
    "JobUpdatePayload",
    "JobResponse",
    "JobSummary",
    "JobDetail",
    "JobListResponse",
]
