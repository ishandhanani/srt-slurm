# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mock Status API server for integration testing.

Uses shared contract models to validate payloads match the API spec.
Stores all events in-memory for assertion in tests.
"""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from srtctl.contract import (
    JobCreatePayload,
    JobResponse,
    JobStatus,
    JobUpdatePayload,
)

app = FastAPI()

# In-memory storage for assertions
jobs: dict[str, dict] = {}
events: list[dict] = []


def reset():
    """Clear all stored data between tests."""
    jobs.clear()
    events.clear()


@app.post("/api/jobs", response_model=JobResponse, status_code=201)
async def create_job(payload: JobCreatePayload):
    """Create a job record. Validates payload against shared contract."""
    job_data = payload.model_dump(exclude_none=True)
    job_data["status"] = JobStatus.SUBMITTED.value
    jobs[payload.job_id] = job_data
    events.append({"type": "create", "job_id": payload.job_id, **job_data})
    return JobResponse(job_id=payload.job_id, status=JobStatus.SUBMITTED.value)


@app.put("/api/jobs/{job_id}", response_model=JobResponse)
async def update_job(job_id: str, payload: JobUpdatePayload):
    """Update job status. Validates payload against shared contract."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    update_data = payload.model_dump(exclude_none=True)
    jobs[job_id].update(update_data)
    events.append({"type": "update", "job_id": job_id, **update_data})
    return JobResponse(job_id=job_id, status=payload.status)


def create_test_client() -> TestClient:
    """Create a fresh TestClient with clean state."""
    reset()
    return TestClient(app)
