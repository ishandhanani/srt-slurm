# Status API Specification v1

srtslurm can optionally report job status to an external HTTP API via fire-and-forget POST/PUT requests.

## Configuration

In `srtslurm.yaml` or recipe YAML:

```yaml
reporting:
  status:
    endpoint: "https://status.example.com"
```

If not configured, status reporting is disabled and jobs run normally.

## Endpoints

### POST /api/jobs

Create a job record. Called at submission time.

**Request:**
```json
{
  "job_id": "12345",
  "job_name": "benchmark-run",
  "cluster": "gpu-cluster-01",
  "recipe": "configs/benchmark.yaml",
  "submitted_at": "2025-01-26T10:30:00Z",
  "metadata": {
    "tags": ["pipeline:98765", "suite:kv-router-comparison"]
  }
}
```

**Response:** `201 Created`
```json
{
  "job_id": "12345",
  "status": "submitted"
}
```

### PUT /api/jobs/{job_id}

Update job status. Called during execution and at completion.

**Request (during execution):**
```json
{
  "status": "workers",
  "stage": "workers",
  "message": "Starting workers",
  "updated_at": "2025-01-26T10:35:00Z"
}
```

**Request (at completion):**
```json
{
  "status": "completed",
  "exit_code": 0,
  "logs_url": "s3://bucket/outputs/12345/",
  "benchmark_results": {
    "throughput": 1250.5,
    "latency_p50_ms": 42.1,
    "latency_p99_ms": 128.7
  }
}
```

All fields except `status` are optional.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | **Required.** New job status |
| `stage` | string | Current execution stage |
| `message` | string | Human-readable status message |
| `updated_at` | string | ISO 8601 timestamp (server defaults to now) |
| `started_at` | string | Job start timestamp |
| `completed_at` | string | Job completion timestamp |
| `exit_code` | int | Process exit code |
| `logs_url` | string | S3 URL where logs were uploaded |
| `benchmark_results` | object | Parsed benchmark metrics |
| `metadata` | object | Additional metadata (merged with existing) |

**Response:** `200 OK`
```json
{
  "job_id": "12345",
  "status": "completed"
}
```

### GET /api/jobs/{job_id}

Get full job details including event history.

### GET /api/jobs

List jobs with pagination and filters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Results per page (max 100) |
| `status` | string | - | Filter by status |
| `cluster` | string | - | Filter by cluster |

## Status Values

```text
submitted -> starting -> workers -> frontend -> benchmark -> completed | failed
```

Status reflects which stage is currently executing, not readiness.

## Contract Models

The canonical Pydantic models live in `srtctl.contract`:

```python
from srtctl.contract import (
    JobStatus,          # Status enum
    JobStage,           # Stage enum
    JobCreatePayload,   # POST request body
    JobUpdatePayload,   # PUT request body
    JobResponse,        # POST/PUT response
    JobSummary,         # List endpoint item
    JobDetail,          # GET endpoint response
    JobListResponse,    # List endpoint wrapper
)
```

## Behavior

- All requests have a 5-second timeout
- Failures are logged at DEBUG level and ignored
- Job execution is never blocked by status reporting failures
