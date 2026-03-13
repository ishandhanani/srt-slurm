#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Trace Replay Benchmark using aiperf
# Replays requests from a trace file at their original timestamps
#
# Usage: bench.sh ENDPOINT MODEL_NAME TRACE_FILE [TTFT_THRESHOLD] [ITL_THRESHOLD]

set -e

# Ensure Python output is unbuffered for real-time logging
export PYTHONUNBUFFERED=1

ENDPOINT=$1
MODEL_NAME=${2:-"model"}
TRACE_FILE=$3
TTFT_THRESHOLD=${4:-2000}
ITL_THRESHOLD=${5:-25}

# Validate trace file
if [ -z "${TRACE_FILE}" ]; then
    echo "ERROR: TRACE_FILE is required"
    exit 1
fi

if [ ! -f "${TRACE_FILE}" ]; then
    echo "ERROR: Trace file not found: ${TRACE_FILE}"
    exit 1
fi

# Setup directories
BASE_DIR="/logs"
ARTIFACT_DIR="${BASE_DIR}/artifacts"
mkdir -p "${ARTIFACT_DIR}"

# Increase file descriptor limit for high concurrency
ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Increase aiperf HTTP timeout to avoid ReadTimeout during tokenizer downloads
export AIPERF_HTTP_SO_RCVTIMEO=120

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
    fi
fi

echo "=============================================="
echo "Trace Replay Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Trace file: ${TRACE_FILE}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "=============================================="

# Install aiperf if not present
echo "Installing aiperf..."
pip install "aiperf @ git+https://github.com/ai-dynamo/aiperf.git@8db6e96e978720fc597293bd84e705aa82db233e"  
pip install tiktoken

# Get trace file stats
TRACE_LINES=$(wc -l < "${TRACE_FILE}")
echo "Trace contains ${TRACE_LINES} requests"

# Run small benchmark for warmup
echo ""
echo "Running warmup benchmark..."
aiperf profile \
    -m "${MODEL_NAME}" \
    --tokenizer "/model/" \
    --tokenizer-trust-remote-code \
    --url "${ENDPOINT}" \
    --streaming \
    --ui simple \
    --concurrency 10 \
    --request-count 20
echo "Warmup complete"

# Setup artifact directory with model and timestamp
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TRACE_BASE_NAME="$(basename "${TRACE_FILE}" .jsonl)"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_${TRACE_BASE_NAME}_${TIMESTAMP}"
mkdir -p "${RUN_ARTIFACT_DIR}"

echo ""
echo "Running trace replay benchmark..."
echo "Input file: ${TRACE_FILE}"
echo "Artifact dir: ${RUN_ARTIFACT_DIR}"
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark"

# Run aiperf profile with fixed-schedule to replay at original timestamps
# Use /model/ as tokenizer path (mounted model directory) to avoid HuggingFace download
for concurrency in 1 5 25 50; do
    aiperf profile \
        -m "${MODEL_NAME}" \
        --tokenizer "${MODEL_NAME}" \
        --tokenizer-trust-remote-code \
        --url "${ENDPOINT}" \
        --streaming \
        --input-file "${TRACE_FILE}" \
        --custom-dataset-type mooncake_trace \
        --prompt-corpus coding \
        --concurrency "${concurrency}" \
        --concurrency-ramp-duration 60 \
        --benchmark-duration 1200 \
        --benchmark-grace-period 60 \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --record-processors 8 \
        --profile-export-level raw \
        --export-http-trace \
        --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}" \
        --ui dashboard \
        --artifact-dir "${RUN_ARTIFACT_DIR}/concurrency_${concurrency}"
    done

BENCH_EXIT_CODE=$?

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Benchmark complete (exit code: ${BENCH_EXIT_CODE})"
echo ""
echo "=============================================="
echo "Trace Replay Benchmark Complete"
echo "Results saved to: ${RUN_ARTIFACT_DIR}"
echo "=============================================="

# List artifacts
ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true

# Print results summary
SCRIPT_DIR="$(dirname "$0")"
python3 "${SCRIPT_DIR}/print_results.py" "${RUN_ARTIFACT_DIR}" || true

exit $BENCH_EXIT_CODE
