#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aiperf: Throughput/latency benchmark using the aiperf profiler
# Expects: endpoint isl osl concurrencies req_rate model_path model_name is_disaggregated total_gpus prefill_gpus decode_gpus isl_stddev osl_stddev request_count

set -e

# Ensure Python output is unbuffered for real-time logging
export PYTHONUNBUFFERED=1

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-}  # Empty means closed-loop mode
MODEL_PATH=${6:-/model/}
MODEL_NAME=${7:-"model"}

# Override MODEL_PATH to use container mount path
# (the passed path is the host lustre path which doesn't exist inside the container)
MODEL_PATH="/model/"
IS_DISAGGREGATED=${8:-false}
TOTAL_GPUS=${9:-0}
PREFILL_GPUS=${10:-0}
DECODE_GPUS=${11:-0}
ISL_STDDEV=${12:-0}
OSL_STDDEV=${13:-0}
REQUEST_COUNT_OVERRIDE=${14:-}  # Optional: explicit request count or empty for default formula

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
    fi
fi

# Parse concurrency list (x-separated)
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

echo "Aiperf Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; isl_stddev=${ISL_STDDEV}; osl_stddev=${OSL_STDDEV}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE:-closed-loop}; request_count=${REQUEST_COUNT_OVERRIDE:-auto}; model=${MODEL_NAME}"

# Wait for model to be ready
wait_for_model_ready() {
    echo "Waiting for model '${MODEL_NAME}' at ${ENDPOINT}/v1/models (checking every 5s)..."
    while ! curl -s "${ENDPOINT}/v1/models" | jq -e --arg model "$MODEL_NAME" '.data[]? | select(.id == $model)' >/dev/null 2>&1; do
        echo "[$(date '+%H:%M:%S')] Model not ready yet, sleeping 5s before checking again ${ENDPOINT}/v1/models"
        sleep 5
    done
    echo "Model '${MODEL_NAME}' is now available!"
    curl -s "${ENDPOINT}/v1/models" | jq .
}

SCRIPT_DIR="$(dirname "$0")"

# Run aiperf benchmark for a given concurrency
run_perf() {
    local concurrency=$1
    local isl=$2
    local osl=$3
    local result_dir=$4
    local isl_stddev=$5
    local osl_stddev=$6
    local request_count_override=$7
    
    local key="concurrency_${concurrency}"
    local artifact_dir="${result_dir}/${key}"
    mkdir -p "$artifact_dir"
    
    # Scale request counts based on concurrency (can be overridden via config)
    local warmup_count=$((concurrency * 1))
    local request_count
    if [ -n "${request_count_override}" ]; then
        # Evaluate the request count override (can be a number or expression like "concurrency * 30")
        request_count=$(python3 -c "concurrency=${concurrency}; print(int(${request_count_override}))")
    else
        request_count=$((concurrency * 2))
    fi
    
    echo "Running aiperf: concurrency=$concurrency, isl=$isl, osl=$osl, isl_stddev=$isl_stddev, osl_stddev=$osl_stddev, warmup=$warmup_count, requests=$request_count"
    echo "Artifact dir: $artifact_dir"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    
    # Build request rate args if specified (open-loop mode)
    # If req_rate is empty, run in closed-loop mode (no --request-rate arg)
    REQUEST_RATE_ARGS=""
    if [ -n "${REQ_RATE}" ]; then
        REQUEST_RATE_ARGS="--request-rate ${REQ_RATE} --request-rate-mode poisson"
    fi
    
    aiperf profile --artifact-dir "$artifact_dir" \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_PATH" \
        --tokenizer-trust-remote-code \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "$ENDPOINT" \
        --synthetic-input-tokens-mean "$isl" \
        --synthetic-input-tokens-stddev "$isl_stddev" \
        --output-tokens-mean "$osl" \
        --output-tokens-stddev "$osl_stddev" \
        --extra-inputs "max_tokens:$osl" \
        --extra-inputs "min_tokens:$osl" \
        --extra-inputs "ignore_eos:true" \
        --extra-inputs '{"nvext":{"ignore_eos":true}}' \
        --concurrency "$concurrency" \
        ${REQUEST_RATE_ARGS} \
        --request-count "$request_count" \
        --warmup-request-count "$warmup_count" \
        --random-seed 42 \
        --workers-max 200 \
        --request-timeout-seconds 1000 \
        --profile-export-level records \
        -H 'Authorization: Bearer NOT USED' \
        -H 'Accept: text/event-stream' \
        --record-processors 8 \
        "${SERVER_METRICS_ARGS[@]}" \
        --ui dashboard
    
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    
    # Print results summary table
    python3 "${SCRIPT_DIR}/print_results.py" "$artifact_dir"
    
    ls -la "$artifact_dir"
    echo "-----------------------------------------"
}

# Increase file descriptor limit (try higher first, fall back to lower)
ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Wait for model to be ready
wait_for_model_ready

# Setup result directory
EPOCH=$(date +%s)
result_dir="/logs/aiperf_isl_${ISL}_osl_${OSL}_${EPOCH}"
mkdir -p "$result_dir"

# Write input config for reference
cat > "${result_dir}/input_config.json" <<EOF
{
    "gpu_count": ${TOTAL_GPUS},
    "concurrencies": "${CONCURRENCIES}",
    "isl": ${ISL},
    "osl": ${OSL},
    "isl_stddev": ${ISL_STDDEV},
    "osl_stddev": ${OSL_STDDEV},
    "request_count_override": "${REQUEST_COUNT_OVERRIDE:-auto}",
    "endpoint": "${ENDPOINT}",
    "model": "${MODEL_NAME}",
    "model_path": "${MODEL_PATH}",
    "is_disaggregated": ${IS_DISAGGREGATED},
    "prefill_gpus": ${PREFILL_GPUS},
    "decode_gpus": ${DECODE_GPUS},
    "request_rate": "${REQ_RATE:-closed-loop}"
}
EOF

# Run benchmark for each concurrency level
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    run_perf "$concurrency" "$ISL" "$OSL" "$result_dir" "$ISL_STDDEV" "$OSL_STDDEV" "$REQUEST_COUNT_OVERRIDE"
done

echo "Aiperf benchmark complete. Results in $result_dir"
