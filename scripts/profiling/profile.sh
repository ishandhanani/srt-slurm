#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Torch profiling script for sglang.launch_server
# This script runs bench_one_batch_server with profiling enabled

model_name="deepseek-ai/DeepSeek-R1"
head_node="${HEAD_NODE:-127.0.0.1}"
head_port="${HEAD_PORT:-8000}"

# Parse arguments (same as sa-bench for consistency)
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$5

echo "Torch Profiling Configuration:"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Profiling dir: ${SGLANG_TORCH_PROFILER_DIR}"
echo "  Prefill workers: ${n_prefill}"
echo "  Decode workers: ${n_decode}"
echo "  Prefill GPUs: ${prefill_gpus}"
echo "  Decode GPUs: ${decode_gpus}"
echo "  Total GPUs: ${total_gpus}"

# Wait for server to be ready using inline wait function
wait_until_ready() {
    local SERVER_URL="$1"
    while true; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" "${SERVER_URL}/health" || echo "000")
        if [ "$status_code" -eq 200 ]; then
            echo "Server ${SERVER_URL} is ready"
            break
        fi
        echo "Server not ready yet (status: ${status_code}), waiting..."
        top -b -n 1 | head -10
        PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | tr -d ' ' | head -n1)
        [ -n "$PID" ] && py-spy dump -s --pid $PID > /logs/py-spy-dump-${SLURM_NODEID:-0}.txt || echo "No GPU process found"
        sleep 30
    done
}

# Parse prefill/decode leader IP lists from environment (comma-separated)
IFS=',' read -r -a PREFILL_IPS <<< "${PROFILE_PREFILL_IPS:-}"
IFS=',' read -r -a DECODE_IPS <<< "${PROFILE_DECODE_IPS:-}"

wait_all_workers_ready() {
    local ips=("$@")
    for ip in "${ips[@]}"; do
        if [[ -z "${ip}" ]]; then
            continue
        fi
        echo "Waiting for worker at http://${ip}:30000 to be ready..."
        wait_until_ready "http://${ip}:30000"
    done
}

# For PD disaggregation, wait for router once and then wait for all worker servers
if [[ "${PROFILING_MODE}" == "prefill" ]]; then
    echo "Waiting for router at http://${head_node}:${head_port} to be ready..."
    wait_until_ready "http://${head_node}:${head_port}"
fi

if [[ "${#PREFILL_IPS[@]}" -gt 0 || "${#DECODE_IPS[@]}" -gt 0 ]]; then
    echo "Waiting for all profiling workers to be ready..."
    wait_all_workers_ready "${PREFILL_IPS[@]}" "${DECODE_IPS[@]}"
else
    # Backward-compatible single-node behavior
    echo "Waiting for local ${PROFILING_MODE} server at http://127.0.0.1:30000 to be ready..."
    wait_until_ready "http://127.0.0.1:30000"
fi

# Determine profiling parameters strictly from environment 
PROFILE_STEPS_ARG=""
CLI_ARGS=""
[[ -n "${PROFILE_CONCURRENCY}" ]] && CLI_ARGS+=" --batch-size ${PROFILE_CONCURRENCY}"
# Require ISL/OSL to be provided; do not pass them as CLI args here
if [[ -z "${PROFILE_ISL}" || -z "${PROFILE_OSL}" ]]; then
    echo "Error: isl and osl must be set for profiling."
    exit 1
fi

# Configure profiling steps range; set defaults independently if missing
if [[ -z "${PROFILE_START_STEP}" ]]; then
    echo "Warning: PROFILE_START_STEP not set; defaulting to 0"
    PROFILE_START_STEP=0
fi
if [[ -z "${PROFILE_STOP_STEP}" ]]; then
    echo "Warning: PROFILE_STOP_STEP not set; defaulting to 50"
    PROFILE_STOP_STEP=50
fi


echo "Running profiler..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Create profiling output directory only when torch profiler dir is provided
ACTIVITIES=""
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    ACTIVITIES='["CPU", "GPU", "MEM"]'
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}" 2>/dev/null || true
    export SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR}
else
    ACTIVITIES='["CUDA_PROFILER"]'
    mkdir -p "/logs/profiles" 2>/dev/null || true
fi

set -x

start_profile_on_worker() {
    local ip="$1"
    if [[ -z "${ip}" ]]; then
        return
    fi
    echo "Starting profiling on http://${ip}:30000"
    curl -X POST "http://${ip}:30000/start_profile" -H "Content-Type: application/json" -d "{\"start_step\": \"$PROFILE_START_STEP\", \"num_steps\": $((PROFILE_STOP_STEP-PROFILE_START_STEP)), \"activities\": $ACTIVITIES}"
}

if [[ "${#PREFILL_IPS[@]}" -gt 0 || "${#DECODE_IPS[@]}" -gt 0 ]]; then
    for ip in "${PREFILL_IPS[@]}"; do
        start_profile_on_worker "${ip}"
    done
    for ip in "${DECODE_IPS[@]}"; do
        start_profile_on_worker "${ip}"
    done
else
    # Fallback to local single-node profiling
    echo "Starting profiling on local server http://127.0.0.1:30000"
    curl -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json" -d "{\"start_step\": \"$PROFILE_START_STEP\", \"num_steps\": $((PROFILE_STOP_STEP-PROFILE_START_STEP)), \"activities\": $ACTIVITIES}"
fi

# Only the prefill profiling job needs to generate traffic through the router.
if [[ "${PROFILING_MODE}" == "prefill" ]]; then
    python3 -m sglang.bench_serving \
    --backend sglang \
    --model ${model_name} \
    --host ${head_node} --port ${head_port} \
    --dataset-name random \
    --max-concurrency $PROFILE_CONCURRENCY \
    --num-prompts 128 \
    --random-input-len $PROFILE_ISL \
    --random-output-len $PROFILE_OSL \
    --random-range-ratio 1 \
    --warmup-request 5

    pip install lm-eval tenacity > /dev/null
    python -m lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args base_url=http://${head_node}:${head_port}/v1/completions,model=${model_name},tokenized_requests=False,tokenizer_backend=None,num_concurrent=${PROFILE_CONCURRENCY},timeout=6000,max_retries=1 \
    --limit 10
fi

exit_code=$?
set +x

echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Torch profiling completed for ${PROFILING_MODE} mode with exit code ${exit_code}"
echo "Profiling results saved to ${SGLANG_TORCH_PROFILER_DIR}"

exit ${exit_code}
