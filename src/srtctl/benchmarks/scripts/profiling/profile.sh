#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Profiling script for sglang/trtllm workers
# For torch profiling: Sends /start_profile API calls to workers
# For nsys profiling: nsys handles capture via time-based delay (no API calls needed)
#
# NOTE: The orchestrator (do_sweep.py) already waits for all workers to be healthy
# before running this script, so we don't need to wait here.

model_name="${PROFILE_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"
head_node="${HEAD_NODE:-127.0.0.1}"
head_port="${HEAD_PORT:-8000}"

# Parse arguments
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
total_gpus=$5

# Configurable num_prompts (default: 128)
num_prompts="${PROFILE_NUM_PROMPTS:-128}"

# Profiler type: "nsys" or "torch" (default: torch for backward compat)
profiler_type="${PROFILER_TYPE:-torch}"

# Backend type for benchmark_serving.py
backend_type="${BACKEND_TYPE:-dynamo}"

echo "Profiling Configuration:"
echo "  Profiler type: ${profiler_type}"
echo "  Backend type: ${backend_type}"
echo "  Profiling mode: ${PROFILING_MODE}"
echo "  Profiling dir: ${SGLANG_TORCH_PROFILER_DIR}"
echo "  Prefill workers: ${n_prefill}"
echo "  Decode workers: ${n_decode}"
echo "  Prefill GPUs: ${prefill_gpus}"
echo "  Decode GPUs: ${decode_gpus}"
echo "  Total GPUs: ${total_gpus}"
echo "  ISL: ${PROFILE_ISL}"
echo "  OSL: ${PROFILE_OSL}"
echo "  Concurrency: ${PROFILE_CONCURRENCY}"
echo "  Num prompts: ${num_prompts}"

# Validate required parameters
if [[ -z "${PROFILE_ISL}" || -z "${PROFILE_OSL}" ]]; then
    echo "Error: PROFILE_ISL and PROFILE_OSL must be set"
    exit 1
fi
if [[ -z "${PROFILE_CONCURRENCY}" ]]; then
    echo "Error: PROFILE_CONCURRENCY must be set"
    exit 1
fi

# Parse leader IP lists from environment (comma-separated)
IFS=',' read -r -a PREFILL_IPS <<< "${PROFILE_PREFILL_IPS:-}"
IFS=',' read -r -a DECODE_IPS <<< "${PROFILE_DECODE_IPS:-}"
IFS=',' read -r -a AGG_IPS <<< "${PROFILE_AGG_IPS:-}"

# Get phase-specific start/stop steps
get_phase_start_step() {
    local phase="$1"
    local var_name="PROFILE_${phase}_START_STEP"
    echo "${!var_name:-0}"
}

get_phase_stop_step() {
    local phase="$1"
    local var_name="PROFILE_${phase}_STOP_STEP"
    echo "${!var_name:-50}"
}

# Start profiling on a worker (only for torch profiler, not nsys)
start_profile_on_worker() {
    local ip="$1"
    local start_step="$2"
    local stop_step="$3"
    
    if [[ -z "${ip}" ]]; then
        return
    fi
    
    local num_steps=$((stop_step - start_step))
    if [[ "${num_steps}" -le 0 ]]; then
        echo "Error: invalid step range: start=${start_step} stop=${stop_step}"
        return 1
    fi
    
    # Determine activities based on profiler type
    local ACTIVITIES
    if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
        ACTIVITIES='["CPU", "GPU", "MEM"]'
    else
        ACTIVITIES='["CUDA_PROFILER"]'
    fi
    
    echo "Starting profiling on http://${ip}:30000 (steps ${start_step}-${stop_step})"
    curl -sS -X POST "http://${ip}:30000/start_profile" \
        -H "Content-Type: application/json" \
        -d "{\"start_step\": ${start_step}, \"num_steps\": ${num_steps}, \"activities\": ${ACTIVITIES}}" || true
}

# Check if we have any workers to profile
if [[ "${#PREFILL_IPS[@]}" -eq 0 && "${#DECODE_IPS[@]}" -eq 0 && "${#AGG_IPS[@]}" -eq 0 ]]; then
    echo "Error: No worker IPs provided for profiling"
    echo "Set PROFILE_PREFILL_IPS, PROFILE_DECODE_IPS, or PROFILE_AGG_IPS"
    exit 1
fi

# Create profiling output directory
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    mkdir -p "${SGLANG_TORCH_PROFILER_DIR}" 2>/dev/null || true
fi

echo ""
echo "Starting profiling..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

set -x

# Get phase-specific steps
prefill_start=$(get_phase_start_step PREFILL)
prefill_stop=$(get_phase_stop_step PREFILL)
decode_start=$(get_phase_start_step DECODE)
decode_stop=$(get_phase_stop_step DECODE)
agg_start=$(get_phase_start_step AGG)
agg_stop=$(get_phase_stop_step AGG)

# Start profiling on workers - ONLY for torch profiler
# nsys uses time-based capture (-y delay, -d duration) so no API calls needed
if [[ "${profiler_type}" == "torch" ]]; then
    echo "Torch profiler: calling /start_profile API on workers..."
    for ip in "${PREFILL_IPS[@]}"; do
        start_profile_on_worker "${ip}" "${prefill_start}" "${prefill_stop}"
    done
    for ip in "${DECODE_IPS[@]}"; do
        start_profile_on_worker "${ip}" "${decode_start}" "${decode_stop}"
    done
    for ip in "${AGG_IPS[@]}"; do
        start_profile_on_worker "${ip}" "${agg_start}" "${agg_stop}"
    done
else
    echo "nsys profiler: skipping /start_profile API (using time-based capture)"
fi

# Only the prefill profiling job needs to generate traffic through the router.
if [[ "${PROFILING_MODE}" == "prefill" ]]; then
    # Path to benchmark scripts (mounted at /srtctl-benchmarks)
    BENCH_SCRIPT_DIR="/srtctl-benchmarks/sa-bench"
    
    # =========================================================================
    # PHASE 1: WARMUP (not profiled)
    # Run warmup traffic to compile CUDA graphs and warm KV caches
    # =========================================================================
    echo ""
    echo "=== PHASE 1: WARMUP ==="
    warmup_prompts=$((PROFILE_CONCURRENCY * 2))
    echo "Running ${warmup_prompts} warmup prompts at 250 req/s..."
    
    python3 -u "${BENCH_SCRIPT_DIR}/benchmark_serving.py" \
        --backend "${backend_type}" \
        --model "${model_name}" \
        --tokenizer "/model/" \
        --host "${head_node}" --port "${head_port}" \
        --endpoint /v1/completions \
        --dataset-name random \
        --max-concurrency "${PROFILE_CONCURRENCY}" \
        --num-prompts "${warmup_prompts}" \
        --random-input-len "${PROFILE_ISL}" \
        --random-output-len "${PROFILE_OSL}" \
        --random-range-ratio 1 \
        --request-rate 250 \
        --ignore-eos \
        --disable-tqdm || true
    
    echo "Warmup complete."
    
    # =========================================================================
    # PHASE 2: PROFILING TRAFFIC
    # Generate actual profiled traffic
    # =========================================================================
    echo ""
    echo "=== PHASE 2: PROFILING TRAFFIC ==="
    echo "Generating ${num_prompts} profiling prompts..."
    
    python3 -u "${BENCH_SCRIPT_DIR}/benchmark_serving.py" \
        --backend "${backend_type}" \
        --model "${model_name}" \
        --tokenizer "/model/" \
        --host "${head_node}" --port "${head_port}" \
        --endpoint /v1/completions \
        --dataset-name random \
        --max-concurrency "${PROFILE_CONCURRENCY}" \
        --num-prompts "${num_prompts}" \
        --random-input-len "${PROFILE_ISL}" \
        --random-output-len "${PROFILE_OSL}" \
        --random-range-ratio 1 \
        --ignore-eos \
        --disable-tqdm
fi

exit_code=$?
set +x

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "Profiling completed with exit code ${exit_code}"
if [[ -n "${SGLANG_TORCH_PROFILER_DIR}" ]]; then
    echo "Profiling results saved to ${SGLANG_TORCH_PROFILER_DIR}"
fi

exit ${exit_code}
