#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SA-Bench: Throughput/latency benchmark
# Expects: endpoint isl osl concurrencies [req_rate]
#
# Optional profiling (via worker profiling endpoints):
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling (or "none" to disable)
#   PROFILE_OUTPUT_DIR: Directory inside the container to save profiler output (e.g., /logs/profiles)
#   WORKER_PORT: Default port to use when an endpoint is provided as IP only (defaults to 9090)
#
# Worker targets (prefer *_ENDPOINTS; *_IPS is supported for backward-compat):
#   PROFILE_PREFILL_ENDPOINTS: Comma-separated list of prefill worker endpoints (ip:port or ip)
#   PROFILE_DECODE_ENDPOINTS: Comma-separated list of decode worker endpoints (ip:port or ip)
#   PROFILE_AGG_ENDPOINTS: Comma-separated list of aggregated worker endpoints (ip:port or ip)
#   PROFILE_PREFILL_IPS / PROFILE_DECODE_IPS / PROFILE_AGG_IPS: Comma-separated IPs (uses WORKER_PORT)
#
# Step ranges (stop_step is exclusive; num_steps = stop_step - start_step):
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP
#   PROFILE_AGG_START_STEP / PROFILE_AGG_STOP_STEP

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

MODEL_NAME="deepseek-ai/DeepSeek-R1"
MODEL_PATH="/model/"
WORK_DIR="$(dirname "$0")"

echo "SA-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}"

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

# Quick curl to verify endpoint is working
echo "Verifying endpoint..."
curl -s "${ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 10
    }' | head -c 200
echo ""

# Start profiling before benchmark
start_all_profiling

# Warmup
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    echo "Warming up with concurrency $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    num_prompts=$((concurrency * 5))
    set -x
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio 1 \
        --ignore-eos \
        --request-rate 250 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency"
    set +x
done

# Benchmark
result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"

for concurrency in "${CONCURRENCY_LIST[@]}"; do
    num_prompts=$((concurrency * 5))
    result_filename="isl_${ISL}_osl_${OSL}_concurrency_${concurrency}_req_rate_${REQ_RATE}.json"
    
    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    set -x
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --save-result --result-dir "$result_dir" --result-filename "$result_filename"
    set +x

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

stop_all_profiling

echo "SA-Bench complete. Results in $result_dir"
