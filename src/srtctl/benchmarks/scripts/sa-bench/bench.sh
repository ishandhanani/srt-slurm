#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SA-Bench: Throughput/latency benchmark
# Expects: endpoint isl osl concurrencies req_rate model_name is_disaggregated total_gpus prefill_gpus decode_gpus skip_initial_test random_range_ratio

set -e

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}
MODEL_PATH=${6:-/model/}
MODEL_NAME=${7:-"model"}
IS_DISAGGREGATED=${8:-false}
TOTAL_GPUS=${9:-0}
PREFILL_GPUS=${10:-0}
DECODE_GPUS=${11:-0}
SKIP_INITIAL_TEST=${12:-false}
RANDOM_RANGE_RATIO=${13:-0.8}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

WORK_DIR="$(dirname "$0")"

echo "SA-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}; model=${MODEL_NAME}; skip_initial_test=${SKIP_INITIAL_TEST}; random_range_ratio=${RANDOM_RANGE_RATIO}"

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

# Prepare extra args for benchmark_serving.py
EXTRA_ARGS=""
if [ "$SKIP_INITIAL_TEST" = "true" ]; then
    EXTRA_ARGS="--no-test-input"
    echo "Skip initial test enabled - will pass --no-test-input to benchmark_serving.py"
fi

# Quick curl to verify endpoint is working (with retry loop for worker initialization)
# Skip this for GEN-only benchmarks where single requests would deadlock
if [ "$SKIP_INITIAL_TEST" != "true" ]; then
    echo "Verifying endpoint..."
    MAX_RETRIES=60
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s --connect-timeout 5 --max-time 10 "${ENDPOINT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"stream\": false, \"max_tokens\": 10}" 2>/dev/null | head -c 200; then
            echo ""
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "ERROR: Endpoint verification failed after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "Waiting for endpoint... ($i/$MAX_RETRIES)"
        sleep 5
    done
else
    echo "Skipping endpoint verification (skip_initial_test=true)"
fi

ulimit -n 65536 2>/dev/null || true  # May fail in containers without CAP_SYS_RESOURCE

# Benchmark
result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"

for concurrency in "${CONCURRENCY_LIST[@]}"; do

    num_warmup_prompts=$((concurrency * 2))
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_warmup_prompts" \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --ignore-eos \
        --request-rate 250 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        $EXTRA_ARGS

    num_prompts=$((concurrency * 10))
    
    # Generate result filename based on mode
    if [ "$IS_DISAGGREGATED" = "true" ]; then
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    else
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}.json"
    fi
    
    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio $RANDOM_RANGE_RATIO \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --use-chat-template \
        --save-result --result-dir "$result_dir" --result-filename "$result_filename" \
        $EXTRA_ARGS
    
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

echo "SA-Bench complete. Results in $result_dir"

