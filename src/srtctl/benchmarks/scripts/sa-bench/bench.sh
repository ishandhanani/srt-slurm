#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SA-Bench: Throughput/latency benchmark
# Expects: endpoint isl osl concurrencies [req_rate]

set -e

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}
MODEL_PATH=${6:-/model/}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

MODEL_NAME="deepseek-ai/DeepSeek-R1"
WORK_DIR="$(dirname "$0")"

echo "SA-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}"

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

# Warmup
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    echo "Warming up with concurrency $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    num_prompts=$((concurrency * 5))
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
        --request-rate 250 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency"
done

# Benchmark
result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$result_dir"

for concurrency in "${CONCURRENCY_LIST[@]}"; do
    num_prompts=$((concurrency * 5))
    result_filename="isl_${ISL}_osl_${OSL}_concurrency_${concurrency}_req_rate_${REQ_RATE}.json"
    
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
        --random-range-ratio 0.8 \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --save-result --result-dir "$result_dir" --result-filename "$result_filename"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

echo "SA-Bench complete. Results in $result_dir"

