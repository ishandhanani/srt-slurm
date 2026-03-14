#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prefill benchmark: trace-replay on together-ai-basic-no-delays_1osl dataset.
# Args: endpoint model_name splits_dir concurrencies(x-sep) total_gpus

ENDPOINT=$1
MODEL_NAME=$2
SPLITS_DIR=$3          # container path, e.g. /prefill-data
CONCURRENCIES=$4       # e.g. "1x10x25x50"
TOTAL_GPUS=${5:-0}

IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

# Timing table: duration_s and ramp_s per concurrency
get_timing() {
    local conc=$1
    case "$conc" in
        1)   echo "120 5"   ;;
        2)   echo "180 10"  ;;
        4)   echo "240 20"  ;;
        8)   echo "360 40"  ;;
        10)  echo "330 50"  ;;
        16)  echo "480 80"  ;;
        25)  echo "480 125" ;;
        32)  echo "540 160" ;;
        50)  echo "600 250" ;;
        64)  echo "600 300" ;;
        *)   # Formula fallback for other values
             local ramp=$(( conc * 5 < 300 ? conc * 5 : 300 ))
             local dur=$(( ramp * 2 + 60 ))
             dur=$(( dur < 120 ? 120 : (dur > 600 ? 600 : dur) ))
             echo "$dur $ramp"
             ;;
    esac
}

ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Warmup: synthetic ISL=1000, OSL=1000, concurrency=4, 12 requests
echo "Running warmup (isl=1000 osl=1000 concurrency=4 count=12)..."
aiperf profile \
    -m "$MODEL_NAME" \
    --tokenizer /model/ \
    --tokenizer-trust-remote-code \
    --url "$ENDPOINT" \
    --streaming \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --synthetic-input-tokens-mean 1000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 1000 \
    --output-tokens-stddev 0 \
    --extra-inputs "max_tokens:1000" \
    --extra-inputs "min_tokens:1000" \
    --extra-inputs "ignore_eos:true" \
    --concurrency 4 \
    --request-count 12 \
    --warmup-request-count 0 \
    --workers-max 200 \
    --request-timeout-seconds 1200 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream' \
    --artifact-dir "/tmp/prefill-warmup"
echo "Warmup complete."

EPOCH=$(date +%s)
RESULT_DIR="/logs/prefill_${EPOCH}"
mkdir -p "$RESULT_DIR"

for conc in "${CONCURRENCY_LIST[@]}"; do
    read -r duration_s ramp_s <<< "$(get_timing "$conc")"
    split_file="${SPLITS_DIR}/conc_${conc}.jsonl"
    artifact_dir="${RESULT_DIR}/concurrency_${conc}"
    mkdir -p "$artifact_dir"

    echo "Concurrency: ${conc}  duration=${duration_s}s  ramp=${ramp_s}s"
    echo "Split: $split_file"

    aiperf profile \
        -m "$MODEL_NAME" \
        --tokenizer /model/ \
        --tokenizer-trust-remote-code \
        --url "$ENDPOINT" \
        --streaming \
        --input-file "$split_file" \
        --custom-dataset-type mooncake_trace \
        --concurrency "$conc" \
        --concurrency-ramp-duration "$ramp_s" \
        --benchmark-duration "$duration_s" \
        --benchmark-grace-period 30 \
        --workers-max 200 \
        --request-timeout-seconds 1200 \
        --record-processors 8 \
        --profile-export-level records \
        -H 'Authorization: Bearer NOT USED' \
        -H 'Accept: text/event-stream' \
        --ui dashboard \
        --artifact-dir "$artifact_dir"

    echo "$(date '+%Y-%m-%d %H:%M:%S') — concurrency ${conc} complete"
done

echo "Prefill benchmark complete. Results in $RESULT_DIR"
