#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPQA evaluation script using sglang.test.run_eval with gpqa

head_node="localhost"
head_port=8000
model_name="deepseek-ai/DeepSeek-R1"  # Default model name

# Parse arguments from SLURM job
n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
num_examples=${5:-198}  # Default: 198
max_tokens=${6:-32768}    # Default: 32768
repeat=${7:-8}          # Default: 8
num_threads=${8:-128}   # Default: 128
# Note: --thinking-mode removed because dynamo frontend doesn't support chat_template_kwargs

echo "GPQA Benchmark Config: num_examples=${num_examples}; max_tokens=${max_tokens}; repeat=${repeat}; num_threads=${num_threads}"

# Source utilities for wait_for_model
source /scripts/utils/benchmark_utils.sh

wait_for_model_timeout=1500 # 25 minutes
wait_for_model_check_interval=5 # check interval -> 5s
wait_for_model_report_interval=60 # wait_for_model report interval -> 60s

wait_for_model $head_node $head_port $n_prefill $n_decode $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval

# Create results directory
result_dir="/logs/accuracy"
mkdir -p $result_dir

echo "Running GPQA evaluation..."

# Set OPENAI_API_KEY if not set
if [ -z "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="EMPTY"
fi

# Run the evaluation
python3 -m sglang.test.run_eval \
    --base-url "http://${head_node}:${head_port}" \
    --model ${model_name} \
    --eval-name gpqa \
    --num-examples ${num_examples} \
    --max-tokens ${max_tokens} \
    --repeat ${repeat} \
    --num-threads ${num_threads}

# Copy the result file from /tmp to our logs directory
# The result file is named gpqa_{model_name}.json
result_file=$(ls -t /tmp/gpqa_*.json 2>/dev/null | head -n1)

if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename $result_file)"
else
    echo "Warning: Could not find result file in /tmp"
fi

echo "GPQA evaluation complete"
