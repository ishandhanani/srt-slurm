# Exemplar Benchmarking Recipes

This directory contains recipes for running aiperf-based benchmarks on the Exemplar (DeepSeek V3) model.

## Benchmark Types

There are two benchmark types available:

### 1. `aiperf` - Synthetic Load Generation

Uses aiperf's internal load generator with user-specified request rates and concurrency levels.

**Config fields:**
- `benchmark.req_rate` - Request rate for open-loop mode (poisson arrival)
- `benchmark.concurrencies` - List of concurrency levels to test (e.g., `['4','16','32','64']`)
- `benchmark.isl` - Input sequence length
- `benchmark.osl` - Output sequence length

**Example:**
```yaml
benchmark:
  type: "aiperf"
  isl: 36000
  osl: 8000
  concurrencies: ['4','16','32','64','128','256']
  req_rate: 100
```

**Code:**
- Runner: `src/srtctl/benchmarks/aiperf.py`
- Script: `src/srtctl/benchmarks/scripts/aiperf/bench.sh`

### 2. `trace-replay` - Trace-Based Load Generation

Replays request traces with fixed timestamps from a mooncake-style JSONL trace file.

**Config fields:**
- `benchmark.trace_file` - Path to the JSONL trace file (relative to workspace root)
- `benchmark.ttft_threshold_ms` - Goodput TTFT threshold in ms (default: 2000)
- `benchmark.itl_threshold_ms` - Goodput ITL threshold in ms (default: 25)

**Example:**
```yaml
benchmark:
  type: "trace-replay"
  trace_file: "traces/conversation_trace_synth_16.00x1+10.00_speedup1_maxisl110000.jsonl"
  ttft_threshold_ms: 20000
  itl_threshold_ms: 50
```

**Code:**
- Runner: `src/srtctl/benchmarks/trace_replay.py`
- Script: `src/srtctl/benchmarks/scripts/trace-replay/bench.sh`

## Recipe Files

Most yamls in this directory use `aiperf` benchmark type. Files with `*_trace.yaml` suffix use `trace-replay`:

| File | Benchmark Type |
|------|----------------|
| `ctx2_dep8_gen2_dep8_batch32_8_nvfp4_router_trace.yaml` | trace-replay |
| `agg4_dep8_batch8_nvfp4_trace.yaml` | trace-replay |
| All other `*.yaml` files | aiperf |

## Using Trace Files

To run trace-based benchmarks:

1. Place your trace files in the `traces/` directory at the repository root
2. Set `benchmark.trace_file` to the relative path, e.g.:
   ```yaml
   benchmark:
     type: "trace-replay"
     trace_file: "traces/your_trace.jsonl"
   ```

The trace directory is automatically mounted into the container.

### Generating Trace Files

Trace files are synthetically generated using Dynamo's prefix data generator tool.

**Tool:** https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/prefix_data_generator 
> [!NOTE]
> Before [PR #6117](https://github.com/ai-dynamo/dynamo/pull/6117) is merged to main, use the `synthesizer.py` in this branch for correct dataset behavior.

**Source trace:** Download `conversation_trace.jsonl` (Original Mooncake Trace Dataset) from [dynamo_exemplar/traces](https://gitlab-master.nvidia.com/jothomson/dynamo_exemplar/-/blob/karenc/dsv3_gb200/traces/)  

**Example generation command:**

The trace file `conversation_trace_synth_16.00x1+10.00_speedup1_maxisl110000.jsonl` used in the `*_trace.yaml` recipes was generated with:

```bash
datagen synthesize \
    --input-file conversation_trace.jsonl \
    --prefix-len-multiplier 16 \
    --prompt-len-multiplier 10 \
    --max-isl 110000 \
    --num-requests 10000
```
This trace and a few others can be found here: [dynamo_exemplar/traces](https://gitlab-master.nvidia.com/jothomson/dynamo_exemplar/-/blob/karenc/dsv3_gb200/traces/conversation_trace_synth_16.00x1+10.00_speedup1_maxisl110000.jsonl?ref_type=heads)  


## TRTLLM Attention DP Support

To use TRTLLM attention data parallelism (necessary for DEP8 configs), use the Dynamo 0.9.0 container with TRTLLM ADP support :

`gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:9590b0130162891b49126fc77a88fe7770e02dd6-43393213-trtllm-arm64` ([original commit](https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/commit/9590b0130162891b49126fc77a88fe7770e02dd6))

### Enroot Setup

For enroot-based clusters, import the container as a squashfs file, e.g.

```bash
enroot import --output dyn_090_with_adp_trtllm.sqsh \
  docker://gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:9590b0130162891b49126fc77a88fe7770e02dd6-43393213-trtllm-arm64
```

Then reference the `.sqsh` file in your `srtslurm.yaml` or recipe:

```yaml
model:
  container: "dyn-090-with-adp-trtllm"  
```

> [!NOTE]
> Dynamo 0.9.0 with TRTLLM ADP is not yet in an official release.
