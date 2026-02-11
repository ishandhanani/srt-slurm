# Rate-Matching Integration Worklog

## Methodology Adaptation: trtllm-bench → SA-bench

### Overview
The rate-matching methodology was originally implemented using `trtllm-bench` for CTX-only and custom SLURM scripts for GEN-only. To make this engine-agnostic in srt-slurm, we need to adapt the approach to use SA-bench.

---

## Required Changes by Section

### Section 1: CTX-only Measurement

**Original (rate-matching repo):**
- Tool: `trtllm-bench --model_path <path> throughput`
- Mode: Server-side saturation (no `--concurrency` flag)
- Config: `OSL=1`, `disable_overlap_scheduler: true`
- Metric: `avg_request_throughput_req_s` (requests/second)

**Adaptation for srt-slurm:**
- Tool: SA-bench (engine-agnostic)
- Challenge: **SA-bench requires a serving endpoint** - disaggregated mode needs both prefill AND decode workers for `/v1/completions` to work
- Status: **BLOCKED** - Need alternative approach (see CTX-only Options below)

**Parameter Changes:**
| Parameter | trtllm-bench | SA-bench Equivalent |
|-----------|--------------|---------------------|
| `OSL=1` | CLI arg | `benchmark.osl: 1` |
| `--concurrency` (none = saturate) | Not used | `concurrencies: "high"` + `req_rate: inf` |
| `num_requests` | `max_batch * tp * 100` | SA-bench handles internally |
| `disable_overlap_scheduler` | `extra_llm_api` | `trtllm_config.prefill.disable_overlap_scheduler: true` |

### Section 2: GEN-only Measurement

**Original (rate-matching repo):**
- Mode: Normal disaggregated (both CTX + GEN workers running)
- Env vars: `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1`, `TLLM_BENCHMARK_REQ_QUEUES_SIZE=${concurrency}`
- Post-processing: Filter per-iteration logs for `num_ctx_tokens == 0`
- Metric: `avg_step_time_ms` → `interactivity = 1000 / step_time`

**Adaptation for srt-slurm:**
- Mode: Same - normal disaggregated
- Env vars: Set via `decode_environment` in YAML
- Post-processing: Same log filtering approach
- Status: **IN PROGRESS** - Job 11057 running

**Parameter Changes:**
| Parameter | rate-matching | srt-slurm Equivalent |
|-----------|---------------|----------------------|
| `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1` | SLURM script | `backend.decode_environment.TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP: "1"` |
| `TLLM_BENCHMARK_REQ_QUEUES_SIZE` | SLURM script | `backend.decode_environment.TLLM_BENCHMARK_REQ_QUEUES_SIZE: "32"` |
| `print_iter_log: true` | YAML config | `trtllm_config.decode.print_iter_log: true` |
| `stream_interval: 20` | YAML config | `trtllm_config.decode.stream_interval: 20` |

**Code Changes Required:**
- `process_gen_results.py`: Port log parsing from `process_gen_iterlog_withctx.py`
  - Log file pattern: `*_decode_w*.out` (was `gen_only*.txt`)
  - Filter: `num_ctx_tokens == 0`
  - Trim: `df.iloc[50:-10]` (skip first 50, last 10)
  - Batch filter: `num_scheduled_requests == concurrency`

### Section 3: Rate Matching Calculation

**Original (rate-matching repo):**
```python
throughput_per_user = 1 / elapsed_time_avg * mtp_accept_rate[mtp_num]
output_throughput = throughput_per_user * concurrency
gen_req_rate = output_throughput / (osl * avg_random_ratio)
ctx_gen_inst_ratio = gen_req_rate / ctx_request_rate
output_tput_per_gpu = output_throughput / (ctx_gpus * ctx_gen_inst_ratio + gen_gpus)
```

**Adaptation for srt-slurm:**
- Same calculation logic, different input sources
- CTX req_rate: From CTX-only measurement (TBD how to obtain)
- GEN metrics: From post-processed per-iteration logs

### Section 4: E2E Config Generation

**Original (rate-matching repo):**
- Output: Custom YAML format for their SLURM scripts

**Adaptation for srt-slurm:**
- Output: srt-slurm compatible YAML ready for `srtctl apply`
- Must include: `resources`, `backend.trtllm_config`, `benchmark` sections
- Add `_sol_metadata` (as comments, not YAML fields) for tracking

---

## CTX-only Measurement Options (Deep Dive)

### Problem Statement
SA-bench uses the `/v1/completions` endpoint which requires **both prefill AND decode workers** in disaggregated mode. With only prefill workers (`decode_workers: 0`), the Dynamo frontend returns 404.

### Option A: Aggregated Mode with OSL=1
**Approach:** Use non-disaggregated mode (`agg_nodes: 1`) with `osl: 1` to minimize decode time.

```yaml
resources:
  agg_nodes: 1
  agg_workers: 1
  gpus_per_node: 8

benchmark:
  type: "sa-bench"
  isl: 8192
  osl: 1
```

**Pros:**
- Works with existing SA-bench
- Engine-agnostic

**Cons:**
- Not true CTX-only isolation (still has minimal decode)
- Different model loading path than disaggregated
- TTFT includes some decode overhead

**Validity:** ~90% valid - The `osl: 1` means only 1 decode step, so TTFT ≈ prefill time. But there's still router/scheduling overhead from the aggregated path.

### Option B: Modify SA-bench to Support Prefill-only Endpoint
**Approach:** Add a new endpoint or mode to SA-bench that only measures prefill.

**Required Changes:**
1. Add `/v1/prefill` or `/v1/ttft` endpoint to SA-bench
2. Modify Dynamo/TRT-LLM to expose prefill-only endpoint
3. Or use `max_tokens: 0` if server supports it

**Pros:**
- True CTX-only isolation
- Engine-agnostic (if other engines also support the endpoint)

**Cons:**
- Requires server-side changes
- Not all engines may support prefill-only endpoint

### Option C: Extract TTFT from Disaggregated Run
**Approach:** Run normal disaggregated with `osl: 1`, extract TTFT metric separately.

```yaml
resources:
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 1    # Need at least 1 for endpoint to work
  decode_workers: 1

benchmark:
  type: "sa-bench"
  isl: 8192
  osl: 1            # Minimal decode
```

**Pros:**
- Works with existing infrastructure
- TTFT metric isolates prefill time

**Cons:**
- Uses extra GPUs for decode worker (wasteful)
- Some overlap between prefill and decode may affect timing

**Validity:** ~95% valid - SA-bench measures TTFT from first token which is pure prefill time. The decode worker handles the single output token but TTFT is measured before that.

### Option D: Per-iteration Log Parsing (Like GEN-only)
**Approach:** Run disaggregated with `osl: 1`, parse prefill worker per-iteration logs for `num_ctx_tokens > 0` iterations.

**Pros:**
- True CTX-only metrics from prefill worker
- Consistent methodology with GEN-only approach

**Cons:**
- Requires same log parsing infrastructure
- Need to validate log format contains CTX throughput metrics

### Recommended Approach: Option C + D Hybrid
1. Run disaggregated with minimal decode (`decode_workers: 1`, `osl: 1`)
2. Extract TTFT from SA-bench output (Option C)
3. Parse prefill worker per-iteration logs for detailed CTX metrics (Option D)
4. Validate both approaches give consistent results

This maintains engine-agnosticism while providing CTX-only isolation through metrics/log filtering.

---

## 2026-02-11: Initial POC Setup

### Goals
Validate rate-matching methodology can be replicated in srt-slurm using SA-bench instead of trtllm-bench.

### Validation Targets (from verified H200 8k/1k data)
- **CTX-only**: `req_rate = 2.94` (Xianjie target: 3.01, 97.7% match)
- **GEN-only TEP c32**: `interactivity = 38.31`, `throughput_per_gpu = 153.24`
- **E2E validation**: Results should be ~10% worse than SOL predictions (expected behavior)

### Completed
1. Created branch `nlevin/rate-matching` in srt-slurm-rate-matching repo
2. Created directory structure:
   - `tools/rate_matching/` - Scripts for rate-matching pipeline
   - `recipes/trtllm/h200/rate_matching/` - Benchmark configs
3. Created `ctx_only_8k1k.yaml` - CTX-only benchmark config
   - `decode_nodes: 0, decode_workers: 0` to run prefill only
   - `osl: 1` to isolate prefill time
   - Config aligned with rate-matching repo verified settings
4. Created `gen_only_8k1k_tep_c32.yaml` - GEN-only benchmark config
   - Normal disaggregated setup (prefill + decode workers)
   - `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP: "1"` for decode isolation
   - `TLLM_BENCHMARK_REQ_QUEUES_SIZE: "32"` matching concurrency
   - `stream_interval: 20` for decode (vs 1 for prefill)
   - `print_iter_log: true` to enable per-iteration logging for post-processing
5. Set up venv and installed srt-slurm dependencies
6. Ran `make setup ARCH=x86_64` to download NATS/ETCD binaries

### Jobs Submitted
- **Job 11056**: CTX-only validation (`ctx_only_8k1k.yaml`) - RUNNING on worker-13
- **Job 11057**: GEN-only TEP c32 validation (`gen_only_8k1k_tep_c32.yaml`) - RUNNING on worker-[4,6]

### Config Alignment Notes

#### CTX-only config alignment with rate-matching repo:
| Parameter | Rate-matching | srt-slurm config | Status |
|-----------|--------------|------------------|--------|
| `disable_overlap_scheduler` | `true` | `true` | ✅ |
| `print_iter_log` | `true` | `true` | ✅ |
| `cuda_graph_config` | `null` | `null` | ✅ |
| `stream_interval` | 1 | `1` | ✅ |
| `max_batch_size` | 2 | `2` | ✅ |
| `max_num_tokens` | 16896 | `16896` | ✅ |
| `free_gpu_memory_fraction` | 0.85 | `0.85` | ✅ |
| `moe_config.backend` | CUTEDSL | CUTLASS | ⚠️ Different but functional |

#### GEN-only config alignment:
| Parameter | Rate-matching | srt-slurm config | Status |
|-----------|--------------|------------------|--------|
| `disable_overlap_scheduler` | false | not set (defaults false) | ✅ |
| `print_iter_log` | `true` | `true` | ✅ |
| `stream_interval` | 20 | `20` | ✅ |
| `max_batch_size` | 128 | `128` | ✅ |
| `max_num_tokens` | 128 | `128` | ✅ |
| `enable_attention_dp` | false (TEP) | `false` | ✅ |
| `use_low_precision_moe_combine` | `true` | `true` | ✅ |
| Env: `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP` | `1` | `1` | ✅ |
| Env: `TLLM_BENCHMARK_REQ_QUEUES_SIZE` | concurrency | `32` | ✅ |

### Findings

#### CTX-only Job (11056) - FAILED
**Issue**: SA-bench cannot run against CTX-only endpoint (no decode workers).
- Server started correctly: `Have 1 prefills and 0 decodes`
- Benchmark failed: `Error: Not Found` on `/v1/completions` endpoint
- **Root cause**: Disaggregated serving requires both prefill AND decode workers for the completion endpoint to work

**See CTX-only Measurement Options section above for detailed analysis.**

#### GEN-only Job (11057) - CANCELLED (hung)
**Issue**: SA-bench hung at "Verifying endpoint..." for 12+ minutes
- Server started correctly: `Have 1 prefills and 1 decodes`
- Frontend logs show: `Completions is ready`, `Prefill router activated`
- Benchmark script started but hung on endpoint verification (curl check)
- Per-iteration logs WERE present in `worker-4_decode_w0.out` with correct format!

**Positive finding**: The per-iteration log format is compatible:
```
iter = 0, global_rank = 0, rank = 0, currank_total_requests = 0/0, host_step_time = 10000.38ms, ... num_scheduled_requests: 1, states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 1}
```

**Next steps**:
1. Debug why SA-bench curl verification is hanging
2. Try running benchmark manually inside container
3. Check if this is a networking/localhost issue

### Next Steps
1. ~~Wait for jobs to complete~~
2. ~~Verify CTX-only job ran with only prefill workers (no decode)~~ - DONE but benchmark failed
3. Verify GEN-only logs contain per-iteration data with `num_ctx_tokens` field
4. Decide on CTX-only approach (trtllm-bench vs skip for POC)
5. Create `process_gen_results.py` with filtering methodology:
   - Filter for `num_ctx_tokens == 0` (pure decode iterations)
   - Skip first 50 iterations, last 10 (`df.iloc[50:-10]`)
   - Filter by target batch: `num_scheduled_requests == 32`
6. Compare results to validation targets
7. Run E2E with existing TEP c32 recipe, confirm ~10% worse than SOL

### Key Methodology Insight
The rate-matching "gen_only" mode does NOT skip prefill workers. Instead:
1. Runs normal disaggregated benchmark (both prefill + decode workers)
2. Sets isolation env vars (`TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1`)
3. Post-processes logs to filter for iterations where `num_ctx_tokens == 0`

This is simpler than expected - no synthetic KV cache injection needed.
