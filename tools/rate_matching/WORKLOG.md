# Rate-Matching Integration Worklog

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

### Next Steps
1. Wait for jobs to complete
2. Verify CTX-only job ran with only prefill workers (no decode)
3. Verify GEN-only logs contain per-iteration data with `num_ctx_tokens` field
4. Create `process_ctx_results.py` to parse CTX logs
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
