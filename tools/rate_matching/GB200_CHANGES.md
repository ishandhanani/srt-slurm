# Rate-Matching Tool: GB200 Support & Multi-Chip Generalization

## Overview

The rate-matching tool was originally built for **H200** (8 GPUs/node, x86_64). We extended it to support **GB200** (4 GPUs/node, aarch64) and in the process made it **chip-agnostic** — it now works for any GPU type (H100, H200, B200, B300, GB200, GB300, etc.) with zero code changes, driven entirely by the sweep YAML.

This document summarizes all changes, why they were needed, and how backward compatibility is preserved.

---

## Changes Summary

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `generate_configs.py` | Multi-node decode support | TP=8 on 4 GPUs/node needs 2 nodes |
| 2 | `generate_configs.py` | Removed hardcoded `cpus-per-gpu` | GB200 cluster doesn't support GRES |
| 3 | `generate_configs.py` | Added `gpus_per_prefill` / `gpus_per_decode` | Explicit GPU allocation per worker |
| 4 | `generate_configs.py` | Deep-merge override mechanism | Allow sweep YAML to override TRT-LLM defaults |
| 5 | `schema.py` | Per-group `decode_overrides` / `prefill_overrides` | Different MoE backends per decode mode |
| 6 | `run_sweep.py` | TP size in GEN SOL filenames | Prevent filename collisions (TEP2 vs TEP4) |
| 7 | `run_sweep.py` | Override propagation through pipeline | Carry per-group overrides to E2E configs |

---

## Detailed Changes

### 1. Multi-Node Decode (`generate_configs.py`)

**Problem**: `decode_nodes` was hardcoded to `1`. On GB200 with 4 GPUs/node, TP=8 requires 2 nodes and TP=16 requires 4 nodes.

**Fix**: Added `_nodes_for_tp()` helper:
```python
def _nodes_for_tp(tp_size: int, gpus_per_node: int) -> int:
    return math.ceil(tp_size / gpus_per_node)
```
Applied to `generate_ctx_sol_config`, `generate_gen_sol_config`, and `generate_e2e_config`.

**H200 impact**: `ceil(8/8) = 1` — unchanged behavior.

---

### 2. Removed Hardcoded `cpus-per-gpu` (`generate_configs.py`)

**Problem**: `sbatch_directives: {"cpus-per-gpu": "16"}` was hardcoded. GB200 cluster doesn't support GRES, causing `sbatch: error: Invalid generic resource (gres) specification`.

**Fix**: Set `sbatch_directives: {}` (empty). Since `--exclusive` is used, all CPUs are already allocated.

**H200 impact**: No functional change — `--exclusive` already provides all CPUs.

---

### 3. Explicit `gpus_per_prefill` / `gpus_per_decode` (`generate_configs.py`)

**Problem**: Without these fields, srtctl defaulted to using all GPUs on the node. On GB200 (4 GPUs/node), a TP=1 prefill worker would get 4 GPUs assigned, launching 4 MPI processes → OOM and rank mismatch errors.

**Fix**: Explicitly set `gpus_per_prefill` and `gpus_per_decode` from `cfg.resources.ctx_gpus_per_instance` and `gen_item.tp_size`.

**H200 impact**: Makes the implicit explicit — same values, no behavior change.

---

### 4. 3-Layer Deep-Merge Override Mechanism (`generate_configs.py`, `schema.py`)

**Problem**: TRT-LLM config values like `moe_config.backend`, `nvfp4_gemm_config`, `enable_lm_head_tp_in_adp` were hardcoded to H200 defaults (CUTLASS, no nvfp4, etc.). No way to customize for different models or chips.

**Fix**: Implemented a 3-layer merge system:

```
Per-group overrides  →  highest priority (wins)
Global overrides     →  middle
Tool defaults        →  lowest (base)
```

**How it works**:
1. `generate_configs.py` builds a base config with sensible defaults
2. Global `backend.trtllm_decode_overrides` from sweep YAML merges on top
3. Per-group `decode_overrides` from `gen_sweep` groups merge on top of that

```python
# In _decode_config():
# Layer 1: global overrides
if cfg.backend.trtllm_decode_overrides:
    config = _deep_merge(config, cfg.backend.trtllm_decode_overrides)
# Layer 2: per-item overrides (highest priority)
if gen_item.decode_overrides:
    config = _deep_merge(config, gen_item.decode_overrides)
```

**H200 impact**: If no overrides are specified, tool defaults are used — identical to before.

---

### 5. Per-Group `decode_overrides` / `prefill_overrides` (`schema.py`)

**Problem**: The global `trtllm_decode_overrides` applied the same MoE backend to ALL decode groups. But TEP (small batch) needs `TRTLLM` backend and DEP (large batch) needs `CUTEDSL` — using `CUTEDSL` everywhere caused 7-13% slower TPOT for TEP configs at low concurrency.

**Fix**: Added optional `decode_overrides` and `prefill_overrides` fields to both `GenSweepItem` and `GenSweepGroup`. These are automatically propagated through the pipeline:

```
GenSweepGroup.decode_overrides
  → GenSweepGroup.expand() injects into each GenSweepItem
    → generate_gen_sol_config() applies in _decode_config()
      → gen_result carries overrides
        → rate_matching_result carries overrides
          → Pareto frontier carries overrides
            → generate_e2e_config() applies same overrides
```

**Sweep YAML example**:
```yaml
gen_sweep:
  tep4:
    defaults:
      mode: tep
      tp_size: 4
    decode_overrides:
      moe_config:
        backend: TRTLLM        # optimal for small batch
    parameters:
      concurrency: [1, 4, 16, 32, 64]
      batch_size:  [1, 4, 16, 32, 64]

  dep8:
    defaults:
      mode: dep
      tp_size: 8
    decode_overrides:
      moe_config:
        backend: CUTEDSL       # optimal for large batch
        use_low_precision_moe_combine: true
    parameters:
      concurrency: [512, 1024]
      batch_size:  [64, 128]
```

**H200 impact**: Fields are optional — existing configs without them work identically.

---

### 6. TP Size in GEN SOL Filenames (`run_sweep.py`)

**Problem**: Filename `gen_sol_{mode}_c{conc}.yaml` didn't include TP size. When sweeping TEP TP=2 and TEP TP=4 with the same concurrency (e.g., c=1), TEP2 config was overwritten by TEP4.

**Fix**: Changed to `gen_sol_{mode}{tp_size}_c{conc}.yaml` (e.g., `gen_sol_tep2_c1.yaml`, `gen_sol_tep4_c1.yaml`).

**H200 impact**: Filenames change but are cosmetic — no functional impact.

---

---

## How to Define the Sweep YAML

The sweep YAML is the **single source of truth** for any rate-matching run. It controls the model, workload, hardware, TRT-LLM backend settings, and what decode configs to sweep. Here's how each section works and why it's needed.

### Full Annotated Structure

```yaml
# ┌─────────────────────────────────────────────────────────────────┐
# │  SWEEP YAML STRUCTURE                                          │
# │                                                                 │
# │  name          → sweep identifier                              │
# │  model         → model path, container, precision              │
# │  workload      → ISL/OSL                                       │
# │  resources     → chip type, GPUs per node, TP sizes, budget    │
# │  ctx_config    → prefill SOL benchmark overrides               │
# │  gen_sweep     → decode groups with per-group overrides        │
# │  backend       → environment vars + global TRT-LLM overrides   │
# │  settings      → automation knobs                              │
# └─────────────────────────────────────────────────────────────────┘
```

### 1. Resources — Chip-Specific Hardware

This is where you tell the tool about your GPU cluster:

```yaml
resources:
  gpu_type: gb200          # GPU identifier (gb200, h200, b200, etc.)
  gpus_per_node: 4         # GPUs per SLURM node
                           #   GB200 = 4, H200 = 8, B200 = 8
  ctx_gpus_per_instance: 1 # TP for prefill workers (usually 1 for MoE models)
  gen_gpus_per_instance: 4 # TP for CTX SOL decode stub
  max_total_gpus: 64       # Budget cap for allocation search
```

The tool uses `gpus_per_node` to automatically compute multi-node configs:
- TP=4 on 4 GPUs/node → 1 node per worker
- TP=8 on 4 GPUs/node → 2 nodes per worker (`ceil(8/4)`)
- TP=16 on 4 GPUs/node → 4 nodes per worker (`ceil(16/4)`)

### 2. CTX Config — Prefill SOL Overrides

Override the tool's prefill defaults to match your actual recipe:

```yaml
ctx_config:
  benchmark_concurrency: 32       # High enough to saturate the prefill worker
  max_batch_size: 4               # From your recipe (tool default: 2 for ISL>2048)
  max_num_tokens: 32768           # From your recipe (tool default: 16896)
  free_gpu_memory_fraction: 0.6   # From your recipe (tool default: 0.85)
```

### 3. Gen Sweep — Decode Groups with Per-Group Overrides

Each group defines a decode mode + TP size and sweeps across concurrencies. **Per-group `decode_overrides`** let you set different TRT-LLM backends per mode:

```yaml
gen_sweep:
  # TEP groups: small batch, latency-optimized
  # → Use TRTLLM MoE backend (lower kernel overhead at small batch)
  tep4:
    expansion: zip                   # pairs concurrency[i] with batch_size[i]
    parameters:
      concurrency: [1, 4, 16, 32, 64]
      batch_size:  [1, 4, 16, 32, 64]
      gpu_memory_fraction: [0.9, 0.9, 0.9, 0.9, 0.9]
    defaults:
      mode: tep
      tp_size: 4
    decode_overrides:                # ← PER-GROUP: only applies to this group
      moe_config:
        backend: TRTLLM

  # DEP groups: large batch, throughput-optimized
  # → Use CUTEDSL MoE backend (higher throughput at large batch)
  dep8:
    expansion: zip
    parameters:
      concurrency: [512, 1024]
      batch_size:  [64, 128]
      gpu_memory_fraction: [0.8, 0.8]
    defaults:
      mode: dep
      tp_size: 8
    decode_overrides:                # ← PER-GROUP: only applies to this group
      moe_config:
        backend: CUTEDSL
        use_low_precision_moe_combine: true
```

### 4. Backend — Environment Variables + Global Overrides

The `backend` section has two levels:

**Environment variables** — passed to prefill/decode workers:
```yaml
backend:
  prefill_environment:
    TRTLLM_SERVER_DISABLE_GC: "1"
    TRTLLM_ENABLE_PDL: "1"
    ENROOT_ALLOW_DEV: "1"           # GB200-specific
    NCCL_GRAPH_MIXING_SUPPORT: "0"
  decode_environment:
    TRTLLM_SERVER_DISABLE_GC: "1"
    TRTLLM_ENABLE_PDL: "1"
    ENROOT_ALLOW_DEV: "1"
    NCCL_GRAPH_MIXING_SUPPORT: "0"
```

**Global TRT-LLM overrides** — applied to ALL groups, then per-group overrides merge on top:
```yaml
  # Prefill overrides (same for all groups)
  trtllm_prefill_overrides:
    moe_config:
      backend: TRTLLM
    nvfp4_gemm_config:
      allowed_backends: [cutlass, cublaslt, cutedsl, cuda_core]
    cache_transceiver_config:
      max_tokens_in_buffer: 16384

  # GLOBAL decode overrides — common settings shared by ALL decode groups
  # Mode-specific settings go in per-group decode_overrides (see gen_sweep above)
  trtllm_decode_overrides:
    nvfp4_gemm_config:
      allowed_backends: [cutlass, cublaslt, cutedsl, cuda_core]
    enable_lm_head_tp_in_adp: false
```

### 5. How the Overrides Work

It's simple: **whatever you put in the sweep YAML overrides the tool's built-in defaults**. The tool deep-merges your settings on top of its defaults — your values always win.

- **Global overrides** (`trtllm_prefill_overrides` / `trtllm_decode_overrides`) → apply to ALL groups
- **Per-group overrides** (`decode_overrides` inside a gen_sweep group) → apply only to that group, and override the global ones too

So if TEP needs `moe_config.backend: TRTLLM` and DEP needs `CUTEDSL`, just say so in each group. No code changes needed — it's all in the YAML.

### 6. Where Do These Values Come From?

All override values come from your **existing recipe YAMLs** (e.g. `recipes/trtllm/qwen3-235b/*.yaml`). The sweep YAML just tells the rate-matching tool to use those same settings:

| Sweep YAML field | Source | Why needed |
|---|---|---|
| `ctx_config.max_batch_size: 4` | Recipe prefill section | Tool default is 2 for ISL>2048 |
| `ctx_config.max_num_tokens: 32768` | Recipe prefill section | Tool default is 16896 |
| `moe_config.backend: TRTLLM` | Recipe decode (TEP) | Tool default is CUTLASS |
| `moe_config.backend: CUTEDSL` | Recipe decode (DEP) | Tool default is CUTLASS |
| `nvfp4_gemm_config` | Recipe decode section | Not in tool defaults at all |
| `enable_lm_head_tp_in_adp: false` | Recipe decode section | Tool default is true for DEP |
| `ENROOT_ALLOW_DEV: "1"` | GB200 cluster requirement | Not needed on H200 |

---

## Backward Compatibility

All changes are **additive and opt-in**:

| Feature | H200 (no overrides) | GB200 (with overrides) |
|---------|---------------------|------------------------|
| `decode_overrides` | Not set → tool defaults | Set per group → custom backends |
| `trtllm_decode_overrides` | Not set → tool defaults | Set globally → common overrides |
| `gpus_per_node: 8` | `ceil(8/8) = 1 node` | `ceil(8/4) = 2 nodes` |
| `gpus_per_prefill/decode` | Explicit but same as old implicit | Explicit and correct for TP<gpus_per_node |
| `sbatch_directives: {}` | No GRES (--exclusive handles it) | No GRES (same) |

**Zero H200 configs need updating.** Existing sweep YAMLs without the new fields produce identical behavior.

---

## Files Changed

```
tools/rate_matching/
├── schema.py              # Per-group decode_overrides/prefill_overrides fields
├── generate_configs.py    # Multi-node, deep-merge, gpus_per_worker, no cpus-per-gpu
├── run_sweep.py           # TP in filenames, override propagation through pipeline
├── qwen3_235b_gb200_sweep.yaml  # Example GB200 sweep config (new, untracked)

src/srtctl/templates/
├── job_script_minimal.j2  # Container pip install, strip .venv from PATH
```

---

## Validation

Tested with Qwen3-235B on GB200 (4 GPUs/node, aarch64):
- **10 GEN configurations**: TEP TP=2, TEP TP=4, DEP TP=8, DEP TP=16
- **Concurrency range**: 1 to 1024
- **Compared against China team SOL data**: results match within 1-5% after MoE backend fix
- **Per-group overrides verified**: TEP → `TRTLLM`, DEP → `CUTEDSL` in generated configs
- **H200 backward compat verified**: configs without new fields produce identical output

