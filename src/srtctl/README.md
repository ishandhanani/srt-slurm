# srtctl - Python-first SLURM Orchestration

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, replacing the previous Jinja/bash-heavy approach.

## Architecture

```
srtctl/
├── __init__.py              # Package exports
├── cli/
│   ├── submit.py            # srtctl apply - job submission
│   ├── do_sweep.py          # srtctl-sweep - main orchestrator
│   └── setup_head.py        # Head node infrastructure (NATS/etcd)
├── core/
│   ├── config.py            # Config loading and srtslurm.yaml resolution
│   ├── runtime.py           # RuntimeContext - single source of truth
│   ├── topology.py          # Endpoint/Process allocation for workers
│   ├── processes.py         # ProcessRegistry - lifecycle management
│   ├── slurm.py             # SLURM srun launching and node resolution
│   ├── health.py            # Health checks (HTTP polling, worker readiness)
│   ├── schema.py            # Frozen dataclass schemas
│   ├── sweep.py             # Sweep parameter handling
│   └── ip_utils/            # Bash-based IP resolution utilities
│       ├── __init__.py      # Python wrappers for bash functions
│       └── get_node_ip.sh   # IP detection bash functions
├── backends/
│   ├── base.py              # BackendProtocol interface
│   └── sglang.py            # SGLang implementation
├── benchmarks/
│   ├── base.py              # BenchmarkRunner ABC
│   ├── sa_bench.py          # Serving benchmark
│   ├── router.py            # Router benchmark
│   └── ...                  # Other benchmark types
└── templates/               # Jinja2 templates for sbatch scripts
```

## Usage

```bash
srtctl apply -f config.yaml
```

## Key Concepts

### RuntimeContext

Single source of truth for all computed paths and values. Replaces bash
variables scattered throughout Jinja templates.

```python
runtime = RuntimeContext.from_config(config, job_id)
print(runtime.log_dir)       # Computed once
print(runtime.model_path)    # Resolved from config
print(runtime.head_node_ip)  # From SLURM
```

### Endpoints and Processes

Typed Python replaces bash array math:

```python
# Old (Jinja/bash):
# for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
#     leader_idx=$((WORKER_NODE_OFFSET + i * PREFILL_NODES_PER_WORKER))
# done

# New (Python):
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4, num_agg=0,
    gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=8,
    gpus_per_node=8, available_nodes=nodes
)
for endpoint in endpoints:
    print(f"{endpoint.mode} worker {endpoint.index} on {endpoint.nodes}")
```

### ProcessRegistry

Manages process lifecycle with health monitoring:

```python
registry = ProcessRegistry(job_id)
registry.add_process(worker_proc)

# Background thread monitors for failures
if registry.check_failures():
    registry.cleanup()  # Graceful shutdown
```

### Health Checks

HTTP-based health checking for different frontends:

```python
from srtctl.core.health import wait_for_model

# Wait for all workers to register
wait_for_model(
    host=head_ip, port=8000,
    n_prefill=2, n_decode=4,
    frontend_type="sglang",  # or "dynamo"
    timeout=300,
)
```

For aggregated mode, pass `n_prefill=0, n_decode=num_agg`.

### BackendProtocol

Interface for different serving frameworks:

```python
class BackendProtocol(Protocol):
    @property
    def type(self) -> BackendType: ...
    def build_worker_command(self, process, runtime) -> list[str]: ...
```

### Multiple Workers Per Node

The allocator automatically handles placing multiple workers on a single node:

```yaml
resources:
  gpus_per_node: 8
  decode_workers: 2
  gpus_per_decode: 4 # 2 workers × 4 GPUs = 8 GPUs = 1 node
```

`CUDA_VISIBLE_DEVICES` is automatically set per worker (e.g., `0,1,2,3` and `4,5,6,7`).

## Profiling Support

srtctl supports GPU profiling with NVIDIA Nsight Systems (nsys) and PyTorch Profiler.
The recommended approach is **CLI flags** -- layer profiling on top of any existing recipe
without modifying YAML files.

### Support Matrix

#### Profiler Type Support

| Profiler | TRT-LLM | SGLang | Notes |
|----------|---------|--------|-------|
| **nsys** (Nsight Systems) | Yes | Yes | GPU kernel traces, NVTX markers, CUDA API calls |
| **torch** (PyTorch Profiler) | No | Yes | CPU/GPU/memory activity via SGLang's `/start_profile` API |

#### Deployment Topology

| Topology | TRT-LLM | SGLang | Notes |
|----------|---------|--------|-------|
| **Aggregated** (all workers same role) | Yes | Yes | SGLang uses `sglang_router` for multi-worker routing |
| **Disaggregated** (separate P+D) | Yes, N prefill + M decode | Yes, 1P + 1D | SGLang NIXL transfer currently limits to 1P+1D |
| **Multi-node TP** (TP spanning nodes) | Yes | Yes | Each worker's `nsys` captures all ranks across its nodes |

#### Profiling Modes

| Mode | TRT-LLM | SGLang | Notes |
|------|---------|--------|-------|
| **Alongside benchmark** | Yes (validated E2E) | Yes | Workers profiled while sa-bench / other benchmark generates traffic |
| **Dedicated runner** | Yes | Yes | `profile.sh` generates its own traffic; requires `--profile-opt isl=X osl=Y concurrency=Z` |
| **Per-phase windows** | Yes | Yes | Separate `--profile-start/stop` vs `--profile-start-decode/stop-decode` |

#### How Each Backend Activates Profiling

| Backend | Mechanism | What happens |
|---------|-----------|--------------|
| **TRT-LLM** | `TLLM_PROFILE_START_STOP=100-105` env var | TRT-LLM executor calls `cudaProfilerStart()` / `cudaProfilerStop()` at those iteration boundaries automatically |
| **SGLang** (nsys) | `/start_profile` HTTP API with `["CUDA_PROFILER"]` | SGLang server calls `cudaProfilerStart()` / `cudaProfilerStop()` internally |
| **SGLang** (torch) | `/start_profile` HTTP API with `["CPU", "GPU", "MEM"]` | SGLang activates PyTorch Profiler; traces written to `SGLANG_TORCH_PROFILER_DIR` |

Both backends use `nsys profile -c cudaProfilerApi` wrapping, meaning nsys waits
for the application to signal when to start/stop capture rather than profiling the
entire process lifetime.

#### Output

| Detail | TRT-LLM | SGLang |
|--------|---------|--------|
| **Profile files** | `.nsys-rep` per worker | `.nsys-rep` per worker (nsys) or PyTorch traces (torch) |
| **Output location** | `outputs/<JOB_ID>/logs/<node>_<role>_w<N>_profile/` | Same |
| **File naming** | `profile_<hostname>_<pid>.nsys-rep` | Same |

#### CLI Flags

| Flag | TRT-LLM | SGLang | Description |
|------|---------|--------|-------------|
| `--nsys` | Yes | Yes | Enable Nsight Systems profiling |
| `--torch-profile` | -- | Yes | Enable PyTorch profiler (mutually exclusive with `--nsys`) |
| `--profile-start N` | Yes | Yes | Start step for prefill/aggregated workers (default: 100) |
| `--profile-stop N` | Yes | Yes | Stop step for prefill/aggregated workers (default: 105) |
| `--profile-start-decode N` | Yes | Yes | Start step for decode workers (defaults to `--profile-start`) |
| `--profile-stop-decode N` | Yes | Yes | Stop step for decode workers (defaults to `--profile-stop`) |
| `--profile-opt KEY=VALUE` | Yes | Yes | Extra profiling config (repeatable), e.g. `gpu_metrics=true` |
| `--nsys-mount PATH` | Yes | Yes | Host path to nsight-systems, or `none` to skip mount |

<details>
<summary><b>Prerequisites: Nsight Systems Setup</b></summary>

### Nsight Systems must be available on the host

The `--nsys` flag mounts nsight-systems from the **host** into the container at runtime.
Most inference containers (TRT-LLM, SGLang) do **not** bundle nsys, so it must be
installed on the bare-metal cluster nodes.

### Step 1: Check if nsys is already installed

SSH to a compute node and run:

```bash
# Check the default install location
ls /opt/nvidia/nsight-systems/
# Expected: a version directory like 2025.1.3/

# Or check if nsys is in PATH
which nsys && nsys --version
```

On DGX/HGX systems nsight-systems is typically pre-installed at `/opt/nvidia/nsight-systems/`.
If it's already there, skip to Step 3.

### Step 2: Install nsight-systems (if not present)

**Option A: apt (Ubuntu/Debian) -- recommended for cluster nodes**

```bash
# Add NVIDIA devtools repo (run on each compute node, or via SLURM prolog)
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
  | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update

# Install CLI only (no GUI needed on cluster nodes)
apt install -y nsight-systems-cli
```

After install, verify:

```bash
ls /opt/nvidia/nsight-systems/
nsys --version
```

**Option B: Download the .run installer (no root required)**

Download from https://developer.nvidia.com/nsight-systems/get-started and run:

```bash
chmod +x NsightSystems-linux-public-*.run
./NsightSystems-linux-public-*.run --accept --target /opt/nvidia/nsight-systems/
```

**Option C: Already available via CUDA Toolkit**

If CUDA Toolkit is installed on the host, nsys may already be at
`/usr/local/cuda/nsight-systems/` or `/opt/nvidia/nsight-systems/`.

### Step 3: Configure the mount path

srtctl auto-mounts `/opt/nvidia/nsight-systems` from the host into the container when
you use `--nsys`. If your install is at the default path, no configuration is needed.

**If nsys is at a non-default path**, set it in `srtslurm.yaml`:

```yaml
# srtslurm.yaml
nsys_mount_path: "/usr/local/cuda/nsight-systems"
```

Or pass it per-job:

```bash
srtctl apply -f recipe.yaml --nsys --nsys-mount /usr/local/cuda/nsight-systems
```

**If nsys is already bundled in your container**, skip the mount entirely:

```bash
srtctl apply -f recipe.yaml --nsys --nsys-mount none
```

### Step 4: Verify with dry-run

```bash
srtctl dry-run -f recipe.yaml --nsys
```

The profiling summary table will show the nsys mount status:
- `nsys mount: /opt/nvidia/nsight-systems present` -- mount configured
- `nsys mount: MISSING` -- fix with steps above

### Quick reference

```bash
# Verify nsys is available
ls /opt/nvidia/nsight-systems/ && echo "OK: nsys found"

# Test profiling without submitting
srtctl dry-run -f recipe.yaml --nsys

# Submit with profiling
srtctl apply -f recipe.yaml --nsys --profile-start 100 --profile-stop 105

# If nsys is elsewhere
srtctl apply -f recipe.yaml --nsys --nsys-mount /path/to/nsight-systems

# If nsys is inside the container
srtctl apply -f recipe.yaml --nsys --nsys-mount none
```

</details>

<details>
<summary><b>Quick Start: CLI Profiling (Recommended)</b></summary>

### Profile any recipe with `--nsys`

```bash
# Nsys profiling with default window (steps 100-105)
srtctl apply -f recipes/my-benchmark.yaml --nsys

# Custom profiling window
srtctl apply -f recipes/my-benchmark.yaml --nsys --profile-start 200 --profile-stop 210

# Different windows for prefill vs decode workers
srtctl apply -f recipes/my-benchmark.yaml --nsys \
  --profile-start 100 --profile-stop 105 \
  --profile-start-decode 500 --profile-stop-decode 505

# PyTorch profiler (SGLang only)
srtctl apply -f recipes/my-benchmark.yaml --torch-profile

# Dry-run to verify config
srtctl dry-run -f recipes/my-benchmark.yaml --nsys
```

### How it works

CLI flags inject a `profiling` config section at submission time. The recipe YAML is
never modified. Workers start wrapped in `nsys profile` (or with torch profiler env vars)
and your chosen benchmark runs normally -- the profiled iterations are captured as the
benchmark generates traffic.

### Two modes

- **Profiling alongside a benchmark** (default): Use `--nsys` with a recipe that already
  has `benchmark.type: sa-bench` (or any other benchmark). The benchmark generates traffic
  and workers capture nsys/torch profiles during the specified iteration window.

- **Dedicated profiling runner**: If your recipe has `benchmark.type: manual` or no
  benchmark, srtctl auto-selects the dedicated `profile.sh` runner that generates its own
  controlled traffic (requires `--profile-opt isl=X osl=Y concurrency=Z`).

### Power-user knobs: `--profile-opt`

Pass any `ProfilingConfig` field as a key=value pair:

```bash
srtctl apply -f recipe.yaml --nsys \
  --profile-opt gpu_metrics=true \
  --profile-opt num_prompts=512 \
  --profile-opt isl=8192 \
  --profile-opt osl=1024 \
  --profile-opt concurrency=64
```

### CLI Flags Reference

- `--nsys` -- Enable NVIDIA Nsight Systems profiling
- `--torch-profile` -- Enable PyTorch profiler (mutually exclusive with `--nsys`)
- `--profile-start N` -- Profiling start step for prefill/agg workers (default: 100)
- `--profile-stop N` -- Profiling stop step for prefill/agg workers (default: 105)
- `--profile-start-decode N` -- Start step for decode workers (defaults to `--profile-start`)
- `--profile-stop-decode N` -- Stop step for decode workers (defaults to `--profile-stop`)
- `--profile-opt KEY=VALUE` -- Extra profiling options (repeatable)

</details>

<details>
<summary><b>Advanced: YAML-Based Profiling Configuration</b></summary>

For CI pipelines or reproducible profiling jobs, you can also set profiling in YAML:

```yaml
profiling:
  type: "nsys"        # "nsys" or "torch"
  isl: 8192           # Input sequence length (only for dedicated profiling runner)
  osl: 1024           # Output sequence length (only for dedicated profiling runner)
  concurrency: 64     # Max concurrent requests (only for dedicated profiling runner)
  num_prompts: 512    # Total prompts to generate
  prefill:
    start_step: 100
    stop_step: 105
  decode:
    start_step: 500
    stop_step: 505

container_mounts:
  /opt/nvidia/nsight-systems: /opt/nvidia/nsight-systems
```

When using profiling alongside a real benchmark (e.g., `benchmark.type: sa-bench`),
the `isl`, `osl`, and `concurrency` fields in the profiling section are optional -- traffic
comes from the benchmark itself.

</details>

<details>
<summary><b>Backend-Specific Profiling Mechanisms</b></summary>

Both SGLang and TRT-LLM use `nsys profile -c cudaProfilerApi` which tells nsys to wait for the application to call `cudaProfilerStart()` and `cudaProfilerStop()`. The difference is **how** each backend triggers these CUDA calls:

### SGLang

SGLang exposes a `/start_profile` HTTP API endpoint. When the profiling script calls this API with `activities: ["CUDA_PROFILER"]` and iteration parameters, SGLang internally calls `cudaProfilerStart()` at the specified iteration and `cudaProfilerStop()` after the specified number of steps. When profiling is enabled, srtctl automatically switches from `dynamo.sglang` to `sglang.launch_server`.

### TRT-LLM

TRT-LLM reads the `TLLM_PROFILE_START_STOP` environment variable at worker startup. The TRT-LLM executor automatically calls `cudaProfilerStart()` and `cudaProfilerStop()` at those iteration boundaries without requiring any external API calls.

```bash
# Set automatically by srtctl:
TLLM_PROFILE_START_STOP="100-105"  # Start at iter 100, stop at iter 105
```

### Summary Table

| Backend | Profiler | Activation Mechanism |
|---------|----------|---------------------|
| SGLang  | torch    | `/start_profile` API with `["CPU", "GPU", "MEM"]` |
| SGLang  | nsys     | `/start_profile` API with `["CUDA_PROFILER"]` |
| TRT-LLM | nsys     | `TLLM_PROFILE_START_STOP` env var |
| TRT-LLM | torch    | Not supported |

</details>

<details>
<summary><b>Analyzing nsys Profiles</b></summary>

### Download profiles to local machine

```bash
scp -r user@cluster:/path/to/outputs/JOB_ID/logs/*_profile/ ./profiles/
```

### Open with Nsight Systems GUI

Install from https://developer.nvidia.com/nsight-systems and open `.nsys-rep` files.

### Command-line analysis

```bash
# Top GPU kernels by time
nsys stats profile.nsys-rep --report cuda_gpu_kern_sum

# NVTX markers (TRT-LLM phases, SGLang operations)
nsys stats profile.nsys-rep --report nvtx_sum

# CUDA API call summary
nsys stats profile.nsys-rep --report cuda_api_sum

# Export to CSV for custom analysis
nsys stats profile.nsys-rep --report cuda_gpu_kern_sum --format csv -o kernels.csv
```

### Key metrics to look for

| Metric | What it tells you |
|--------|-------------------|
| NCCL % | Communication overhead (TP/PP sync) |
| GEMM % | Compute utilization |
| Attention % | Memory bandwidth utilization |
| Gaps in timeline | CPU/scheduling overhead |

</details>

<details>
<summary><b>Advanced: GPU Metrics</b></summary>

To capture GPU performance counters (SM utilization, memory bandwidth), enable `gpu_metrics`:

```bash
srtctl apply -f recipe.yaml --nsys --profile-opt gpu_metrics=true
```

**Note:** This requires elevated privileges. The cluster admin must configure:

```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' > /etc/modprobe.d/nvidia-perf.conf
```

Without this, you'll see: `ERR_NVGPUCTRPERM: Insufficient privilege`

</details>

## Profiling Troubleshooting

<details>
<summary><b>Common Issues and Solutions</b></summary>

### "enroot-mount: failed to mount ... nsight-systems: No such file or directory"

**Cause:** Nsight Systems is not installed on the cluster host at the expected path
(`/opt/nvidia/nsight-systems`). The `--nsys` flag auto-mounts this directory into the
container so nsys is available at runtime.

**Fix:** See the **Prerequisites: Nsight Systems Setup** section above. Options:
- Install nsight-systems on cluster nodes (ask admin)
- Point to the correct path: `--nsys-mount /actual/path/to/nsight-systems`
- Skip the mount if nsys is in the container: `--nsys-mount none`
- Set `nsys_mount_path` in `srtslurm.yaml` for cluster-wide config

### "No reports were generated" / empty `.nsys-rep` files

**Cause:** `cudaProfilerStart()` was never called -- the worker exited or was killed before
reaching the profiling window.

**Fix:** Make sure `--profile-start` is set to a step the worker will actually reach.
If using "alongside benchmark" mode, the benchmark must generate enough traffic to reach
that iteration count. Start with a low value like `--profile-start 5 --profile-stop 10`
for validation.

### SGLang: "Scheduler watchdog timeout"

**Cause:** Large models (e.g., DeepSeek R1 FP8) have very long startup times due to
DeepGEMM warmup and JIT compilation. The default SGLang watchdog (300s) fires before
the model is ready.

**Fix:** Add `watchdog-timeout: 1800` (or higher) to your `sglang_config` for both
prefill and decode workers. Also increase `health_check.max_attempts` in your recipe
if startup exceeds 30 minutes:

```yaml
health_check:
  max_attempts: 360   # 60 minutes at 10s intervals
  interval_seconds: 10
```

### SGLang: "NIXL KVReceiver Exception"

**Cause:** The NIXL KV transfer between prefill and decode nodes failed. This is typically
a cluster RDMA/UCX networking issue, not a profiling issue.

**Fix:** Verify that InfiniBand / RoCE is properly configured between the nodes. For
profiling validation, consider using an aggregated (non-disaggregated) recipe to bypass
NIXL entirely.

### How to choose `--profile-start` and `--profile-stop` values

The profiling window should capture steady-state behavior, not warmup:

- **TRT-LLM:** The executor counts iterations internally. Start after the model has
  processed enough requests to fill its batching pipeline. For sa-bench with concurrency 128,
  `--profile-start 100 --profile-stop 110` typically works well.
- **SGLang:** The `/start_profile` API triggers at the specified iteration. Same guidance
  applies. For disaggregated mode, decode workers often need a higher start value since
  they process requests after prefill: `--profile-start-decode 500 --profile-stop-decode 510`.

A good rule of thumb: set the start step to ~2x your concurrency to ensure the pipeline
is fully loaded.

### `--torch-profile` with TRT-LLM

This combination is not supported. TRT-LLM does not expose PyTorch profiler hooks.
Use `--nsys` instead for TRT-LLM profiling.

### Profile files are very large

nsys profiles can be 1-10 GB per worker depending on the capture window length and
trace options. To reduce size:

- Keep the profiling window short (5-10 steps is usually sufficient)
- Avoid `--profile-opt gpu_metrics=true` unless you specifically need hardware counters

</details>

## Future Improvements

Planned enhancements for the profiling experience:

- **`srtctl profiles <job-id>`** -- List, download, and summarize profile files from a completed job
- **Auto-detect profile window** -- Automatically set start/stop based on benchmark duration and concurrency
- **Profile comparison** -- Compare two profiling runs side-by-side (e.g., before/after an optimization)
- **Submit-time validation** -- Warn at submit time if `--torch-profile` is used with an unsupported backend
- **Aggregated SGLang profiling recipe** -- Pre-built recipe for single-node profiling that avoids NIXL/disagg complexity
- **Profile size estimation** -- Show estimated disk usage in dry-run output based on worker count and window size
- **SGLang multi-P+D profiling** -- Track SGLang NIXL improvements to support N prefill + M decode profiling

## Files Overview

| File                 | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `core/config.py`     | YAML loading, srtslurm.yaml resolution   |
| `core/runtime.py`    | Computed paths/values (RuntimeContext)   |
| `core/topology.py`   | Worker topology and GPU allocation       |
| `core/processes.py`  | Process lifecycle management             |
| `core/slurm.py`      | SLURM srun launching, node IP resolution |
| `core/health.py`     | Health checks, worker readiness polling  |
| `core/ip_utils/`     | Bash-based IP detection utilities        |
| `cli/do_sweep.py`    | Main orchestrator (runs on head node)    |
| `backends/sglang.py` | SGLang backend implementation            |
| `backends/trtllm.py` | TRT-LLM backend implementation           |
| `benchmarks/base.py` | BenchmarkRunner ABC                      |
