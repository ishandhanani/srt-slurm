# Debugging Hangs

This guide covers how to debug hanging jobs using automated CUDA backtrace collection.

## Overview

When SGLang workers hang (e.g., during NCCL collectives, CUDA kernel execution, or waiting on synchronization), it's often difficult to diagnose because:

1. The process appears alive but isn't making progress
2. Standard logs don't show what's happening inside the GPU
3. Manual debugging requires catching the hang at the right moment

The `debug` configuration enables automated backtrace collection using `cuda-gdb`, which captures:
- Active CUDA kernels on each GPU
- Full Python/C++ stack traces from all threads
- The state of each worker process at the time of collection

## Quick Start

Add the `debug` section to your job config:

```yaml
name: my-benchmark
model:
  path: deepseek-r1
  container: latest
  precision: fp8
resources:
  gpu_type: gb200
  prefill_nodes: 1
  decode_nodes: 2

# Enable hang debugging
debug:
  enabled: true
  wait_seconds: 1800  # Collect backtraces after 30 minutes
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `false` | Enable hang debugging |
| `wait_seconds` | int | `600` | Time to wait before collecting backtraces (in seconds) |
| `output_dir` | string | `{log_dir}/backtraces` | Directory for backtrace output files |

### Examples

**Collect after 10 minutes (default):**
```yaml
debug:
  enabled: true
```

**Collect after 1 hour:**
```yaml
debug:
  enabled: true
  wait_seconds: 3600
```

**Custom output directory:**
```yaml
debug:
  enabled: true
  wait_seconds: 1800
  output_dir: /shared/debug/backtraces
```

## How It Works

1. **Named Containers**: Workers are launched with named containers (`sglang_{job_id}_{node}`) using pyxis `--container-name`

2. **Debug Script Launch**: After workers start, the debug script is launched on each node, **attaching to the existing worker container** using the same `--container-name`. This ensures:
   - The script runs in the same PID namespace as the workers
   - Access to `py-spy` and `cuda-gdb` tools installed in the container
   - Can directly attach to worker processes

3. **Wait Period**: The script sleeps for the configured `wait_seconds`

4. **Process Discovery**: The script searches for SGLang TP worker processes:
   - Primary search: Processes named `sglang::scheduler*` (set via `setproctitle`)
   - Fallback: Any `python.*sglang` process

5. **Backtrace Collection**: For each TP worker process:
   - **py-spy** (if available): Captures Python stack traces without pausing the process
   - **cuda-gdb**: Attaches and collects CUDA kernel info and native backtraces

6. **Output**: Two files per process:
   - `{output_dir}/pyspy_{hostname}_{pid}.txt` - Python stack traces
   - `{output_dir}/cudagdb_{hostname}_{pid}.txt` - CUDA kernels and native backtraces

## Understanding the Output

### py-spy Output (pyspy_*.txt)

Shows Python-level stack traces for all threads:

```
Thread 0x7f1234567890 (active): "MainThread"
    torch/cuda/__init__.py:123 synchronize
    sglang/srt/managers/scheduler.py:456 forward_batch
    sglang/srt/managers/scheduler.py:789 event_loop_normal
    ...

Thread 0x7f1234567891 (idle): "ThreadPoolExecutor-0_0"
    concurrent/futures/thread.py:78 _worker
    ...
```

**Advantages of py-spy:**
- Shows Python function names and line numbers
- Non-invasive (doesn't pause the process)
- Easier to understand than native backtraces

### cuda-gdb Output (cudagdb_*.txt)

Contains CUDA kernel info and native backtraces:

```
# CUDA Kernel Information
info cuda kernels output showing:
- Kernel function names
- Grid/block dimensions
- GPU device and SM information

# Thread Backtraces
Thread N (Thread 0x... (LWP ...)):
#0  0x... in function_name () at file.cpp:line
#1  0x... in caller_function () at file.cpp:line
...
```

### Common Hang Patterns

**NCCL Collective Hang:**
```
#0  __futex_abstimed_wait_common ()
#1  ncclCommInitRank ()
...
```
Indicates workers are waiting for NCCL initialization or a collective operation.

**CUDA Kernel Execution:**
```
info cuda kernels:
  Kernel 0: flash_fwd_kernel<...> on device 0
```
Shows which GPU kernels are currently executing.

**Python GIL Contention:**
```
#0  __lll_lock_wait ()
#1  PyGILState_Ensure ()
...
```
Indicates threads waiting for the Python Global Interpreter Lock.

## Process Identification

The debug script specifically targets **TP (Tensor Parallel) worker processes**, which are the processes that:
- Contain the actual model weights
- Execute forward passes
- Participate in NCCL collectives

These processes are identified by their process name `sglang::scheduler*`, which is set via `setproctitle` in SGLang. Each scheduler process contains a `TpModelWorker` instance.

Process names follow the pattern:
```
sglang::scheduler_TP{tp_rank}_PP{pp_rank}_DP{dp_rank}_EP{ep_rank}
```

## Troubleshooting

### No Backtraces Collected

**Symptom:** Output directory is empty or contains error messages.

**Causes:**
1. `py-spy` or `cuda-gdb` not available in the environment
2. Processes exited before backtrace collection
3. Insufficient permissions to attach to processes

**Solutions:**
- Ensure the container has `py-spy` and/or `cuda-gdb` installed
- Install py-spy: `pip install py-spy`
- Increase `wait_seconds` if processes exit quickly
- Check that ptrace is allowed (may need `--cap-add=SYS_PTRACE` for containers)

### cuda-gdb Timeout

**Symptom:** Backtrace file contains "cuda-gdb failed or timed out"

**Causes:**
- Process in uninterruptible state
- cuda-gdb deadlock

**Solutions:**
- The script uses a 60-second timeout per process
- Check if the process is in `D` state (`ps aux | grep sglang`)

### Wrong Processes Captured

**Symptom:** Backtraces show router/frontend processes instead of workers

**Solution:** The script specifically targets `sglang::scheduler*` processes. If you see other processes, check that SGLang is using `setproctitle` correctly.

## Best Practices

1. **Set wait_seconds appropriately**: Set it to slightly before when you expect the hang to occur. If hangs happen at 45 minutes, set `wait_seconds: 2400` (40 minutes).

2. **Use with profiling disabled**: The debug feature works best when profiling is disabled, as profiling can change timing behavior.

3. **Collect from all nodes**: The script automatically runs on all worker nodes, capturing the state of every TP worker.

4. **Preserve logs**: Backtraces are saved to the job's log directory by default, which is preserved after job completion.

5. **Multiple collection points**: If you're unsure when the hang occurs, submit multiple jobs with different `wait_seconds` values.

## Limitations

- **cuda-gdb is intrusive**: Attaching `cuda-gdb` briefly pauses the process (py-spy does not)
- **Single snapshot**: Only captures state at one point in time
- **Tool availability**: Requires `py-spy` and/or `cuda-gdb` in the execution environment
- **Process must be alive**: Cannot debug processes that have crashed
- **Permissions**: May require `CAP_SYS_PTRACE` capability to attach to processes

## See Also

- [Profiling](profiling.md) - For performance profiling with nsys/torch
- [Monitoring](monitoring.md) - For real-time job monitoring
- [Configuration Reference](config-reference.md) - Full config documentation
