# CLI Reference

`srtctl` is the main command-line interface for submitting benchmark jobs to SLURM.

## Table of Contents

- [Quick Start](#quick-start)
- [Interactive Mode](#interactive-mode)
  - [Recipe Browser](#recipe-browser)
  - [Configuration Summary](#configuration-summary)
  - [Interactive Actions Menu](#interactive-actions-menu)
  - [sbatch Preview](#sbatch-preview)
  - [Parameter Modification](#parameter-modification)
  - [Sweep Preview](#sweep-preview)
  - [Submission Confirmation](#submission-confirmation)
  - [Workflow Examples](#workflow-examples)
- [Commands](#commands)
  - [srtctl apply](#srtctl-apply)
  - [srtctl dry-run](#srtctl-dry-run)
  - [srtctl container-pull](#srtctl-container-pull)
- [Output](#output)
- [Sweep Support](#sweep-support)
- [Tips](#tips)

---

## Quick Start

```bash
# Interactive mode - browse recipes, preview, and submit
srtctl

# Submit a job directly
srtctl apply -f recipes/gb200-fp8/sglang-1p4d.yaml

# Preview execution plan without submitting
srtctl dry-run -f config.yaml

# Download containers defined in srtslurm.yaml
srtctl container-pull
```

## Interactive Mode

Running `srtctl` with no arguments launches an interactive TUI (Text User Interface) powered by Rich and Questionary:

```bash
srtctl
# or explicitly:
srtctl -i
```

Interactive mode is ideal for:

- Exploring available recipes without memorizing paths
- Previewing and tweaking configurations before submission
- Understanding what a sweep will expand to
- Quick experimentation and validation

### Recipe Browser

On launch, interactive mode scans the `recipes/` directory and presents recipes organized by subdirectory:

```
? Select a recipe:
  ── gb200-fp8 ──
    sglang-1p4d.yaml
    sglang-2p8d.yaml
    dynamo-router.yaml
  ── h100-fp8 ──
    baseline.yaml
    high-throughput.yaml
  ──────────────
  📁 Browse for file...
```

**Features:**

- Recipes grouped by parent directory for easy navigation
- Arrow keys to navigate, Enter to select
- "Browse for file..." option for configs outside `recipes/`
- If no recipes found, prompts for manual path entry

### Configuration Summary

After selecting a recipe, you'll see a tree-style summary:

```
📋 Configuration
┌─────────────────────────────────────────────────────────────┐
│ deepseek-r1-1p4d                                            │
└─────────────────────────────────────────────────────────────┘

deepseek-r1-1p4d
├── 📦 Model
│   ├── path: deepseek-r1
│   ├── container: latest
│   └── precision: fp8
├── 🖥️  Resources
│   ├── gpu_type: gb200
│   ├── prefill: 1 workers
│   ├── decode: 4 workers
│   └── gpus_per_node: 4
├── 📊 Benchmark
│   ├── type: sa-bench
│   ├── isl: 1024, osl: 1024
│   └── concurrencies: [128, 256, 512]
└── 🔄 Sweep Parameters (if present)
    ├── chunked_prefill_size: [4096, 8192]
    └── max_total_tokens: [8192, 16384]
```

### Interactive Actions Menu

After viewing the config summary, you'll see an action menu:

```
? What would you like to do?
  🚀 Submit job(s)          - Submit to SLURM cluster
  👁️  Preview sbatch script  - View generated SLURM script with syntax highlighting
  ✏️  Modify parameters      - Interactively change values before submission
  🔍 Dry-run                - Full dry-run preview without submission
  📁 Select different config - Choose a different recipe
  ❌ Exit                   - Exit interactive mode
```

### sbatch Preview

The "Preview sbatch script" option shows the exact SLURM script that will be submitted:

```bash
┌─ Generated sbatch Script ────────────────────────────────────────────────────┐
│  1 │ #!/bin/bash                                                             │
│  2 │ #SBATCH --job-name=deepseek-r1-1p4d                                     │
│  3 │ #SBATCH --nodes=5                                                       │
│  4 │ #SBATCH --gpus-per-node=4                                               │
│  5 │ #SBATCH --time=04:00:00                                                 │
│  6 │ #SBATCH --partition=batch                                               │
│  7 │ ...                                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The script is syntax-highlighted with line numbers for easy reading.

### Parameter Modification

The "Modify parameters" option lets you interactively change key settings:

```
Modify Configuration
Press Enter to keep current value, or type new value

? Job name [deepseek-r1-1p4d]: my-experiment
? Prefill workers [1]:
? Decode workers [4]: 8
? Input sequence length [1024]: 2048
? Output sequence length [1024]: 2048
```

**Modifiable fields:**

- `name` - Job name
- `resources.prefill_workers` - Number of prefill workers
- `resources.decode_workers` - Number of decode workers
- `benchmark.isl` - Input sequence length
- `benchmark.osl` - Output sequence length

Modified configs are saved to a temporary file and used for submission.

### Sweep Preview

For configs with a `sweep:` section, interactive mode shows an expansion table:

```
┌─ Sweep Jobs ────────────────────────────────────────────────────────────────┐
│ #  │ Job Name                           │ Parameters                        │
├────┼────────────────────────────────────┼───────────────────────────────────┤
│ 1  │ deepseek-r1-1p4d_cps4096_mtt8192   │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=8192              │
│ 2  │ deepseek-r1-1p4d_cps4096_mtt16384  │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=16384             │
│ 3  │ deepseek-r1-1p4d_cps8192_mtt8192   │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=8192              │
│ 4  │ deepseek-r1-1p4d_cps8192_mtt16384  │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=16384             │
└─────────────────────────────────────────────────────────────────────────────┘

Total jobs: 4
```

### Submission Confirmation

Before submitting, you'll be asked to confirm:

```
? Submit to SLURM? (y/N)
```

For sweeps, the confirmation shows:

- Full configuration summary
- Sweep expansion table
- Total job count

### Workflow Examples

**Exploring a new recipe:**

```
$ srtctl
> Select: gb200-fp8/sglang-1p4d.yaml
> Action: 👁️  Preview sbatch script  (review generated script)
> Action: 🔍 Dry-run                 (full dry-run)
> Action: 📁 Select different config (try another)
```

**Quick experiment with modifications:**

```
$ srtctl
> Select: gb200-fp8/sglang-1p4d.yaml
> Action: ✏️  Modify parameters
  > Change decode_workers: 8
  > Change isl: 2048
> Action: 🚀 Submit job(s)
> Confirm: y
```

**Sweep validation:**

```
$ srtctl
> Select: configs/my-sweep.yaml
> View: Sweep table showing 16 jobs
> Action: 🔍 Dry-run (saves all expanded configs to dry-runs/)
> Review generated configs
> Action: 🚀 Submit job(s)
```

## Commands

### `srtctl apply`

Submit a job or sweep to SLURM.

```bash
srtctl apply -f <config.yaml> [options]
```

**Options:**

| Flag             | Description                              |
| ---------------- | ---------------------------------------- |
| `-f, --file`     | Path to YAML config file (required)      |
| `--sweep`        | Force sweep mode (usually auto-detected) |
| `--setup-script` | Custom setup script from `configs/`      |
| `--tags`         | Comma-separated tags for the run         |
| `-y, --yes`      | Skip confirmation prompts                |

**Examples:**

```bash
# Submit single job
srtctl apply -f recipes/gb200-fp8/sglang-1p4d.yaml

# Submit sweep (auto-detected from sweep: section)
srtctl apply -f configs/my-sweep.yaml

# With tags
srtctl apply -f config.yaml --tags "experiment-1,baseline"
```

### `srtctl dry-run`

Preview the execution plan without actually submitting to SLURM.

```bash
srtctl dry-run -f <config.yaml> [options]
```

**Options:**

| Flag         | Description                         |
| ------------ | ----------------------------------- |
| `-f, --file` | Path to YAML config file (required) |
| `--sweep`    | Force sweep mode                    |

**Examples:**

```bash
# Preview single job - shows execution plan and sbatch script
srtctl dry-run -f config.yaml

# Preview sweep - shows job table and saves configs
srtctl dry-run -f sweep-config.yaml
```

**Enhanced Dry-Run Output:**

The enhanced dry-run shows a complete execution plan:

```
Runtime Context
  Job ID: DRY_RUN
  Run Name: my-config_DRY_RUN
  Head Node: node-01
  Worker Nodes: node-01, node-02, node-03
  Container: /path/to/container.sqsh
  Model: /path/to/model

Endpoint Allocation
┌──────────────┬─────────┬──────────────────────┬─────────┬────────────┐
│ Endpoint     │ Mode    │ Nodes                │ GPUs    │ Total GPUs │
├──────────────┼─────────┼──────────────────────┼─────────┼────────────┤
│ prefill_0    │ prefill │ node-01              │ [0-7]   │ 8          │
│ decode_0     │ decode  │ node-02              │ [0-7]   │ 8          │
│ decode_1     │ decode  │ node-03              │ [0-7]   │ 8          │
└──────────────┴─────────┴──────────────────────┴─────────┴────────────┘

Process Commands
  [prefill_0]
    Node: node-01
    GPUs: 0,1,2,3,4,5,6,7
    HTTP Port: 30000
    Command: srun --nodes=1 --nodelist=node-01 ...

Container Mounts
┌────────────────────────┬────────────────┐
│ Host Path              │ Container Path │
├────────────────────────┼────────────────┤
│ /path/to/logs          │ /logs          │
│ /path/to/model         │ /model         │
└────────────────────────┴────────────────┘
```

This helps you understand:

- How workers will be allocated across nodes
- Which GPUs each worker will use
- The exact commands that will be executed
- Container mounts and environment variables

### `srtctl container-pull`

Download container images defined in `srtslurm.yaml`.

```bash
srtctl container-pull [options]
```

**Options:**

| Flag          | Description                                                |
| ------------- | ---------------------------------------------------------- |
| `--force, -f` | Force re-download even if container already exists         |
| `--local`     | Run directly on login node instead of submitting batch job |

**Examples:**

```bash
# Download containers (submits batch job by default)
srtctl container-pull

# Force re-download all containers
srtctl container-pull --force

# Run directly on login node (if network/permissions allow)
srtctl container-pull --local
```

**How it works:**

By default, `container-pull` submits a SLURM batch job:

- Uses `default_account` and `default_partition` from `srtslurm.yaml`
- Runs `enroot import` on a compute node (better network access and permissions)
- Returns immediately with job ID for monitoring

With `--local`, downloads run directly on the login node.

**Example srtslurm.yaml:**

```yaml
containers:
  # Old format (no source, skipped by container-pull)
  nginx: "/shared/containers/nginx.sqsh"

  # New format (downloaded by container-pull)
  sglang:
    path: "/shared/containers/sglang.sqsh"
    source: "docker://nvcr.io/nvidia/sglang:0.4.1"
```

**Output:**

```
┌─ Containers ────────────────────────────────────────────────────────┐
│ Name   │ Path                           │ Source              │ Status       │
├────────┼────────────────────────────────┼─────────────────────┼──────────────┤
│ nginx  │ /shared/containers/nginx.sqsh  │ none                │ no source    │
│ sglang │ /shared/containers/sglang.sqsh │ docker://nvcr.io/.. │ will download│
└────────┴────────────────────────────────┴─────────────────────┴──────────────┘

  sglang: downloading from docker://nvcr.io/nvidia/sglang:0.4.1...
  sglang: saved to /shared/containers/sglang.sqsh

Summary: 1 downloaded, 1 skipped
```

## Output

When you submit a job, `srtctl` creates an output directory:

```
outputs/<job_id>/
├── config.yaml         # Copy of submitted config
├── sbatch_script.sh    # Generated SLURM script
└── <job_id>.json       # Job metadata
```

## Sweep Support

Configs with a `sweep:` section are automatically detected and expanded:

```yaml
sweep:
  chunked_prefill_size: [4096, 8192]
  max_total_tokens: [8192, 16384]
```

This creates 4 jobs (2 × 2 Cartesian product). See [Parameter Sweeps](sweeps.md) for details.

## Tips

- Use `srtctl` (no args) for exploring recipes interactively
- Use `srtctl apply -f` for scripting and CI pipelines
- Always `dry-run` first for sweeps to check job count
- Check `outputs/<job_id>/` for submitted configs and metadata
