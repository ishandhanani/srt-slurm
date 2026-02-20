# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Setup

### 1. Install

```bash
git clone https://github.com/your-org/srtctl.git
cd srtctl
pip install -e .

# Downloads NATS/ETCD binaries, creates srtslurm.yaml template
make setup ARCH=aarch64  # or ARCH=x86_64
```

### 2. Configure `srtslurm.yaml`

`make setup` generates a `srtslurm.yaml` file in your repo root. This is the cluster-level config where you register model paths, container images, and SLURM defaults.

**Add your container images** — pull from Docker Hub and convert to squashfs for Pyxis/enroot:

```bash
# Pull the SGLang image
docker pull lmsysorg/sglang:dev-cu13-kimi-k2p5-fix

# Convert to squashfs
enroot import dockerd://lmsysorg/sglang:dev-cu13-kimi-k2p5-fix
mv lmsysorg+sglang+dev-cu13-kimi-k2p5-fix.sqsh /path/to/shared/storage/
```

Then register it in `srtslurm.yaml`:

```yaml
containers:
  "sglang-dev": "/path/to/shared/storage/lmsysorg+sglang+dev-cu13-kimi-k2p5-fix.sqsh"
```

**Add your model paths** — download model weights to shared or node-local storage, then register aliases:

```yaml
model_paths:
  kimi25nvfp4: "/raid/models/Kimi-K2.5-nvfp4/"
  my-model: "/lustre/models/my-model/"
```

### 3. Create a recipe

Recipes are YAML files that reference the aliases from `srtslurm.yaml`:

```yaml
name: "kimi-k2.5-tp8-nvfp4-sweep"

model:
  path: "kimi25nvfp4"          # alias from srtslurm.yaml
  container: "sglang-dev"      # alias from srtslurm.yaml
  precision: "fp4"

resources:
  gpu_type: "b200"
  agg_nodes: 1
  agg_workers: 1
  gpus_per_node: 8

backend:
  sglang_config:
    aggregated:
      trust-remote-code: true
      tensor-parallel-size: 8
      quantization: modelopt_fp4

frontend:
  type: "sglang"

health_check:
  max_attempts: 360
  interval_seconds: 10

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [16, 32, 64, 128, 256]
  req_rate: "inf"
```

### 4. Run

```bash
# Validate config without submitting
srtctl dry-run -f recipes/my-recipe.yaml

# Submit to SLURM
srtctl apply -f recipes/my-recipe.yaml
```

Results land in `outputs/<job_id>/logs/benchmark.out`.

## Documentation

**Full documentation:** https://srtctl.gitbook.io/srtctl-docs/

- [Installation](docs/installation.md) - Setup and configuration
- [Monitoring](docs/monitoring.md) - Job logs and debugging
- [Parameter Sweeps](docs/sweeps.md) - Grid searches
- [Profiling](docs/profiling.md) - Torch/nsys profiling
- [Analyzing Results](docs/analyzing.md) - Dashboard and visualization

## Commands

```bash
# Submit job(s)
srtctl apply -f config.yaml

# Submit with custom setup script
srtctl apply -f config.yaml --setup-script custom-setup.sh

# Submit with tags for filtering
srtctl apply -f config.yaml --tags experiment,baseline

# Dry-run (validate without submitting)
srtctl dry-run -f config.yaml

# Launch analysis dashboard
uv run streamlit run analysis/dashboard/app.py
```
