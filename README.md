# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/srtctl.git
cd srtctl
pip install -e .

# One-time setup (downloads NATS/ETCD, creates srtslurm.yaml)
make setup ARCH=aarch64  # or ARCH=x86_64
```

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

## Profiling

Add `--nsys` to any recipe to capture GPU profiles alongside your benchmark:

```bash
srtctl apply -f recipes/my-benchmark.yaml --nsys --profile-start 100 --profile-stop 105
```

### Support Matrix

| | TRT-LLM | SGLang |
|--|---------|--------|
| **nsys** | Yes | Yes |
| **torch profiler** | -- | Yes |
| **Profile alongside benchmark** | Yes | Yes |
| **Disaggregated (P+D)** | Yes (N+M workers) | Yes (1P+1D, NIXL) |
| **Aggregated** | Yes | Yes (multi-worker) |
| **Multi-node TP** | Yes | Yes |
| **Per-phase windows** | Yes | Yes |
| **Activation** | `TLLM_PROFILE_START_STOP` env var | `/start_profile` HTTP API |

Both backends use `nsys profile -c cudaProfilerApi` -- nsys waits for the application
to call `cudaProfilerStart()`/`cudaProfilerStop()` at the configured iteration window.

See [src/srtctl/README.md](src/srtctl/README.md#profiling-support) for the full CLI
reference, YAML config, backend details, and profile analysis tips.
