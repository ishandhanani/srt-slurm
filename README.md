# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Setup

Please run the following command to setup the srtctl tool. This repo requires Dynamo 0.7.0 or later, which is now available on PyPI.

```bash
# One-time setup
make setup ARCH=aarch64  # or ARCH=x86_64
pip install -e .
```

## Documentation

**Full documentation:** https://srtctl.gitbook.io/srtctl-docs/
