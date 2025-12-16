# Analyze Results

```bash
uv run streamlit run analysis/dashboard/app.py

# Another way to launch dashboard
make dashboard
```
Opens interactive dashboard at http://localhost:8501


## Features

### 📊 Interactive Dashboard

- **Pareto Analysis** - TPS/GPU vs TPS/User tradeoffs
- **Latency Breakdown** - TTFT, TPOT, ITL across concurrency levels
- **Node Metrics** - Runtime metrics from prefill/decode nodes
- **Config Comparison** - Side-by-side configuration diffs
- **Run Comparison** - Performance deltas between runs

### 🚀 SLURM Job Submission

- Disaggregated (prefill/decode) or aggregated mode
- Multiple frontends with nginx load balancing (default)
- Automated benchmarking with sa-bench
- Job metadata tracking
