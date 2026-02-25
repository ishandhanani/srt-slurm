# Parameter Sweeps

Parameter sweeps let you run multiple configurations with a single command. Sweeps are automatically detected from config files that contain a `sweep:` section.

## Table of Contents

- [How It Works](#how-it-works)
- [Simple Walkthrough](#simple-walkthrough)
- [Correlated Parameters](#correlated-parameters)
- [Where Placeholders Can Go](#where-placeholders-can-go)
- [Auto-Detection](#auto-detection)
- [Tips](#tips)

---

## How It Works

1. Add a `sweep:` section to your YAML config -- a list of parameter dicts
2. Add `{placeholder}` markers where you want values substituted
3. Run `srtctl apply -f <config>` -- sweep mode is auto-detected
4. `srtctl` generates and submits one job per sweep entry

Each entry in the `sweep` list is one job. All parameters in that entry are expanded together, so correlated values stay consistent.

## Simple Walkthrough

### Step 1: Create a sweep config

```yaml
name: "concurrency-sweep"

model:
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 4

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [{concurrency}]

sweep:
  - {concurrency: 128}
  - {concurrency: 256}
  - {concurrency: 512}
```

### Step 2: Preview with dry-run

```bash
srtctl dry-run -f configs/concurrency-sweep.yaml
```

This shows you what will be generated without submitting.

### Step 3: Submit

```bash
srtctl apply -f configs/concurrency-sweep.yaml
```

This submits 3 separate jobs, one for each concurrency value.

## Correlated Parameters

The key advantage of the list format: parameters that must change together stay grouped in a single dict.

```yaml
# Sweep prefill parallelism mapping on NVL72
resources:
  prefill_nodes: "{prefill_nodes}"
  prefill_workers: "{prefill_workers}"

backend:
  sglang_config:
    prefill:
      tensor-parallel-size: "{tp_size}"
      expert-parallel-size: "{ep_size}"

sweep:
  - {prefill_nodes: 1, prefill_workers: 1, tp_size: 4, ep_size: 1}
  - {prefill_nodes: 2, prefill_workers: 2, tp_size: 4, ep_size: 2}
  - {prefill_nodes: 4, prefill_workers: 4, tp_size: 4, ep_size: 4}
```

This generates 3 jobs. Each job has consistent resource allocation and parallelism -- no invalid combinations.

## Where Placeholders Can Go

Placeholders work anywhere in the YAML:

```yaml
name: "sweep-{param}"
mem-fraction-static: "{mem}"
concurrencies: [{conc}]
dp-size: "{dp}"
prefill_nodes: "{nodes}"
```

Values are substituted as strings. For integer fields (like `prefill_nodes`), the schema coerces `"2"` to `2` automatically.

## Auto-Detection

Sweep configs are automatically detected by the presence of a `sweep:` section:

```bash
# Auto-detected sweep
srtctl apply -f sweep-config.yaml

# Force sweep mode (if auto-detection fails)
srtctl apply -f config.yaml --sweep
```

## Tips

- Always use `srtctl dry-run -f <config>` first to verify
- Start with 2-3 entries before running large sweeps
- Each job gets a unique name based on parameter values
- See `examples/example-sweep.yaml` for a full working example
