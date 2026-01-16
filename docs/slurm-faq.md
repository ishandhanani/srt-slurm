# SLURM FAQ

## Cluster Compatibility Settings

Some SLURM clusters don't support certain SBATCH directives. If you encounter errors during job submission, you may need to adjust these settings in your `srtslurm.yaml`.

## GPU Resource Specification

If you see this error when submitting jobs:

```
sbatch: error: Invalid generic resource (gres) specification
```

Your cluster doesn't support the `--gpus-per-node` directive. Disable it:

```yaml
use_gpus_per_node_directive: false
```

This will omit the `#SBATCH --gpus-per-node` directive from generated job scripts while keeping all other functionality intact.

## Segment-Based Scheduling

If you see this error when submitting jobs:

```
sbatch: error: Invalid --segment specification
```

Your cluster doesn't support the `--segment` directive for topology-aware scheduling. Disable it:

```yaml
use_segment_sbatch_directive: false
```

The `--segment` directive ensures all allocated nodes are within the same network segment/switch for optimal interconnect performance between prefill and decode workers. If your cluster doesn't support it, SLURM will still allocate nodes but may scatter them across the cluster.

## Exclusive Node Access

Some clusters require jobs to explicitly request exclusive access to nodes. If your cluster requires this, enable the `--exclusive` directive:

```yaml
use_exclusive_sbatch_directive: true
```

This adds `#SBATCH --exclusive` to the job script, ensuring your job has sole access to the allocated nodes. This is often required on clusters where GPU jobs must not share nodes with other jobs.
