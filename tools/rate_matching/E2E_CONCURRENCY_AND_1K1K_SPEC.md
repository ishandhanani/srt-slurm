# E2E Concurrency Formula and 1k1k MTP-Only Sweep Spec

## E2E Concurrency (Final)

**Anchor:** Per-worker concurrency from SOL (c8, c32, c64, etc.) — keeps each Pareto point meaningful.

**Two variants per Pareto point:**
1. **1.0x (baseline):** `system_conc = per_worker_conc × gen_instances`
2. **1.05x (headroom):** `system_conc = int(per_worker_conc × gen_instances × 1.05)`

Example (MTP-1 Pareto):

| Pareto Point   | gen | pw_conc | 1.0x sys_conc | 1.05x sys_conc |
|----------------|-----|---------|---------------|----------------|
| c8  TEP 1P6D   | 6   | 8       | 48            | 50             |
| c32 TEP 2P5D   | 5   | 32      | 160           | 168            |
| c64 DEP 2P3D   | 3   | 64      | 192           | 201            |
| c128 DEP 4P3D   | 3   | 128     | 384           | 403            |
| c256 DEP 2P1D   | 1   | 256     | 256           | 268            |

## Job Count (1k1k MTP-Only)

- **CTX-only SOL:** 1
- **GEN-only SOL:** 16 (8 per MTP level × 2 levels: MTP-1, MTP-3)
- **E2E validation:** ~20 (~10 Pareto points × 2 concurrency variants)

**Total: ~37 SLURM jobs**

## Implementation Checklist

1. **Schema:** Add `E2EValidationSettings` with `e2e_concurrency_variants: ["1.0", "1.05"]` (or similar); optional pass/fail tolerances.
2. **generate_configs.py:** For each Pareto point, emit one E2E config per variant (1.0x and 1.05x system concurrency).
3. **run_sweep.py:** Phase 6 submits and processes both E2E jobs per Pareto point; store results with a `concurrency_variant` key.
4. **Dashboard/status:** Show both variants per point (e.g. table columns or separate rows); optionally highlight best (e.g. by throughput meeting TTFT).
5. **1k1k sweep YAML:** MTP-only gen_sweep (MTP-1 and MTP-3), workload 1k/1k; no STP in this sweep.
6. **Robustness:** Add pass/fail checks (TPOT tolerance, throughput tolerance, TTFT constraint) and surface in report.
