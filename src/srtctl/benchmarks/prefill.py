# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill benchmark runner.

Measures prefill throughput using trace-replay on the together-ai-basic-no-delays_1osl
dataset (long ISL, very short OSL ~1 token). Uses aiperf duration-based mode with
disjoint session splits per concurrency to avoid KV-cache reuse across runs.

Dataset must be pre-split before running: run_together_aiperf.sh --split-dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

# Container mount point for split dataset files
DATASET_MOUNT_PATH = Path("/prefill-data")


@register_benchmark("prefill")
class PrefillRunner(AIPerfBenchmarkRunner):
    """Prefill throughput benchmark via trace-replay on together-ai dataset.

    Uses aiperf duration-based mode with disjoint session splits per concurrency.
    Dataset must be pre-split: run_together_aiperf.sh --split-dataset beforehand.
    """

    @property
    def name(self) -> str:
        return "Prefill"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/prefill/bench.sh"

    def _resolve_dataset_path(self, dataset_dir: str) -> Path:
        p = Path(dataset_dir)
        return p if p.is_absolute() else Path.cwd() / p

    def get_extra_mounts(self, config: SrtConfig) -> dict[Path, Path]:
        if config.benchmark.prefill_dataset_dir is None:
            return {}
        host_path = self._resolve_dataset_path(config.benchmark.prefill_dataset_dir).resolve()
        return {host_path: DATASET_MOUNT_PATH}

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark
        if b.prefill_concurrencies is None:
            errors.append("benchmark.prefill_concurrencies is required for prefill benchmark")
        if b.prefill_dataset_dir is None:
            errors.append("benchmark.prefill_dataset_dir is required for prefill benchmark")
        else:
            ds_path = self._resolve_dataset_path(b.prefill_dataset_dir)
            if not ds_path.exists():
                errors.append(f"benchmark.prefill_dataset_dir not found: {ds_path}")
            else:
                # Check that split files exist for each requested concurrency
                concs = b.prefill_concurrencies or []
                for c in concs:
                    if not (ds_path / f"conc_{c}.jsonl").exists():
                        errors.append(
                            f"Missing split file: {ds_path}/conc_{c}.jsonl — run --split-dataset first"
                        )
        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        b = config.benchmark
        r = config.resources
        endpoint = f"http://localhost:{runtime.frontend_port}"
        concurrencies = "x".join(str(c) for c in (b.prefill_concurrencies or []))
        if r.is_disaggregated:
            total_gpus = r.prefill_gpus + r.decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node

        return [
            "bash",
            self.script_path,
            endpoint,
            config.served_model_name,
            str(DATASET_MOUNT_PATH),
            concurrencies,
            str(total_gpus),
        ]
