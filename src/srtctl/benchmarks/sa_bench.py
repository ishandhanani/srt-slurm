# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench throughput/latency benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("sa-bench")
class SABenchRunner(BenchmarkRunner):
    """SA-Bench throughput and latency benchmark.

    Tests serving throughput at various concurrency levels.

    Required config fields:
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "4x8x16x32")

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
    """

    @property
    def name(self) -> str:
        return "SA-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sa-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sa-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for sa-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for sa-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for sa-bench")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrencies) if concurrencies else "",
            str(b.req_rate) if b.req_rate else "inf",
            config.model.path,
        ]
