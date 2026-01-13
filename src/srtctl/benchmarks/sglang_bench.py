# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang benchmark runner using sglang.bench_serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("sglang-bench")
class SGLangBenchRunner(BenchmarkRunner):
    """SGLang benchmark runner.

    Uses sglang.bench_serving to generate traffic. Supports profiling when
    profiling.type is set to "torch" or "nsys".

    Required config fields (in benchmark section):
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "128x256")

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
    """

    @property
    def name(self) -> str:
        return "SGLang-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sglang-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sglang-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for sglang-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for sglang-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for sglang-bench")

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
        ]
