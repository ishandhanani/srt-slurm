# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiperf throughput/latency benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("aiperf")
class AiperfRunner(AIPerfBenchmarkRunner):
    """Aiperf throughput and latency benchmark.

    Tests serving throughput at various concurrency levels using the aiperf profiler.
    Supports both closed-loop (concurrency-based) and open-loop (request-rate) modes.

    Required config fields:
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "4x8x16x32")

    Optional:
        - benchmark.req_rate: Request rate for open-loop mode (poisson arrival).
          If not set, runs in closed-loop mode (concurrency-based).
    """

    @property
    def name(self) -> str:
        return "Aiperf"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/aiperf/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "aiperf")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for aiperf")
        if b.osl is None:
            errors.append("benchmark.osl is required for aiperf")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for aiperf")
        # req_rate is optional - if not set, runs in closed-loop mode

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        r = config.resources
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        # Compute GPU info for result filename
        is_disaggregated = r.is_disaggregated
        if is_disaggregated:
            prefill_gpus = r.prefill_gpus
            decode_gpus = r.decode_gpus
            total_gpus = prefill_gpus + decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node
            prefill_gpus = 0
            decode_gpus = 0

        # For aiperf, empty req_rate means closed-loop mode
        # Treat "inf" as closed-loop for backwards compatibility with sa-bench configs
        req_rate = b.req_rate
        if req_rate == "inf":
            req_rate = None

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrencies) if concurrencies else "",
            str(req_rate) if req_rate else "",  # Empty string means closed-loop
            config.model.path,
            config.served_model_name,
            str(is_disaggregated).lower(),
            str(total_gpus),
            str(prefill_gpus),
            str(decode_gpus),
        ]

