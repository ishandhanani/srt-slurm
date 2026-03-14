# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runners for srtctl."""

# Import runners to trigger registration
from srtctl.benchmarks import (
    aiperf,
    gpqa,
    longbenchv2,
    mmlu,
    mooncake_router,
    prefill,
    profiling,
    router,
    sa_bench,
    trace_replay,
)
from srtctl.benchmarks.base import (
    AIPerfBenchmarkRunner,
    BenchmarkRunner,
    get_runner,
    list_benchmarks,
    register_benchmark,
)

__all__ = [
    "AIPerfBenchmarkRunner",
    "BenchmarkRunner",
    "get_runner",
    "list_benchmarks",
    "register_benchmark",
    # Runners
    "sa_bench",
    "mmlu",
    "gpqa",
    "longbenchv2",
    "router",
    "mooncake_router",
    "profiling",
    "aiperf",
    "trace_replay",
    "prefill",
]
