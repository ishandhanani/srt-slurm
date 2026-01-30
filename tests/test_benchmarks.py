# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark runners."""

import pytest

from srtctl.benchmarks import get_runner, list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestBenchmarkRegistry:
    """Test benchmark runner registry."""

    def test_list_benchmarks(self):
        """All expected benchmarks are registered."""
        benchmarks = list_benchmarks()
        assert "sa-bench" in benchmarks
        assert "mmlu" in benchmarks
        assert "gpqa" in benchmarks
        assert "longbenchv2" in benchmarks
        assert "router" in benchmarks
        assert "profiling" in benchmarks

    def test_get_runner_valid(self):
        """Can get runner for valid benchmark type."""
        runner = get_runner("sa-bench")
        assert runner.name == "SA-Bench"
        assert "sa-bench" in runner.script_path

    def test_get_runner_invalid(self):
        """Raises ValueError for unknown benchmark type."""
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            get_runner("nonexistent-benchmark")


class TestSABenchRunner:
    """Test SA-Bench runner."""

    def test_validate_config_missing_isl(self):
        """Validates that isl is required."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert any("isl" in e for e in errors)

    def test_validate_config_valid(self):
        """Valid config passes validation."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench", isl=1024, osl=1024, concurrencies="4x8"
            ),
        )
        errors = runner.validate_config(config)
        assert errors == []


class TestMooncakeRouterRunner:
    """Test Mooncake Router benchmark runner."""

    def test_build_command_includes_tokenizer_path(self):
        """Build command passes tokenizer path to aiperf.

        This fixes a bug where aiperf couldn't load the tokenizer because it was
        using the served model name (e.g., "Qwen/Qwen3-32B") to find the tokenizer,
        but that's not a valid HuggingFace ID or local path. The fix passes
        --tokenizer /model explicitly since the model is mounted there.
        """
        from unittest.mock import MagicMock

        from srtctl.benchmarks.mooncake_router import MooncakeRouterRunner

        runner = MooncakeRouterRunner()

        config = MagicMock()
        config.benchmark = MagicMock()
        config.benchmark.mooncake_workload = "conversation"
        config.benchmark.ttft_threshold_ms = 2000
        config.benchmark.itl_threshold_ms = 25
        config.served_model_name = "Qwen/Qwen3-32B"

        runtime = MagicMock()
        runtime.frontend_port = 8000
        runtime.is_hf_model = False  # Local model mounted at /model

        cmd = runner.build_command(config, runtime)

        # Command: bash, script, endpoint, model_name, workload, ttft, itl, tokenizer_path
        assert cmd[7] == "/model"  # tokenizer path


class TestScriptsExist:
    """Test that benchmark scripts exist."""

    def test_scripts_dir_exists(self):
        """Scripts directory exists."""
        assert SCRIPTS_DIR.exists()

    def test_sa_bench_script_exists(self):
        """SA-Bench script exists."""
        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        assert script.exists()

    def test_mmlu_script_exists(self):
        """MMLU script exists."""
        script = SCRIPTS_DIR / "mmlu" / "bench.sh"
        assert script.exists()

