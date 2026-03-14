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


class TestPrefillRunner:
    """Test prefill benchmark runner."""

    def test_benchmark_type_exists(self):
        """BenchmarkType.PREFILL is registered."""
        from srtctl.core.schema import BenchmarkType

        assert BenchmarkType.PREFILL == "prefill"

    def test_prefill_registered(self):
        """Prefill runner is registered."""
        benchmarks = list_benchmarks()
        assert "prefill" in benchmarks

    def test_validate_config_missing_fields(self):
        """Validates that prefill_concurrencies and prefill_dataset_dir are required."""
        from srtctl.benchmarks.prefill import PrefillRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = PrefillRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(type="prefill"),
        )
        errors = runner.validate_config(config)
        assert any("prefill_concurrencies" in e for e in errors)
        assert any("prefill_dataset_dir" in e for e in errors)

    def test_validate_config_missing_dataset_dir(self):
        """Validates that prefill_dataset_dir is required even when concurrencies are set."""
        from srtctl.benchmarks.prefill import PrefillRunner
        from srtctl.core.schema import BenchmarkConfig, ModelConfig, ResourceConfig, SrtConfig

        runner = PrefillRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200"),
            benchmark=BenchmarkConfig(type="prefill", prefill_concurrencies=[1, 10, 25]),
        )
        errors = runner.validate_config(config)
        assert any("prefill_dataset_dir" in e for e in errors)

    def test_recipe_loads(self):
        """tep8_egl_decode_only2 recipe parses without error."""
        from pathlib import Path

        from srtctl.core.schema import SrtConfig

        recipe = Path(__file__).parent.parent / "recipes" / "sprint" / "tep8_egl_decode_only2.yaml"
        config = SrtConfig.from_yaml(recipe)
        assert config.name == "tep8_egl_decode_only2"
        assert config.benchmark.type == "prefill"
        assert config.benchmark.prefill_concurrencies == [1, 2, 4, 8, 16, 32, 64]
        assert config.benchmark.prefill_dataset_dir == "together-ai-basic-no-delays_1osl/splits"


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

