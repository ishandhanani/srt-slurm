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
        assert "custom" in benchmarks

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


class TestCustomBenchmarkRunner:
    """Test custom benchmark runner."""

    def test_custom_registered(self):
        """Custom benchmark is in registry."""
        benchmarks = list_benchmarks()
        assert "custom" in benchmarks

    def test_get_custom_runner(self):
        """Can get custom benchmark runner."""
        runner = get_runner("custom")
        assert runner.name == "Custom Benchmark"
        assert runner.script_path == ""  # Custom doesn't use script path

    def test_validate_requires_command(self):
        """Validates that command is required for custom type."""
        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="custom"),  # No command
        )
        errors = runner.validate_config(config)
        assert any("command" in e for e in errors)

    def test_validate_valid_config(self):
        """Valid custom config passes validation."""
        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="echo hello"),
            ),
        )
        errors = runner.validate_config(config)
        assert errors == []

    def test_build_command_simple(self):
        """Build command returns simple command as list."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="echo hello world"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.format_string.return_value = "echo hello world"
        mock_runtime.frontend_port = 8000

        cmd = runner.build_command(config, mock_runtime)
        assert cmd == ["echo", "hello", "world"]

    def test_build_command_expands_placeholders(self):
        """Command template placeholders are expanded."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="curl {nginx_url}/v1/health"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        # Simulate what format_string does with nginx_url kwarg
        mock_runtime.format_string.return_value = "curl http://localhost:8000/v1/health"

        cmd = runner.build_command(config, mock_runtime)
        assert cmd == ["curl", "http://localhost:8000/v1/health"]

    def test_build_command_multiline(self):
        """Multiline commands are handled correctly."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="uvx aiperf profile --url {nginx_url} --concurrency 128"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "uvx aiperf profile --url http://localhost:8000 --concurrency 128"

        cmd = runner.build_command(config, mock_runtime)
        assert cmd[0] == "uvx"
        assert "aiperf" in cmd
        assert "--concurrency" in cmd
        assert "128" in cmd

    def test_custom_container_image_in_benchmark_config(self):
        """Custom container_image can be specified in benchmark config."""
        from srtctl.core.formatting import FormattablePath, FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="echo test"),
                container_image=FormattablePath(template="/custom/container.sqsh"),
            ),
        )

        assert config.benchmark.container_image is not None
        assert config.benchmark.container_image.template == "/custom/container.sqsh"

    def test_build_command_with_shell_operators_uses_bash(self):
        """Commands with shell operators (&&, ||, |, etc.) use bash -c."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="pip install foo && python run.py"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "pip install foo && python run.py"

        cmd = runner.build_command(config, mock_runtime)
        # Should wrap in bash -c for proper shell interpretation
        assert cmd == ["bash", "-c", "pip install foo && python run.py"]

    def test_build_command_with_pipe_uses_bash(self):
        """Commands with pipe operator use bash -c."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="cat file.txt | grep pattern"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "cat file.txt | grep pattern"

        cmd = runner.build_command(config, mock_runtime)
        assert cmd == ["bash", "-c", "cat file.txt | grep pattern"]

    def test_build_command_with_semicolon_uses_bash(self):
        """Commands with semicolon use bash -c."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="cd /app; python main.py"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "cd /app; python main.py"

        cmd = runner.build_command(config, mock_runtime)
        assert cmd == ["bash", "-c", "cd /app; python main.py"]

    def test_build_command_with_command_substitution_uses_bash(self):
        """Commands with $(...) substitution use bash -c."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="echo $(date)"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "echo $(date)"

        cmd = runner.build_command(config, mock_runtime)
        assert cmd == ["bash", "-c", "echo $(date)"]

    def test_build_command_without_shell_operators_uses_shlex(self):
        """Simple commands without shell operators use shlex parsing."""
        from unittest.mock import MagicMock

        from srtctl.benchmarks.custom import CustomBenchmarkRunner
        from srtctl.core.formatting import FormattableString
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = CustomBenchmarkRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="custom",
                command=FormattableString(template="python script.py --arg value"),
            ),
        )

        mock_runtime = MagicMock()
        mock_runtime.frontend_port = 8000
        mock_runtime.format_string.return_value = "python script.py --arg value"

        cmd = runner.build_command(config, mock_runtime)
        # Should NOT wrap in bash -c, use shlex split
        assert cmd == ["python", "script.py", "--arg", "value"]

