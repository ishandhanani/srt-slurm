# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling configuration, validation, and benchmark runner."""

import pytest

from srtctl.benchmarks import get_runner
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestProfilingConfig:
    """Tests for ProfilingConfig dataclass."""

    def test_profiling_defaults(self):
        """Test profiling config defaults."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig()

        assert profiling.enabled is False
        assert profiling.is_nsys is False
        assert profiling.is_torch is False
        assert profiling.type == "none"

    def test_nsys_profiling(self):
        """Test nsys profiling configuration."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            isl=1024,
            osl=512,
            concurrency=32,
        )

        assert profiling.enabled is True
        assert profiling.is_nsys is True
        assert profiling.is_torch is False

        # Test nsys prefix generation
        prefix = profiling.get_nsys_prefix("/output/test")
        assert "nsys" in prefix
        assert "profile" in prefix
        assert "/output/test" in prefix

    def test_torch_profiling(self):
        """Test torch profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            isl=2048,
            osl=1024,
            concurrency=64,
            prefill=ProfilingPhaseConfig(start_step=5, stop_step=15),
            decode=ProfilingPhaseConfig(start_step=10, stop_step=20),
        )

        assert profiling.enabled is True
        assert profiling.is_torch is True
        assert profiling.is_nsys is False

        # Test env vars generation for prefill
        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["PROFILING_MODE"] == "prefill"
        assert env["PROFILE_ISL"] == "2048"
        assert env["PROFILE_OSL"] == "1024"
        assert env["PROFILE_CONCURRENCY"] == "64"
        assert env["PROFILE_PREFILL_START_STEP"] == "5"
        assert env["PROFILE_PREFILL_STOP_STEP"] == "15"
        assert env["SGLANG_TORCH_PROFILER_DIR"] == "/logs/profiles/prefill"

        # Test env vars generation for decode (different steps)
        env_decode = profiling.get_env_vars("decode", "/logs/profiles")
        assert env_decode["PROFILE_DECODE_START_STEP"] == "10"
        assert env_decode["PROFILE_DECODE_STOP_STEP"] == "20"

    def test_aggregated_profiling(self):
        """Test aggregated profiling configuration."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            isl=1024,
            osl=512,
            concurrency=32,
            aggregated=ProfilingPhaseConfig(start_step=0, stop_step=100),
        )

        env = profiling.get_env_vars("agg", "/logs/profiles")
        assert env["PROFILE_AGG_START_STEP"] == "0"
        assert env["PROFILE_AGG_STOP_STEP"] == "100"


class TestProfilingValidation:
    """Tests for profiling config validation in SrtConfig."""

    def test_disagg_requires_prefill_and_decode(self):
        """Disaggregated mode requires both prefill and decode profiling configs."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Missing decode config should fail (with valid single worker config)
        with pytest.raises(ValidationError, match="both profiling.prefill and profiling.decode"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=1,
                    decode_workers=1,
                ),
                profiling=ProfilingConfig(
                    type="torch",
                    isl=1024,
                    osl=128,
                    concurrency=1,
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                    # Missing decode config
                ),
            )

    def test_agg_requires_aggregated_config(self):
        """Aggregated mode requires aggregated profiling config."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Aggregated mode without aggregated profiling config should fail
        with pytest.raises(ValidationError, match="profiling.aggregated to be set"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                profiling=ProfilingConfig(
                    type="torch",
                    isl=1024,
                    osl=128,
                    concurrency=1,
                    # Missing aggregated config
                ),
            )

    def test_profiling_requires_traffic_params(self):
        """Profiling requires isl/osl/concurrency."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Missing concurrency should fail
        with pytest.raises(ValidationError, match="isl/osl/concurrency must be set"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", prefill_nodes=1, decode_nodes=1),
                profiling=ProfilingConfig(
                    type="torch",
                    isl=1024,
                    osl=128,
                    # Missing concurrency
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                    decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
                ),
            )

    def test_profiling_requires_single_worker_disagg(self):
        """Profiling in disaggregated mode requires exactly 1P + 1D."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Multiple prefill workers should fail
        with pytest.raises(ValidationError, match="exactly 1 prefill and 1 decode"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    prefill_nodes=1,
                    decode_nodes=1,
                    prefill_workers=2,  # More than 1!
                    decode_workers=1,
                ),
                profiling=ProfilingConfig(
                    type="torch",
                    isl=1024,
                    osl=128,
                    concurrency=1,
                    prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                    decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
                ),
            )

    def test_profiling_requires_single_worker_agg(self):
        """Profiling in aggregated mode requires exactly 1 agg worker."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Multiple agg workers should fail
        with pytest.raises(ValidationError, match="exactly 1 aggregated worker"):
            SrtConfig(
                name="test",
                model=ModelConfig(path="/model", container="/container", precision="fp8"),
                resources=ResourceConfig(
                    gpu_type="h100",
                    agg_nodes=2,
                    agg_workers=2,  # More than 1!
                ),
                profiling=ProfilingConfig(
                    type="torch",
                    isl=1024,
                    osl=128,
                    concurrency=1,
                    aggregated=ProfilingPhaseConfig(start_step=0, stop_step=50),
                ),
            )

    def test_valid_profiling_config_disagg(self):
        """Valid profiling config with 1P + 1D passes validation."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Should not raise
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            profiling=ProfilingConfig(
                type="torch",
                isl=1024,
                osl=128,
                concurrency=1,
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )
        assert config.profiling.enabled


class TestProfilingAutoSwitch:
    """Test that profiling auto-switches benchmark type."""

    def test_profiling_enabled_overrides_benchmark_type(self):
        """When profiling is enabled, benchmark type should be treated as 'profiling'."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # User sets benchmark.type to "manual" but has profiling enabled
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
            ),
            benchmark=BenchmarkConfig(type="manual"),  # User says manual
            profiling=ProfilingConfig(
                type="torch",
                isl=1024,
                osl=128,
                concurrency=1,
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

        # The orchestrator should detect profiling.enabled and use "profiling" runner
        assert config.profiling.enabled is True
        assert config.benchmark.type == "manual"  # Original value unchanged

        # Simulate the auto-switch logic from do_sweep.py
        benchmark_type = config.benchmark.type
        if config.profiling.enabled:
            benchmark_type = "profiling"

        # Verify the profiling runner can be retrieved
        runner = get_runner(benchmark_type)
        assert runner.name == "Profiling"


class TestProfilingRunner:
    """Test Profiling benchmark runner."""

    def test_get_profiling_runner(self):
        """Can get profiling runner."""
        runner = get_runner("profiling")
        assert runner.name == "Profiling"
        assert "profiling" in runner.script_path

    def test_validate_config_requires_profiling_enabled(self):
        """Validates that profiling must be enabled."""
        from srtctl.benchmarks.profiling import ProfilingRunner
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = ProfilingRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            profiling=ProfilingConfig(type="none"),  # Not enabled
        )
        errors = runner.validate_config(config)
        assert any("torch" in e or "nsys" in e for e in errors)

    def test_validate_config_requires_params(self):
        """Validates that isl/osl/concurrency are required."""
        from srtctl.benchmarks.profiling import ProfilingRunner
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = ProfilingRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            profiling=ProfilingConfig(type="none", isl=None, osl=None, concurrency=None),
        )
        errors = runner.validate_config(config)
        assert any("isl" in e for e in errors)
        assert any("osl" in e for e in errors)
        assert any("concurrency" in e for e in errors)

    def test_profiling_script_exists(self):
        """Profiling script exists."""
        script = SCRIPTS_DIR / "profiling" / "profile.sh"
        assert script.exists()
