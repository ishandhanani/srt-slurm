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

        prefix = profiling.get_nsys_prefix("/output/test")
        assert "nsys" in prefix
        assert "profile" in prefix
        # cudaProfilerApi-based capture (application controls start/stop)
        assert "-c" in prefix
        assert "cudaProfilerApi" in prefix
        assert "--capture-range-end=stop" in prefix
        # Trace types
        assert "-t" in prefix
        assert "cuda,nvtx,python-gil" in prefix
        # Node-level CUDA graph tracing for detailed kernel visibility
        assert "--cuda-graph-trace" in prefix
        assert "node" in prefix
        # Output pattern with hostname + PID for multi-node safety
        joined = " ".join(prefix)
        assert "%h" in joined
        assert "%p" in joined

    def test_num_prompts_default(self):
        """Test num_prompts field default value."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(type="nsys", isl=1024, osl=512, concurrency=32)
        assert profiling.num_prompts == 128

    def test_num_prompts_custom(self):
        """Test num_prompts field custom value."""
        from srtctl.core.schema import ProfilingConfig

        profiling = ProfilingConfig(
            type="nsys",
            isl=1024,
            osl=512,
            concurrency=32,
            num_prompts=1024,
        )
        assert profiling.num_prompts == 1024

    def test_env_vars_include_num_prompts_and_type(self):
        """Test that env vars include PROFILE_NUM_PROMPTS and PROFILER_TYPE."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            isl=1024,
            osl=512,
            concurrency=32,
            num_prompts=512,
            prefill=ProfilingPhaseConfig(start_step=10, stop_step=100),
            decode=ProfilingPhaseConfig(start_step=20, stop_step=200),
        )

        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["PROFILE_NUM_PROMPTS"] == "512"
        assert env["PROFILER_TYPE"] == "nsys"

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

    def test_profiling_requires_traffic_params_with_manual_benchmark(self):
        """Profiling with manual benchmark requires isl/osl/concurrency."""
        from marshmallow import ValidationError

        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Missing concurrency should fail when benchmark is manual
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

    def test_profiling_no_traffic_params_with_sa_bench(self):
        """Profiling alongside sa-bench does NOT require traffic params."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # No isl/osl/concurrency in profiling section -- should pass because
        # sa-bench provides the traffic
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
            benchmark=BenchmarkConfig(type="sa-bench", isl=8192, osl=1024, concurrencies=[64]),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=100, stop_step=105),
                decode=ProfilingPhaseConfig(start_step=500, stop_step=505),
            ),
        )
        assert config.profiling.enabled
        assert config.profiling.isl is None  # Not set -- that's fine

    def test_profiling_allows_multi_worker_disagg(self):
        """Profiling now supports multiple workers in disaggregated mode."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Multiple prefill workers should now succeed (constraint removed)
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                prefill_nodes=2,
                decode_nodes=1,
                prefill_workers=2,  # More than 1 - now allowed!
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

    def test_profiling_allows_multi_worker_agg(self):
        """Profiling now supports multiple workers in aggregated mode."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Multiple agg workers should now succeed (constraint removed)
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                agg_nodes=2,
                agg_workers=2,  # More than 1 - now allowed!
            ),
            profiling=ProfilingConfig(
                type="torch",
                isl=1024,
                osl=128,
                concurrency=1,
                aggregated=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )
        assert config.profiling.enabled

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
    """Test that profiling auto-switches benchmark type correctly."""

    def test_profiling_with_manual_uses_profiling_runner(self):
        """When profiling enabled + benchmark=manual, auto-switch to profiling runner."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

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
            benchmark=BenchmarkConfig(type="manual"),
            profiling=ProfilingConfig(
                type="torch",
                isl=1024,
                osl=128,
                concurrency=1,
                prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
                decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
            ),
        )

        assert config.profiling.enabled is True
        assert config.benchmark.type == "manual"

        # Simulate the auto-switch logic from benchmark_stage.py
        benchmark_type = config.benchmark.type
        if config.profiling.enabled and benchmark_type in ("manual", "profiling"):
            benchmark_type = "profiling"

        runner = get_runner(benchmark_type)
        assert runner.name == "Profiling"

    def test_profiling_with_sa_bench_preserves_benchmark(self):
        """When profiling enabled + benchmark=sa-bench, don't override to profiling runner."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

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
            benchmark=BenchmarkConfig(type="sa-bench", isl=8192, osl=1024, concurrencies=[64]),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=100, stop_step=105),
                decode=ProfilingPhaseConfig(start_step=500, stop_step=505),
            ),
        )

        assert config.profiling.enabled is True
        assert config.benchmark.type == "sa-bench"

        # Simulate the auto-switch logic -- sa-bench should NOT be overridden
        benchmark_type = config.benchmark.type
        if config.profiling.enabled and benchmark_type in ("manual", "profiling"):
            benchmark_type = "profiling"

        assert benchmark_type == "sa-bench"
        runner = get_runner(benchmark_type)
        assert runner.name == "SA-Bench"


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


class TestCLIProfilingInjection:
    """Test CLI flag-based profiling injection."""

    @staticmethod
    def _make_args(**kwargs):
        """Create a mock argparse.Namespace with profiling defaults."""
        import argparse

        defaults = {
            "nsys": False,
            "torch_profile": False,
            "profile_start": 100,
            "profile_stop": 105,
            "profile_start_decode": None,
            "profile_stop_decode": None,
            "profile_opt": [],
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_flags_returns_none(self):
        """No profiling flags means no overrides."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args()
        result = build_profiling_overrides(args, {"resources": {}})
        assert result is None

    def test_nsys_disaggregated(self):
        """--nsys with disaggregated resources builds prefill+decode phase configs."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(nsys=True, profile_start=100, profile_stop=105)
        config_data = {"resources": {"prefill_nodes": 1, "decode_nodes": 1}}
        result = build_profiling_overrides(args, config_data)

        assert result is not None
        prof = result["profiling"]
        assert prof["type"] == "nsys"
        assert prof["prefill"] == {"start_step": 100, "stop_step": 105}
        assert prof["decode"] == {"start_step": 100, "stop_step": 105}
        assert "aggregated" not in prof

    def test_nsys_aggregated(self):
        """--nsys with aggregated resources builds aggregated phase config."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(nsys=True)
        config_data = {"resources": {"agg_nodes": 2}}
        result = build_profiling_overrides(args, config_data)

        prof = result["profiling"]
        assert prof["type"] == "nsys"
        assert prof["aggregated"] == {"start_step": 100, "stop_step": 105}
        assert "prefill" not in prof
        assert "decode" not in prof

    def test_different_decode_window(self):
        """--profile-start-decode/stop-decode set different window for decode."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(
            nsys=True,
            profile_start=100,
            profile_stop=105,
            profile_start_decode=500,
            profile_stop_decode=505,
        )
        config_data = {"resources": {"prefill_nodes": 1, "decode_nodes": 1}}
        result = build_profiling_overrides(args, config_data)

        prof = result["profiling"]
        assert prof["prefill"] == {"start_step": 100, "stop_step": 105}
        assert prof["decode"] == {"start_step": 500, "stop_step": 505}

    def test_torch_profile_flag(self):
        """--torch-profile sets type to 'torch'."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(torch_profile=True)
        config_data = {"resources": {"prefill_nodes": 1, "decode_nodes": 1}}
        result = build_profiling_overrides(args, config_data)

        assert result["profiling"]["type"] == "torch"

    def test_nsys_and_torch_mutual_exclusion(self):
        """--nsys and --torch-profile together raises error."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(nsys=True, torch_profile=True)
        with pytest.raises(ValueError, match="Cannot use both"):
            build_profiling_overrides(args, {"resources": {}})

    def test_profile_opt_key_value(self):
        """--profile-opt passes through as typed key=value pairs."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(
            nsys=True,
            profile_opt=["gpu_metrics=true", "num_prompts=512", "isl=8192"],
        )
        config_data = {"resources": {"prefill_nodes": 1, "decode_nodes": 1}}
        result = build_profiling_overrides(args, config_data)

        prof = result["profiling"]
        assert prof["gpu_metrics"] is True
        assert prof["num_prompts"] == 512
        assert prof["isl"] == 8192

    def test_profile_opt_bad_format(self):
        """--profile-opt without = raises error."""
        from srtctl.cli.submit import build_profiling_overrides

        args = self._make_args(nsys=True, profile_opt=["bad_format"])
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            build_profiling_overrides(args, {"resources": {}})


class TestCLIProfilingTRTLLM:
    """Test CLI profiling with TRT-LLM backend."""

    def test_trtllm_nsys_env_vars(self):
        """TRT-LLM nsys profiling sets TLLM_PROFILE_START_STOP env var."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="nsys",
            isl=8192,
            osl=1024,
            concurrency=64,
            prefill=ProfilingPhaseConfig(start_step=100, stop_step=105),
            decode=ProfilingPhaseConfig(start_step=500, stop_step=505),
        )

        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["TLLM_PROFILE_START_STOP"] == "100-105"
        assert env["TLLM_PROFILE_RECORD_GC"] == "1"
        assert env["TLLM_NVTX_DEBUG"] == "1"

        env_decode = profiling.get_env_vars("decode", "/logs/profiles")
        assert env_decode["TLLM_PROFILE_START_STOP"] == "500-505"

    def test_trtllm_profiling_with_sa_bench_config(self):
        """TRT-LLM config with profiling + sa-bench validates successfully."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )
        from srtctl.backends.trtllm import TRTLLMProtocol

        config = SrtConfig(
            name="test-trtllm-profile",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h200",
                prefill_nodes=1,
                decode_nodes=1,
                prefill_workers=1,
                decode_workers=1,
                gpus_per_node=8,
            ),
            backend=TRTLLMProtocol(),
            benchmark=BenchmarkConfig(type="sa-bench", isl=8192, osl=1024, concurrencies=[64]),
            profiling=ProfilingConfig(
                type="nsys",
                prefill=ProfilingPhaseConfig(start_step=100, stop_step=105),
                decode=ProfilingPhaseConfig(start_step=500, stop_step=505),
            ),
        )
        assert config.profiling.enabled
        assert config.profiling.isl is None  # Not required with sa-bench
        assert config.benchmark.type == "sa-bench"


class TestCLIProfilingSGLang:
    """Test CLI profiling with SGLang backend."""

    def test_sglang_profiling_switches_module(self):
        """SGLang backend switches to sglang.launch_server when profiling enabled."""
        from unittest.mock import MagicMock

        from srtctl.backends.sglang import SGLangProtocol, SGLangServerConfig

        backend = SGLangProtocol(
            sglang_config=SGLangServerConfig(
                prefill={"model_path": "/model/", "tensor_parallel_size": 4},
            ),
        )

        process = MagicMock()
        process.endpoint_mode = "prefill"
        process.node = "node1"
        process.http_port = 30000
        process.gpu_indices = [0, 1, 2, 3]
        process.bootstrap_port = None

        endpoint_processes = [process]

        runtime = MagicMock()
        runtime.model_path.name = "test-model"

        # Without profiling -- uses dynamo.sglang
        cmd_normal = backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=runtime,
            frontend_type="dynamo",
            profiling_enabled=False,
        )
        assert "dynamo.sglang" in cmd_normal

        # With profiling -- switches to sglang.launch_server
        cmd_profiled = backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=runtime,
            frontend_type="dynamo",
            profiling_enabled=True,
        )
        assert "sglang.launch_server" in cmd_profiled
        # Should NOT have --disaggregation-mode when profiling
        assert "--disaggregation-mode" not in cmd_profiled

    def test_sglang_torch_profiler_env(self):
        """SGLang torch profiling sets SGLANG_TORCH_PROFILER_DIR."""
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        profiling = ProfilingConfig(
            type="torch",
            isl=1024,
            osl=128,
            concurrency=1,
            prefill=ProfilingPhaseConfig(start_step=0, stop_step=50),
            decode=ProfilingPhaseConfig(start_step=0, stop_step=50),
        )

        env = profiling.get_env_vars("prefill", "/logs/profiles")
        assert env["SGLANG_TORCH_PROFILER_DIR"] == "/logs/profiles/prefill"
        assert "TLLM_PROFILE_START_STOP" not in env

    def test_sglang_nsys_wrapping(self):
        """SGLang nsys profiling prepends nsys prefix to command."""
        from unittest.mock import MagicMock

        from srtctl.backends.sglang import SGLangProtocol

        backend = SGLangProtocol()
        nsys_prefix = ["nsys", "profile", "-o", "/output/profile_%h_%p"]

        process = MagicMock()
        process.endpoint_mode = "agg"
        process.node = "node1"
        process.http_port = 30000
        process.gpu_indices = [0]
        process.bootstrap_port = None

        runtime = MagicMock()
        runtime.model_path.name = "test-model"

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="dynamo",
            profiling_enabled=True,
            nsys_prefix=nsys_prefix,
        )
        assert cmd[0] == "nsys"
        assert cmd[1] == "profile"


class TestBackwardsCompatibility:
    """Verify existing profiling patterns still work after changes."""

    def test_yaml_only_profiling_still_works(self):
        """Existing YAML-only profiling config (with traffic params) still works."""
        from srtctl.core.schema import (
            ModelConfig,
            ProfilingConfig,
            ProfilingPhaseConfig,
            ResourceConfig,
            SrtConfig,
        )

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
                type="nsys",
                isl=8192,
                osl=1024,
                concurrency=64,
                prefill=ProfilingPhaseConfig(start_step=100, stop_step=105),
                decode=ProfilingPhaseConfig(start_step=500, stop_step=505),
            ),
        )
        assert config.profiling.enabled

    def test_no_profiling_unchanged(self):
        """Config without profiling section still works normally."""
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=128, concurrencies=[32]),
        )
        assert config.profiling.enabled is False
        assert config.benchmark.type == "sa-bench"

    def test_dedicated_profiling_runner_still_works(self):
        """ProfilingRunner is still available and works with manual benchmark."""
        runner = get_runner("profiling")
        assert runner.name == "Profiling"
        assert "profiling" in runner.script_path

