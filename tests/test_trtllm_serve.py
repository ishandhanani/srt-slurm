# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trtllm-serve frontend support."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.backends.trtllm import TRTLLMProtocol
from srtctl.core.health import check_trtllm_serve_health, wait_for_model
from srtctl.core.topology import Process
from srtctl.frontends import TRTLLMServeFrontend, get_frontend


# ============================================================================
# Helper fixtures
# ============================================================================


@dataclass
class MockTopology:
    """Mock FrontendTopology for testing."""

    frontend_nodes: list[str]
    frontend_port: int = 8080


@dataclass
class MockFrontendConfig:
    """Mock FrontendConfig for testing."""

    type: str = "trtllm-serve"
    args: dict | None = None
    env: dict | None = None


@dataclass
class MockResourceConfig:
    """Mock ResourceConfig for testing."""

    num_prefill: int = 0
    num_decode: int = 0
    num_agg: int = 0


@dataclass
class MockConfig:
    """Mock SrtConfig for testing."""

    frontend: MockFrontendConfig
    resources: MockResourceConfig


def make_process(
    node: str,
    mode: str,
    index: int,
    http_port: int = 30000,
    node_rank: int = 0,
) -> Process:
    """Create a Process for testing."""
    return Process(
        node=node,
        gpu_indices=frozenset(range(8)),
        sys_port=8081 + index,
        http_port=http_port,
        endpoint_mode=mode,
        endpoint_index=index,
        node_rank=node_rank,
    )


# ============================================================================
# Worker Command Tests
# ============================================================================


class TestBuildWorkerCommandTRTLLMServe:
    """Test build_worker_command() with frontend_type='trtllm-serve'."""

    def test_trtllm_serve_command_format(self, tmp_path):
        """trtllm-serve path produces correct command structure."""
        backend = TRTLLMProtocol()
        process = make_process("node1", "prefill", 0, http_port=30000)

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.model_path = Path("/models/my-model")

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="trtllm-serve",
        )

        assert cmd[0] == "trtllm-llmapi-launch"
        assert cmd[1] == "trtllm-serve"
        assert "/model" in cmd[2]  # positional model arg
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "30000" in cmd
        assert "--backend" in cmd
        assert "pytorch" in cmd
        assert "--config" in cmd

    def test_trtllm_serve_no_dynamo_args(self, tmp_path):
        """trtllm-serve path does NOT contain dynamo-specific args."""
        backend = TRTLLMProtocol()
        process = make_process("node1", "decode", 0)

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.model_path = Path("/models/my-model")

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="trtllm-serve",
        )

        cmd_str = " ".join(cmd)
        assert "dynamo.trtllm" not in cmd_str
        assert "--request-plane" not in cmd_str
        assert "--disaggregation-mode" not in cmd_str
        assert "--served-model-name" not in cmd_str

    def test_trtllm_serve_config_per_endpoint(self, tmp_path):
        """Config filename includes mode and endpoint index."""
        backend = TRTLLMProtocol()
        process = make_process("node1", "prefill", 2, http_port=30000)

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.model_path = Path("/models/my-model")

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="trtllm-serve",
        )

        # Config path should include mode and index
        config_arg_idx = cmd.index("--config") + 1
        assert "trtllm_worker_config_prefill_2.yaml" in cmd[config_arg_idx]

        # File should be written to host log_dir
        assert (tmp_path / "trtllm_worker_config_prefill_2.yaml").exists()

    def test_trtllm_serve_writes_config_yaml(self, tmp_path):
        """Config is written as valid YAML."""
        backend = TRTLLMProtocol(
            trtllm_config=MagicMock(
                prefill={"mem-fraction-static": 0.8},
                decode=None,
                aggregated=None,
            )
        )
        process = make_process("node1", "prefill", 0, http_port=30000)

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.model_path = Path("/models/my-model")

        backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="trtllm-serve",
        )

        config_path = tmp_path / "trtllm_worker_config_prefill_0.yaml"
        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded["mem-fraction-static"] == 0.8


class TestBuildWorkerCommandDynamoUnchanged:
    """Regression: dynamo path should be unchanged."""

    def test_dynamo_command_format(self, tmp_path):
        """Dynamo path still produces dynamo.trtllm command."""
        backend = TRTLLMProtocol()
        process = make_process("node1", "prefill", 0)

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.model_path = Path("/models/my-model")

        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
            frontend_type="dynamo",
        )

        assert "dynamo.trtllm" in cmd
        assert "--request-plane" in cmd
        assert "nats" in cmd
        assert "--disaggregation-mode" in cmd
        assert "--model-path" in cmd


# ============================================================================
# Disagg Config Generation Tests
# ============================================================================


class TestDisaggConfigGeneration:
    """Test disagg_config.yaml generation in TRTLLMServeFrontend."""

    @patch("srtctl.frontends.trtllm.start_srun_process")
    @patch("srtctl.frontends.trtllm.get_hostname_ip")
    def test_disagg_config_has_correct_urls(self, mock_get_ip, mock_srun, tmp_path):
        """Generated config has correct context/generation server URLs."""
        mock_get_ip.side_effect = lambda node: {"node1": "10.0.0.1", "node2": "10.0.0.2", "node3": "10.0.0.3"}[node]
        mock_srun.return_value = MagicMock()

        frontend = TRTLLMServeFrontend()
        topology = MockTopology(frontend_nodes=["node0"], frontend_port=8080)
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_prefill=1, num_decode=2),
        )

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            make_process("node1", "prefill", 0, http_port=30000),
            make_process("node2", "decode", 0, http_port=30000),
            make_process("node3", "decode", 1, http_port=31000),
        ]

        frontend.start_frontends(topology, runtime, config, MagicMock(), processes)

        # Read generated config
        config_path = tmp_path / "disagg_config.yaml"
        assert config_path.exists()
        disagg = yaml.safe_load(config_path.read_text())

        assert disagg["hostname"] == "0.0.0.0"
        assert disagg["port"] == 8080
        assert disagg["backend"] == "pytorch"
        assert disagg["context_servers"]["num_instances"] == 1
        assert disagg["context_servers"]["urls"] == ["10.0.0.1:30000"]
        assert disagg["generation_servers"]["num_instances"] == 2
        assert set(disagg["generation_servers"]["urls"]) == {"10.0.0.2:30000", "10.0.0.3:31000"}

    @patch("srtctl.frontends.trtllm.start_srun_process")
    @patch("srtctl.frontends.trtllm.get_hostname_ip")
    def test_non_leader_processes_skipped(self, mock_get_ip, mock_srun, tmp_path):
        """Only leader processes contribute URLs to config."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = TRTLLMServeFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_prefill=1, num_decode=1),
        )

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            make_process("node1", "prefill", 0, http_port=30000, node_rank=0),  # leader
            make_process("node1", "prefill", 0, http_port=0, node_rank=1),  # non-leader
            make_process("node2", "decode", 0, http_port=30000, node_rank=0),  # leader
        ]

        frontend.start_frontends(topology, runtime, config, MagicMock(), processes)

        disagg = yaml.safe_load((tmp_path / "disagg_config.yaml").read_text())
        assert disagg["context_servers"]["num_instances"] == 1
        assert disagg["generation_servers"]["num_instances"] == 1

    @patch("srtctl.frontends.trtllm.start_srun_process")
    @patch("srtctl.frontends.trtllm.get_hostname_ip")
    def test_url_format_no_http_prefix(self, mock_get_ip, mock_srun, tmp_path):
        """URLs use IP:port format without http:// prefix."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = TRTLLMServeFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_prefill=1),
        )

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [make_process("node1", "prefill", 0, http_port=30000)]

        frontend.start_frontends(topology, runtime, config, MagicMock(), processes)

        disagg = yaml.safe_load((tmp_path / "disagg_config.yaml").read_text())
        url = disagg["context_servers"]["urls"][0]
        assert not url.startswith("http://")
        assert url == "10.0.0.1:30000"


# ============================================================================
# Frontend command tests
# ============================================================================


class TestTRTLLMServeFrontendCommand:
    """Test the trtllm-serve disaggregated command construction."""

    @patch("srtctl.frontends.trtllm.start_srun_process")
    @patch("srtctl.frontends.trtllm.get_hostname_ip")
    def test_command_structure(self, mock_get_ip, mock_srun, tmp_path):
        """Command uses trtllm-serve disaggregated -c <config>."""
        mock_get_ip.return_value = "10.0.0.1"
        mock_srun.return_value = MagicMock()

        frontend = TRTLLMServeFrontend()
        topology = MockTopology(frontend_nodes=["node0"])
        config = MockConfig(
            frontend=MockFrontendConfig(),
            resources=MockResourceConfig(num_prefill=1, num_decode=1),
        )

        runtime = MagicMock()
        runtime.log_dir = tmp_path
        runtime.container_image = "/container.sqsh"
        runtime.container_mounts = {}

        processes = [
            make_process("node1", "prefill", 0),
            make_process("node2", "decode", 0),
        ]

        frontend.start_frontends(topology, runtime, config, MagicMock(), processes)

        call_args = mock_srun.call_args
        cmd = call_args.kwargs["command"]

        assert cmd[0] == "trtllm-serve"
        assert cmd[1] == "disaggregated"
        assert "-c" in cmd
        assert "/logs/disagg_config.yaml" in cmd


# ============================================================================
# get_frontend() Tests
# ============================================================================


class TestGetFrontendTRTLLMServe:
    """Test frontend factory returns TRTLLMServeFrontend."""

    def test_get_trtllm_serve_frontend(self):
        """get_frontend('trtllm-serve') returns TRTLLMServeFrontend."""
        frontend = get_frontend("trtllm-serve")
        assert isinstance(frontend, TRTLLMServeFrontend)
        assert frontend.type == "trtllm-serve"

    def test_trtllm_serve_health_endpoint(self):
        """TRTLLMServeFrontend uses /health endpoint."""
        frontend = TRTLLMServeFrontend()
        assert frontend.health_endpoint == "/health"


# ============================================================================
# Health Check Tests
# ============================================================================


class TestCheckTRTLLMServeHealth:
    """Test check_trtllm_serve_health() function."""

    def test_returns_ready(self):
        """Always returns ready=True (called on HTTP 200)."""
        result = check_trtllm_serve_health({}, expected_prefill=2, expected_decode=4)

        assert result.ready is True
        assert result.prefill_ready == 2
        assert result.prefill_expected == 2
        assert result.decode_ready == 4
        assert result.decode_expected == 4
        assert "trtllm-serve" in result.message

    def test_ready_with_empty_json(self):
        """Works with empty JSON response."""
        result = check_trtllm_serve_health({}, expected_prefill=0, expected_decode=1)
        assert result.ready is True

    def test_ready_with_arbitrary_json(self):
        """Works with arbitrary JSON content."""
        result = check_trtllm_serve_health(
            {"status": "ok", "some_field": 123},
            expected_prefill=1,
            expected_decode=2,
        )
        assert result.ready is True


class TestWaitForModelTRTLLMServe:
    """Test wait_for_model() with frontend_type='trtllm-serve'."""

    @patch("srtctl.core.health.requests.get")
    def test_http_200_returns_true(self, mock_get):
        """HTTP 200 from /health means ready (no JSON parsing)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = wait_for_model(
            host="10.0.0.1",
            port=8000,
            n_prefill=2,
            n_decode=4,
            frontend_type="trtllm-serve",
            timeout=5.0,
        )

        assert result is True
        # Should have called /health endpoint
        mock_get.assert_called_with("http://10.0.0.1:8000/health", timeout=5.0)

    @patch("srtctl.core.health.requests.get")
    def test_non_200_keeps_polling(self, mock_get):
        """Non-200 responses cause continued polling until timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = wait_for_model(
            host="10.0.0.1",
            port=8000,
            n_prefill=1,
            n_decode=1,
            frontend_type="trtllm-serve",
            timeout=2.0,
            poll_interval=0.5,
        )

        assert result is False


# ============================================================================
# Infrastructure Skip Tests
# ============================================================================


class TestInfraSkip:
    """Test that infrastructure is only needed for dynamo frontend."""

    def test_needs_infra_false_for_trtllm_serve(self):
        """trtllm-serve should not need infrastructure."""
        # This tests the logic pattern used in do_sweep.py
        frontend_type = "trtllm-serve"
        needs_infra = frontend_type == "dynamo"
        assert needs_infra is False

    def test_needs_infra_true_for_dynamo(self):
        """dynamo should need infrastructure."""
        frontend_type = "dynamo"
        needs_infra = frontend_type == "dynamo"
        assert needs_infra is True

    def test_needs_infra_false_for_sglang(self):
        """sglang should not need infrastructure."""
        frontend_type = "sglang"
        needs_infra = frontend_type == "dynamo"
        assert needs_infra is False
