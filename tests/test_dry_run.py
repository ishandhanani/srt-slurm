# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for enhanced dry-run functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from srtctl.backends import SGLangProtocol
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig
from srtctl.core.topology import allocate_endpoints, endpoints_to_processes


class TestNodesMock:
    """Tests for Nodes.from_mock() classmethod."""

    def test_mock_generates_node_names(self):
        """Mock nodes are named node-01, node-02, etc."""
        nodes = Nodes.from_mock(num_nodes=4)

        assert nodes.head == "node-01"
        assert nodes.infra == "node-01"
        assert nodes.bench == "node-01"
        assert nodes.worker == ("node-01", "node-02", "node-03", "node-04")

    def test_mock_with_dedicated_infra_node(self):
        """With dedicated infra node, first node is infra-only."""
        nodes = Nodes.from_mock(num_nodes=4, etcd_nats_dedicated_node=True)

        assert nodes.infra == "node-01"
        assert nodes.head == "node-02"
        assert nodes.bench == "node-02"
        assert nodes.worker == ("node-02", "node-03", "node-04")

    def test_mock_dedicated_infra_requires_two_nodes(self):
        """Dedicated infra node requires at least 2 nodes."""
        with pytest.raises(ValueError, match="at least 2 nodes"):
            Nodes.from_mock(num_nodes=1, etcd_nats_dedicated_node=True)


class TestRuntimeContextDryRun:
    """Tests for RuntimeContext.from_config_dry_run() classmethod."""

    def test_dry_run_context_creates_mock_nodes(self):
        """Dry-run creates mock node names based on total_nodes."""
        config = SrtConfig(
            name="test-dry-run",
            model=ModelConfig(
                path="/models/test-model",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=2,
            ),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        assert runtime.job_id == "DRY_RUN"
        assert runtime.run_name == "test-dry-run_DRY_RUN"
        assert runtime.nodes.head == "node-01"
        assert len(runtime.nodes.worker) == 3  # total_nodes = 3

    def test_dry_run_context_uses_placeholder_paths(self):
        """Dry-run uses placeholder log directory."""
        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="/models/test",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                agg_nodes=2,
            ),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        assert "dry-run-outputs" in str(runtime.log_dir)
        assert "DRY_RUN" in str(runtime.log_dir)

    def test_dry_run_context_handles_hf_models(self):
        """Dry-run correctly handles HuggingFace model paths."""
        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="hf:meta-llama/Llama-2-7b",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                agg_nodes=1,
            ),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        assert runtime.is_hf_model is True
        assert "Llama-2-7b" in str(runtime.model_path)

    def test_dry_run_skips_file_validation(self):
        """Dry-run doesn't validate model/container paths exist."""
        # Use paths that don't exist - should not raise
        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="/nonexistent/model/path",
                container="/nonexistent/container.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                agg_nodes=1,
            ),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            # Should not raise FileNotFoundError
            runtime = RuntimeContext.from_config_dry_run(config)

        assert runtime.model_path == Path("/nonexistent/model/path")
        assert runtime.container_image == Path("/nonexistent/container.sqsh")

    def test_dry_run_with_dedicated_infra_node(self):
        """Dry-run handles etcd_nats_dedicated_node configuration."""
        from srtctl.core.schema import InfraConfig

        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="/models/test",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=1,
            ),
            infra=InfraConfig(etcd_nats_dedicated_node=True),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        # With dedicated infra: 2 worker nodes + 1 infra = 3 total
        assert runtime.nodes.infra == "node-01"  # First is infra-only
        assert runtime.nodes.head == "node-02"  # Second is head
        assert runtime.infra_node_ip != runtime.head_node_ip


class TestDryRunEndpointAllocation:
    """Tests for endpoint allocation in dry-run mode."""

    def test_dry_run_allocates_endpoints(self):
        """Dry-run can allocate endpoints using mock nodes."""
        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="/models/test",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                prefill_workers=1,
                decode_nodes=2,
                decode_workers=4,
            ),
            backend=SGLangProtocol(),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        resources = config.resources
        endpoints = allocate_endpoints(
            num_prefill=resources.num_prefill,
            num_decode=resources.num_decode,
            num_agg=resources.num_agg,
            gpus_per_prefill=resources.gpus_per_prefill,
            gpus_per_decode=resources.gpus_per_decode,
            gpus_per_agg=resources.gpus_per_agg,
            gpus_per_node=resources.gpus_per_node,
            available_nodes=runtime.nodes.worker,
        )

        assert len(endpoints) == 5  # 1 prefill + 4 decode
        prefill_endpoints = [e for e in endpoints if e.mode == "prefill"]
        decode_endpoints = [e for e in endpoints if e.mode == "decode"]
        assert len(prefill_endpoints) == 1
        assert len(decode_endpoints) == 4

    def test_dry_run_converts_endpoints_to_processes(self):
        """Dry-run can convert endpoints to processes."""
        config = SrtConfig(
            name="test",
            model=ModelConfig(
                path="/models/test",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                prefill_workers=1,
                decode_nodes=1,
                decode_workers=2,
            ),
        )

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            runtime = RuntimeContext.from_config_dry_run(config)

        resources = config.resources
        endpoints = allocate_endpoints(
            num_prefill=resources.num_prefill,
            num_decode=resources.num_decode,
            num_agg=resources.num_agg,
            gpus_per_prefill=resources.gpus_per_prefill,
            gpus_per_decode=resources.gpus_per_decode,
            gpus_per_agg=resources.gpus_per_agg,
            gpus_per_node=resources.gpus_per_node,
            available_nodes=runtime.nodes.worker,
        )

        processes = endpoints_to_processes(endpoints)

        assert len(processes) >= 3  # 1 prefill + 2 decode
        for proc in processes:
            assert proc.node.startswith("node-")
            assert proc.sys_port > 0
            assert len(proc.gpu_indices) > 0


class TestDisplayEnhancedDryRun:
    """Tests for the enhanced dry-run display function."""

    def test_display_dry_run_runs_without_error(self, capsys):
        """display_enhanced_dry_run() runs without raising exceptions."""
        from srtctl.cli.submit import display_enhanced_dry_run

        config = SrtConfig(
            name="test-display",
            model=ModelConfig(
                path="/models/test",
                container="/containers/test.sqsh",
                precision="fp8",
            ),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=1,
            ),
            backend=SGLangProtocol(),
        )

        config_path = Path("/tmp/test.yaml")

        with patch("srtctl.core.runtime.get_srtslurm_setting", return_value="eth0"):
            # Should not raise
            display_enhanced_dry_run(config, config_path)

        # Check something was printed
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out or len(captured.out) > 0
