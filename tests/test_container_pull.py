# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for container-pull functionality and ContainerEntry schema."""

from pathlib import Path
from unittest.mock import patch

import pytest

from srtctl.core.schema import ClusterConfig, ContainerEntry


class TestContainerEntrySchema:
    """Tests for ContainerEntry schema and backwards compatibility."""

    def test_container_entry_with_path_only(self):
        """ContainerEntry can be created with just a path."""
        entry = ContainerEntry(path="/path/to/container.sqsh")
        
        assert entry.path == "/path/to/container.sqsh"
        assert entry.source is None

    def test_container_entry_with_source(self):
        """ContainerEntry can include optional source URL."""
        entry = ContainerEntry(
            path="/path/to/container.sqsh",
            source="docker://nvcr.io/nvidia/sglang:0.4.1",
        )
        
        assert entry.path == "/path/to/container.sqsh"
        assert entry.source == "docker://nvcr.io/nvidia/sglang:0.4.1"


class TestClusterConfigContainers:
    """Tests for ClusterConfig containers field parsing."""

    def test_string_format_still_works(self):
        """Old format (containers: {name: "/path"}) still works."""
        schema = ClusterConfig.Schema()
        
        data = {
            "containers": {
                "sglang": "/shared/containers/sglang.sqsh",
                "nginx": "/shared/containers/nginx.sqsh",
            }
        }
        
        config = schema.load(data)
        
        assert config.containers is not None
        assert "sglang" in config.containers
        assert config.containers["sglang"].path == "/shared/containers/sglang.sqsh"
        assert config.containers["sglang"].source is None
        assert config.containers["nginx"].path == "/shared/containers/nginx.sqsh"

    def test_dict_format_works(self):
        """New format (containers: {name: {path: ..., source: ...}}) works."""
        schema = ClusterConfig.Schema()
        
        data = {
            "containers": {
                "sglang": {
                    "path": "/shared/containers/sglang.sqsh",
                    "source": "docker://nvcr.io/nvidia/sglang:0.4.1",
                }
            }
        }
        
        config = schema.load(data)
        
        assert config.containers is not None
        assert config.containers["sglang"].path == "/shared/containers/sglang.sqsh"
        assert config.containers["sglang"].source == "docker://nvcr.io/nvidia/sglang:0.4.1"

    def test_mixed_formats_work(self):
        """Can mix old and new formats in same containers dict."""
        schema = ClusterConfig.Schema()
        
        data = {
            "containers": {
                # Old format (still works)
                "nginx": "/shared/containers/nginx.sqsh",
                # New format (enables container-pull)
                "sglang": {
                    "path": "/shared/containers/sglang.sqsh",
                    "source": "docker://nvcr.io/nvidia/sglang:0.4.1",
                }
            }
        }
        
        config = schema.load(data)
        
        assert config.containers is not None
        # Old format
        assert config.containers["nginx"].path == "/shared/containers/nginx.sqsh"
        assert config.containers["nginx"].source is None
        # New format
        assert config.containers["sglang"].path == "/shared/containers/sglang.sqsh"
        assert config.containers["sglang"].source == "docker://nvcr.io/nvidia/sglang:0.4.1"

    def test_serialization_preserves_format(self):
        """Serialization keeps simple strings for entries without source."""
        schema = ClusterConfig.Schema()
        
        data = {
            "containers": {
                "nginx": "/shared/containers/nginx.sqsh",
                "sglang": {
                    "path": "/shared/containers/sglang.sqsh",
                    "source": "docker://nvcr.io/nvidia/sglang:0.4.1",
                }
            }
        }
        
        config = schema.load(data)
        serialized = schema.dump(config)
        
        # Simple path (no source) serializes back to string
        assert serialized["containers"]["nginx"] == "/shared/containers/nginx.sqsh"
        # Entry with source serializes to dict
        assert isinstance(serialized["containers"]["sglang"], dict)
        assert serialized["containers"]["sglang"]["path"] == "/shared/containers/sglang.sqsh"
        assert serialized["containers"]["sglang"]["source"] == "docker://nvcr.io/nvidia/sglang:0.4.1"


class TestResolveContainerPath:
    """Tests for resolve_container_path() function."""

    def test_resolves_string_entry(self):
        """Resolves old format to path string."""
        from srtctl.core.config import resolve_container_path
        
        mock_containers = {
            "sglang": "/path/to/sglang.sqsh",
        }
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=mock_containers):
            path = resolve_container_path("sglang")
        
        assert path == "/path/to/sglang.sqsh"

    def test_resolves_dict_entry(self):
        """Resolves new format to path from dict."""
        from srtctl.core.config import resolve_container_path
        
        mock_containers = {
            "sglang": {"path": "/path/to/sglang.sqsh", "source": "docker://..."},
        }
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=mock_containers):
            path = resolve_container_path("sglang")
        
        assert path == "/path/to/sglang.sqsh"

    def test_returns_none_for_unknown(self):
        """Returns None for unknown container names."""
        from srtctl.core.config import resolve_container_path
        
        mock_containers = {
            "sglang": "/path/to/sglang.sqsh",
        }
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=mock_containers):
            path = resolve_container_path("unknown")
        
        assert path is None


class TestGetContainerEntries:
    """Tests for get_container_entries() function."""

    def test_normalizes_string_format(self):
        """String format is normalized to dict with source=None."""
        from srtctl.core.config import get_container_entries
        
        mock_containers = {
            "nginx": "/path/to/nginx.sqsh",
        }
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=mock_containers):
            entries = get_container_entries()
        
        assert "nginx" in entries
        assert entries["nginx"]["path"] == "/path/to/nginx.sqsh"
        assert entries["nginx"]["source"] is None

    def test_preserves_dict_format(self):
        """Dict format is preserved with source."""
        from srtctl.core.config import get_container_entries
        
        mock_containers = {
            "sglang": {"path": "/path/to/sglang.sqsh", "source": "docker://..."},
        }
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=mock_containers):
            entries = get_container_entries()
        
        assert "sglang" in entries
        assert entries["sglang"]["path"] == "/path/to/sglang.sqsh"
        assert entries["sglang"]["source"] == "docker://..."

    def test_returns_empty_when_no_containers(self):
        """Returns empty dict when no containers defined."""
        from srtctl.core.config import get_container_entries
        
        with patch("srtctl.core.config.get_srtslurm_setting", return_value=None):
            entries = get_container_entries()
        
        assert entries == {}


class TestContainerPullCommand:
    """Tests for container-pull CLI command."""

    def test_skips_entries_without_source(self, capsys):
        """Skips containers with no source defined (local mode)."""
        from srtctl.cli.container_pull import container_pull
        
        mock_containers = {
            "nginx": {"path": "/path/to/nginx.sqsh", "source": None},
        }
        
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            exit_code = container_pull(force=False, local=True)
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "no source" in captured.out.lower()

    def test_skips_existing_files(self, capsys, tmp_path):
        """Skips containers that already exist (local mode)."""
        from srtctl.cli.container_pull import container_pull
        
        # Create a file that "already exists"
        existing_file = tmp_path / "existing.sqsh"
        existing_file.touch()
        
        mock_containers = {
            "existing": {
                "path": str(existing_file),
                "source": "docker://some/image",
            },
        }
        
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            exit_code = container_pull(force=False, local=True)
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "exists" in captured.out.lower()

    def test_force_triggers_redownload(self, capsys, tmp_path):
        """--force flag triggers re-download even if file exists (local mode)."""
        from srtctl.cli.container_pull import container_pull
        
        # Create a file that "already exists"
        existing_file = tmp_path / "existing.sqsh"
        existing_file.touch()
        
        mock_containers = {
            "existing": {
                "path": str(existing_file),
                "source": "docker://some/image",
            },
        }
        
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            with patch("srtctl.cli.container_pull.run_enroot_import", return_value=True) as mock_import:
                exit_code = container_pull(force=True, local=True)
        
        # Should have tried to import even though file exists
        mock_import.assert_called_once()
        assert exit_code == 0

    def test_downloads_missing_containers(self, capsys, tmp_path):
        """Downloads containers that don't exist (local mode)."""
        from srtctl.cli.container_pull import container_pull
        
        # Use a path that doesn't exist
        missing_file = tmp_path / "missing.sqsh"
        
        mock_containers = {
            "new": {
                "path": str(missing_file),
                "source": "docker://some/image",
            },
        }
        
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            with patch("srtctl.cli.container_pull.run_enroot_import", return_value=True) as mock_import:
                exit_code = container_pull(force=False, local=True)
        
        mock_import.assert_called_once_with("docker://some/image", missing_file)
        assert exit_code == 0

    def test_default_submits_batch_job(self, capsys, tmp_path):
        """Default behavior submits a batch job."""
        from srtctl.cli.container_pull import container_pull
        
        # Use a path that doesn't exist
        missing_file = tmp_path / "missing.sqsh"
        
        mock_containers = {
            "new": {
                "path": str(missing_file),
                "source": "docker://some/image",
            },
        }
        
        # submit_container_pull_job returns (job_id, log_path) tuple
        mock_result = ("12345", "/logs/container-pull-12345.log")
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            with patch("srtctl.cli.container_pull.submit_container_pull_job", return_value=mock_result) as mock_submit:
                exit_code = container_pull(force=False, local=False)
        
        mock_submit.assert_called_once_with(mock_containers, force=False)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "12345" in captured.out

    def test_local_runs_directly(self, capsys, tmp_path):
        """--local flag runs directly instead of submitting batch job."""
        from srtctl.cli.container_pull import container_pull
        
        # Use a path that doesn't exist
        missing_file = tmp_path / "missing.sqsh"
        
        mock_containers = {
            "new": {
                "path": str(missing_file),
                "source": "docker://some/image",
            },
        }
        
        with patch("srtctl.cli.container_pull.get_container_entries", return_value=mock_containers):
            with patch("srtctl.cli.container_pull.run_enroot_import", return_value=True) as mock_import:
                exit_code = container_pull(force=False, local=True)
        
        mock_import.assert_called_once()
        assert exit_code == 0


class TestContainerAliasResolution:
    """Tests for container alias resolution in config loading."""

    def test_new_format_alias_resolution(self):
        """Container aliases work with new format."""
        from srtctl.core.config import resolve_config_with_defaults
        
        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "sglang", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        }
        
        cluster_config = {
            "containers": {
                "sglang": {
                    "path": "/shared/containers/sglang.sqsh",
                    "source": "docker://nvcr.io/nvidia/sglang:0.4.1",
                }
            }
        }
        
        resolved = resolve_config_with_defaults(user_config, cluster_config)
        
        # Should resolve to the path, not the dict
        assert resolved["model"]["container"] == "/shared/containers/sglang.sqsh"

    def test_old_format_alias_still_works(self):
        """Container aliases still work with old format."""
        from srtctl.core.config import resolve_config_with_defaults
        
        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "sglang", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        }
        
        cluster_config = {
            "containers": {
                "sglang": "/shared/containers/sglang.sqsh",
            }
        }
        
        resolved = resolve_config_with_defaults(user_config, cluster_config)
        
        assert resolved["model"]["container"] == "/shared/containers/sglang.sqsh"

    def test_nginx_container_alias_new_format(self):
        """Frontend nginx_container works with new container format."""
        from srtctl.core.config import resolve_config_with_defaults
        
        user_config = {
            "name": "test",
            "model": {"path": "/model", "container": "/container.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
            "frontend": {"nginx_container": "nginx"},
        }
        
        cluster_config = {
            "containers": {
                "nginx": {
                    "path": "/shared/containers/nginx.sqsh",
                    "source": "docker://nginx:1.27.4",
                }
            }
        }
        
        resolved = resolve_config_with_defaults(user_config, cluster_config)
        
        assert resolved["frontend"]["nginx_container"] == "/shared/containers/nginx.sqsh"
