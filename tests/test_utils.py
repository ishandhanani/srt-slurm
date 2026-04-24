# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for utility functions.

Covers:
- submit.py: get_job_name, is_sweep_config, find_yaml_files
- config.py: load_cluster_config, resolve_config_with_defaults, validate_config_file
- slurm.py: get_slurm_job_id, get_slurm_nodelist, get_container_mounts_str
- sglang.py: _config_to_cli_args
"""

import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from srtctl.backends.sglang import _config_to_cli_args
from srtctl.cli.submit import find_yaml_files, get_job_name, is_sweep_config
from srtctl.core.config import load_cluster_config, resolve_config_with_defaults, validate_config_file
from srtctl.core.slurm import get_container_mounts_str, get_slurm_job_id, get_slurm_nodelist


# =============================================================================
# submit.py
# =============================================================================


class TestGetJobName:
    def test_returns_config_name_by_default(self, monkeypatch):
        monkeypatch.delenv("RUNNER_NAME", raising=False)
        config = MagicMock()
        config.name = "my-job"
        assert get_job_name(config) == "my-job"

    def test_returns_runner_name_when_set(self, monkeypatch):
        monkeypatch.setenv("RUNNER_NAME", "runner-42")
        config = MagicMock()
        config.name = "my-job"
        assert get_job_name(config) == "runner-42"

    def test_empty_runner_name_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv("RUNNER_NAME", "")
        config = MagicMock()
        config.name = "my-job"
        # Empty string is falsy — should fall back
        assert get_job_name(config) == "my-job"


class TestIsSweepConfig:
    def test_sweep_config_returns_true(self, tmp_path):
        p = tmp_path / "sweep.yaml"
        p.write_text("sweep:\n  param: [1, 2, 3]\nname: test\n")
        assert is_sweep_config(p) is True

    def test_plain_config_returns_false(self, tmp_path):
        p = tmp_path / "plain.yaml"
        p.write_text("name: test\nresources:\n  gpu_type: b200\n")
        assert is_sweep_config(p) is False

    def test_override_config_without_sweep_returns_false(self, tmp_path):
        p = tmp_path / "override.yaml"
        p.write_text("base:\n  name: test\noverride_tp8:\n  resources: {}\n")
        assert is_sweep_config(p) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert is_sweep_config(tmp_path / "missing.yaml") is False

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert is_sweep_config(p) is False


class TestFindYamlFiles:
    def test_finds_yaml_and_yml(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: 1")
        (tmp_path / "b.yml").write_text("x: 2")
        (tmp_path / "c.txt").write_text("not yaml")
        results = find_yaml_files(tmp_path)
        names = {p.name for p in results}
        assert "a.yaml" in names
        assert "b.yml" in names
        assert "c.txt" not in names

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.yaml").write_text("x: 1")
        (tmp_path / "top.yaml").write_text("x: 2")
        results = find_yaml_files(tmp_path)
        names = {p.name for p in results}
        assert "nested.yaml" in names
        assert "top.yaml" in names

    def test_returns_sorted_list(self, tmp_path):
        (tmp_path / "z.yaml").write_text("")
        (tmp_path / "a.yaml").write_text("")
        (tmp_path / "m.yaml").write_text("")
        results = find_yaml_files(tmp_path)
        assert results == sorted(results)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert find_yaml_files(tmp_path) == []


# =============================================================================
# config.py
# =============================================================================


MINIMAL_CLUSTER_YAML = textwrap.dedent("""\
    default_account: my-account
    default_partition: gpu
    model_paths:
      dsr1: /models/deepseek-r1
    containers:
      dynamo-sglang: nvcr.io/nvidia/dynamo-sglang:latest
""")


class TestLoadClusterConfig:
    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)
        result = load_cluster_config()
        assert result is None

    def test_loads_from_env_var(self, tmp_path, monkeypatch):
        cfg = tmp_path / "srtslurm.yaml"
        cfg.write_text(MINIMAL_CLUSTER_YAML)
        monkeypatch.setenv("SRTSLURM_CONFIG", str(cfg))
        result = load_cluster_config()
        assert result is not None
        assert result["default_account"] == "my-account"

    def test_env_var_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "nonexistent.yaml"))
        result = load_cluster_config()
        assert result is None

    def test_loads_from_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)
        (tmp_path / "srtslurm.yaml").write_text(MINIMAL_CLUSTER_YAML)
        result = load_cluster_config()
        assert result is not None
        assert result["default_partition"] == "gpu"

    def test_invalid_yaml_returns_none(self, tmp_path, monkeypatch):
        cfg = tmp_path / "srtslurm.yaml"
        cfg.write_text(": invalid: yaml: {{{")
        monkeypatch.setenv("SRTSLURM_CONFIG", str(cfg))
        result = load_cluster_config()
        assert result is None


class TestResolveConfigWithDefaults:
    def test_no_cluster_config_returns_unchanged(self):
        user = {"name": "test", "model": {"path": "/model"}}
        result = resolve_config_with_defaults(user, None)
        assert result == user

    def test_does_not_mutate_input(self):
        user = {"name": "test", "slurm": {}}
        cluster = {"default_account": "acct"}
        resolve_config_with_defaults(user, cluster)
        assert "account" not in user["slurm"]

    def test_applies_default_account(self):
        user = {"name": "test"}
        cluster = {"default_account": "my-account"}
        result = resolve_config_with_defaults(user, cluster)
        assert result["slurm"]["account"] == "my-account"

    def test_does_not_override_existing_account(self):
        user = {"name": "test", "slurm": {"account": "user-account"}}
        cluster = {"default_account": "cluster-account"}
        result = resolve_config_with_defaults(user, cluster)
        assert result["slurm"]["account"] == "user-account"

    def test_resolves_model_path_alias(self):
        user = {"name": "test", "model": {"path": "dsr1"}}
        cluster = {"model_paths": {"dsr1": "/models/deepseek-r1"}}
        result = resolve_config_with_defaults(user, cluster)
        assert result["model"]["path"] == "/models/deepseek-r1"

    def test_resolves_container_alias(self):
        user = {"name": "test", "model": {"container": "dynamo-sglang"}}
        cluster = {"containers": {"dynamo-sglang": "nvcr.io/nvidia/dynamo:latest"}}
        result = resolve_config_with_defaults(user, cluster)
        assert result["model"]["container"] == "nvcr.io/nvidia/dynamo:latest"

    def test_resolves_nginx_container_alias(self):
        user = {"name": "test", "frontend": {"nginx_container": "nginx-sqsh"}}
        cluster = {"containers": {"nginx-sqsh": "/path/to/nginx.sqsh"}}
        result = resolve_config_with_defaults(user, cluster)
        assert result["frontend"]["nginx_container"] == "/path/to/nginx.sqsh"

    def test_unknown_alias_left_unchanged(self):
        user = {"name": "test", "model": {"path": "unknown-alias"}}
        cluster = {"model_paths": {"dsr1": "/models/deepseek-r1"}}
        result = resolve_config_with_defaults(user, cluster)
        assert result["model"]["path"] == "unknown-alias"


class TestValidateConfigFile:
    def test_nonexistent_file_returns_error(self, tmp_path):
        errors = validate_config_file(tmp_path / "missing.yaml")
        assert len(errors) == 1
        assert "file not found" in errors[0]

    def test_invalid_yaml_returns_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": {{{invalid")
        errors = validate_config_file(p)
        assert len(errors) == 1
        assert "YAML parse error" in errors[0]

    def test_not_a_mapping_returns_error(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n")
        errors = validate_config_file(p)
        assert len(errors) == 1
        assert "not a YAML mapping" in errors[0]

    def test_valid_config_returns_no_errors(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)
        p = tmp_path / "valid.yaml"
        p.write_text(textwrap.dedent("""\
            name: test-job
            model:
              path: /models/my-model
              container: mycontainer:latest
              precision: fp8
            resources:
              gpu_type: b200
              prefill_nodes: 1
              decode_nodes: 1
              prefill_workers: 1
              decode_workers: 1
              gpus_per_node: 8
            backend:
              type: sglang
              sglang_config:
                prefill:
                  tensor-parallel-size: 8
                  disaggregation-mode: prefill
                decode:
                  tensor-parallel-size: 8
                  disaggregation-mode: decode
            benchmark:
              type: sa-bench
              isl: 1024
              osl: 1024
        """))
        errors = validate_config_file(p)
        assert errors == [], f"Unexpected errors: {errors}"


# =============================================================================
# slurm.py
# =============================================================================


class TestGetSlurmJobId:
    def test_returns_slurm_job_id(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_JOBID", raising=False)
        assert get_slurm_job_id() == "12345"

    def test_returns_slurm_jobid_fallback(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("SLURM_JOBID", "67890")
        assert get_slurm_job_id() == "67890"

    def test_slurm_job_id_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "111")
        monkeypatch.setenv("SLURM_JOBID", "222")
        assert get_slurm_job_id() == "111"

    def test_returns_none_outside_slurm(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)
        assert get_slurm_job_id() is None


class TestGetSlurmNodelist:
    def test_returns_empty_list_outside_slurm(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        assert get_slurm_nodelist() == []

    def test_expands_nodelist_via_scontrol(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODELIST", "h100-[01-03]")
        mock_result = MagicMock()
        mock_result.stdout = "h100-01\nh100-02\nh100-03"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            nodes = get_slurm_nodelist()
        assert nodes == ["h100-01", "h100-02", "h100-03"]
        mock_run.assert_called_once_with(
            ["scontrol", "show", "hostnames", "h100-[01-03]"],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_falls_back_to_raw_nodelist_on_scontrol_failure(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODELIST", "single-node")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            nodes = get_slurm_nodelist()
        assert nodes == ["single-node"]

    def test_falls_back_on_subprocess_error(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODELIST", "node01")
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "scontrol")):
            nodes = get_slurm_nodelist()
        assert nodes == ["node01"]


class TestGetContainerMountsStr:
    def test_single_mount(self):
        mounts = {Path("/host/data"): Path("/container/data")}
        result = get_container_mounts_str(mounts)
        assert result == "/host/data:/container/data"

    def test_multiple_mounts(self):
        mounts = {
            Path("/host/model"): Path("/model"),
            Path("/host/logs"): Path("/logs"),
        }
        result = get_container_mounts_str(mounts)
        parts = result.split(",")
        assert len(parts) == 2
        assert "/host/model:/model" in parts
        assert "/host/logs:/logs" in parts

    def test_empty_mounts(self):
        assert get_container_mounts_str({}) == ""


# =============================================================================
# sglang.py
# =============================================================================


class TestConfigToCliArgs:
    def test_string_value(self):
        args = _config_to_cli_args({"model-name": "deepseek"})
        assert args == ["--model-name", "deepseek"]

    def test_integer_value(self):
        args = _config_to_cli_args({"tensor-parallel-size": 8})
        assert args == ["--tensor-parallel-size", "8"]

    def test_bool_true_becomes_flag(self):
        args = _config_to_cli_args({"trust-remote-code": True})
        assert args == ["--trust-remote-code"]

    def test_bool_false_omitted(self):
        args = _config_to_cli_args({"disable-radix-cache": False})
        assert args == []

    def test_none_value_omitted(self):
        args = _config_to_cli_args({"optional-flag": None})
        assert args == []

    def test_list_value(self):
        args = _config_to_cli_args({"cuda-graph-bs": [1, 2, 4, 8]})
        assert args == ["--cuda-graph-bs", "1", "2", "4", "8"]

    def test_underscore_converted_to_dash(self):
        args = _config_to_cli_args({"mem_fraction_static": 0.9})
        assert "--mem-fraction-static" in args

    def test_args_sorted_by_key(self):
        args = _config_to_cli_args({"z-flag": "z", "a-flag": "a"})
        assert args.index("--a-flag") < args.index("--z-flag")


