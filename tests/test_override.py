# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config override (base + override_*) functionality."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.submit import is_override_config, parse_config_arg, submit_override
from srtctl.core.config import deep_merge, expand_zip_override, generate_override_configs


# =============================================================================
# TestDeepMerge
# =============================================================================


class TestDeepMerge:
    def test_scalar_override(self):
        """Scalar fields are overridden."""
        base = {"name": "old", "resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4}}
        result = deep_merge(base, override)
        assert result["resources"]["decode_nodes"] == 4
        assert result["name"] == "old"

    def test_list_replace(self):
        """Lists are fully replaced, not appended."""
        base = {"benchmark": {"concurrencies": [8192, 10240]}}
        override = {"benchmark": {"concurrencies": [4096]}}
        result = deep_merge(base, override)
        assert result["benchmark"]["concurrencies"] == [4096]

    def test_nested_dict_merge(self):
        """Nested dicts are recursively merged, preserving untouched keys."""
        base = {"backend": {"sglang_config": {"prefill": {"tp-size": 32, "trust-remote-code": True}}}}
        override = {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}}
        result = deep_merge(base, override)
        assert result["backend"]["sglang_config"]["prefill"]["tp-size"] == 64
        assert result["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True

    def test_null_deletes_key(self):
        """Setting a value to None deletes the key."""
        base = {"extra_mount": ["/data:/data"], "name": "test"}
        override = {"extra_mount": None}
        result = deep_merge(base, override)
        assert "extra_mount" not in result
        assert result["name"] == "test"

    def test_null_delete_missing_key_is_noop(self):
        """Deleting a non-existent key is a no-op."""
        base = {"name": "test"}
        override = {"nonexistent": None}
        result = deep_merge(base, override)
        assert result == {"name": "test"}

    def test_add_new_key(self):
        """Override can add keys not present in base."""
        base = {"name": "test"}
        override = {"environment": {"NEW_VAR": "value"}}
        result = deep_merge(base, override)
        assert result["environment"]["NEW_VAR"] == "value"
        assert result["name"] == "test"

    def test_base_not_mutated(self):
        """Deep merge does not mutate the original base dict."""
        base = {"resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4}}
        deep_merge(base, override)
        assert base["resources"]["decode_nodes"] == 8

    def test_override_not_mutated(self):
        """Deep merge does not mutate the override dict."""
        base = {"resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4, "extra": [1, 2, 3]}}
        result = deep_merge(base, override)
        result["resources"]["extra"].append(4)
        assert override["resources"]["extra"] == [1, 2, 3]


# =============================================================================
# TestParseConfigArg
# =============================================================================


class TestParseConfigArg:
    def test_plain_path(self):
        """Plain path without selector."""
        path, selector = parse_config_arg("config.yaml")
        assert path == Path("config.yaml")
        assert selector is None

    def test_path_with_override_selector(self):
        """Path with override selector."""
        path, selector = parse_config_arg("config.yaml:override_tp64")
        assert path == Path("config.yaml")
        assert selector == "override_tp64"

    def test_path_with_base_selector(self):
        """Path with base selector."""
        path, selector = parse_config_arg("recipes/test.yaml:base")
        assert path == Path("recipes/test.yaml")
        assert selector == "base"

    def test_invalid_selector_raises(self):
        """Invalid selector raises ValueError."""
        with pytest.raises(ValueError, match="Invalid selector"):
            parse_config_arg("config.yaml:foobar")

    def test_directory_path_no_selector(self):
        """Directory path without selector."""
        path, selector = parse_config_arg("./configs/")
        assert path == Path("./configs/")
        assert selector is None


# =============================================================================
# TestGenerateOverrideConfigs
# =============================================================================


class TestGenerateOverrideConfigs:
    def test_base_only(self):
        """Config with only base returns one config."""
        raw = {"base": {"name": "test", "resources": {"decode_nodes": 8}}}
        configs = generate_override_configs(raw)
        assert len(configs) == 1
        assert configs[0][0] == "base"
        assert configs[0][1]["name"] == "test"

    def test_with_overrides(self):
        """Base + 2 overrides returns 3 configs with correct names."""
        raw = {
            "base": {"name": "test", "resources": {"decode_nodes": 8}},
            "override_small": {"resources": {"decode_nodes": 4}},
            "override_large": {"resources": {"decode_nodes": 16}},
        }
        configs = generate_override_configs(raw)
        assert len(configs) == 3

        # base comes first, overrides sorted alphabetically
        assert configs[0][0] == "base"
        assert configs[0][1]["name"] == "test"

        assert configs[1][0] == "large"
        assert configs[1][1]["name"] == "test_large"
        assert configs[1][1]["resources"]["decode_nodes"] == 16

        assert configs[2][0] == "small"
        assert configs[2][1]["name"] == "test_small"
        assert configs[2][1]["resources"]["decode_nodes"] == 4

    def test_override_name_generation(self):
        """Override auto-generates name = base_name + suffix."""
        raw = {
            "base": {"name": "gb300-fp8"},
            "override_tp64": {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}},
        }
        configs = generate_override_configs(raw)
        assert configs[1][1]["name"] == "gb300-fp8_tp64"

    def test_deep_merge_applied(self):
        """Override fields are deep-merged with base."""
        raw = {
            "base": {
                "name": "test",
                "backend": {"sglang_config": {"prefill": {"tp-size": 32, "trust-remote-code": True}}},
            },
            "override_tp64": {"backend": {"sglang_config": {"prefill": {"tp-size": 64}}}},
        }
        configs = generate_override_configs(raw)
        merged = configs[1][1]
        assert merged["backend"]["sglang_config"]["prefill"]["tp-size"] == 64
        assert merged["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True

    def test_selector_base_only(self):
        """selector='base' returns only the base config."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"name": "a"},
        }
        configs = generate_override_configs(raw, selector="base")
        assert len(configs) == 1
        assert configs[0][0] == "base"

    def test_selector_specific_override(self):
        """selector='override_a' returns only that variant."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"resources": {"decode_nodes": 4}},
            "override_b": {"resources": {"decode_nodes": 16}},
        }
        configs = generate_override_configs(raw, selector="override_a")
        assert len(configs) == 1
        assert configs[0][0] == "a"
        assert configs[0][1]["name"] == "test_a"
        assert configs[0][1]["resources"]["decode_nodes"] == 4

    def test_selector_not_found_raises(self):
        """Selector for non-existent override raises ValueError."""
        raw = {"base": {"name": "test"}}
        with pytest.raises(ValueError, match="not found"):
            generate_override_configs(raw, selector="override_nope")

    def test_non_override_keys_ignored(self):
        """Keys that don't start with 'override_' (other than 'base') are ignored."""
        raw = {
            "base": {"name": "test"},
            "override_a": {"resources": {"decode_nodes": 4}},
            "some_other_key": {"foo": "bar"},
        }
        configs = generate_override_configs(raw)
        assert len(configs) == 2  # base + override_a, some_other_key ignored


# =============================================================================
# TestIsOverrideConfig
# =============================================================================


class TestIsOverrideConfig:
    def test_normal_config_not_detected(self, tmp_path):
        """Normal config (no 'base' key) is not detected as override."""
        config_file = tmp_path / "normal.yaml"
        config_file.write_text(yaml.dump({"name": "test", "resources": {"decode_nodes": 8}}))
        assert is_override_config(config_file) is False

    def test_override_config_detected(self, tmp_path):
        """Config with 'base' key is detected as override."""
        config_file = tmp_path / "override.yaml"
        config_file.write_text(
            yaml.dump({
                "base": {"name": "test", "resources": {"decode_nodes": 8}},
                "override_small": {"resources": {"decode_nodes": 4}},
            })
        )
        assert is_override_config(config_file) is True

    def test_base_only_config_detected(self, tmp_path):
        """Config with only 'base' key (no overrides) is still detected."""
        config_file = tmp_path / "base_only.yaml"
        config_file.write_text(yaml.dump({"base": {"name": "test"}}))
        assert is_override_config(config_file) is True

    def test_empty_file_not_detected(self, tmp_path):
        """Empty YAML file is not detected as override."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        assert is_override_config(config_file) is False


# =============================================================================
# TestSubmitOverrideE2E
# =============================================================================

# Minimal valid SrtConfig for testing
MINIMAL_CONFIG = {
    "name": "test-job",
    "model": {
        "path": "/models/test-model",
        "container": "test-container.sqsh",
        "precision": "fp8",
    },
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
    "benchmark": {"type": "manual"},
}


class TestSubmitOverrideE2E:
    """Integration tests for submit_override with mocked SLURM."""

    def _write_override_config(self, tmp_path, overrides=None):
        """Write an override config YAML to tmp_path and return the path."""
        raw = {"base": MINIMAL_CONFIG.copy()}
        raw["base"] = {**MINIMAL_CONFIG}
        if overrides:
            raw.update(overrides)
        config_file = tmp_path / "override_test.yaml"
        config_file.write_text(yaml.dump(raw, default_flow_style=False))
        return config_file

    def test_dry_run_base_only(self, tmp_path, capsys):
        """Dry-run with base-only override config shows one variant."""
        config_file = self._write_override_config(tmp_path)

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, dry_run=True)

        output = capsys.readouterr().out
        assert "1 variant" in output
        assert "test-job" in output

    def test_dry_run_with_overrides(self, tmp_path, capsys):
        """Dry-run with overrides shows correct number of variants."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={
                "override_small": {"resources": {"decode_nodes": 2}},
                "override_large": {"resources": {"decode_nodes": 4}},
            },
        )

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, dry_run=True)

        output = capsys.readouterr().out
        assert "3 variants" in output
        assert "test-job" in output
        assert "test-job_small" in output
        assert "test-job_large" in output

    def test_dry_run_with_selector(self, tmp_path, capsys):
        """Dry-run with selector shows only selected variant."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={
                "override_small": {"resources": {"decode_nodes": 2}},
                "override_large": {"resources": {"decode_nodes": 4}},
            },
        )

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, selector="override_small", dry_run=True)

        output = capsys.readouterr().out
        assert "1 variant" in output
        assert "override_small" in output
        assert "test-job_small" in output
        # Should NOT mention the large variant
        assert "test-job_large" not in output

    def test_submit_calls_sbatch_for_each_variant(self, tmp_path):
        """Real submit (non-dry-run) calls sbatch for each variant."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={"override_small": {"resources": {"decode_nodes": 2}}},
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0

        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, output_dir=tmp_path)

        # Should have called sbatch twice (base + override_small)
        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 2

    def test_selector_base_submits_one_job(self, tmp_path):
        """Selector 'base' submits exactly one job."""
        config_file = self._write_override_config(
            tmp_path,
            overrides={"override_small": {"resources": {"decode_nodes": 2}}},
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0

        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, selector="base", output_dir=tmp_path)

        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 1


# =============================================================================
# TestExpandZipOverride
# =============================================================================

_ZIP_BASE = {
    "name": "base-job",
    "backend": {"sglang_config": {"prefill": {"tensor-parallel-size": 4, "trust-remote-code": True}}},
    "benchmark": {"concurrencies": [4, 8]},
}


class TestExpandZipOverride:
    def test_basic_equal_length(self):
        """Equal-length lists produce N variants with the correct values."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}}}
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        assert len(variants) == 2
        assert variants[0][0] == "tp_sweep_0"
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 4
        assert variants[1][0] == "tp_sweep_1"
        assert variants[1][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 8

    def test_broadcast_length_one(self):
        """Length-1 lists are broadcast to all variants."""
        zip_dict = {
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "tensor-parallel-size": [4, 8],
                        "mem-fraction-static": [0.85],  # broadcast
                    }
                }
            }
        }
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        assert len(variants) == 2
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["mem-fraction-static"] == 0.85
        assert variants[1][1]["backend"]["sglang_config"]["prefill"]["mem-fraction-static"] == 0.85

    def test_list_of_lists_becomes_literal_list(self):
        """List-of-list elements become literal list values (not further zipped)."""
        zip_dict = {"benchmark": {"concurrencies": [[4, 8], [4, 8, 16]]}}
        variants = expand_zip_override("conc_sweep", zip_dict, _ZIP_BASE)
        assert len(variants) == 2
        assert variants[0][1]["benchmark"]["concurrencies"] == [4, 8]
        assert variants[1][1]["benchmark"]["concurrencies"] == [4, 8, 16]

    def test_scalar_values_broadcast_to_all(self):
        """Scalar values in zip_dict are applied to every variant unchanged."""
        zip_dict = {
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "tensor-parallel-size": [4, 8],
                        "trust-remote-code": False,  # scalar override
                    }
                }
            }
        }
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is False
        assert variants[1][1]["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is False

    def test_auto_name_generation(self):
        """Auto-name is {base_name}_{group}_{i} when zip_dict has no 'name' list."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}}}
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        assert variants[0][1]["name"] == "base-job_tp_sweep_0"
        assert variants[1][1]["name"] == "base-job_tp_sweep_1"

    def test_explicit_name_list_overrides_auto(self):
        """A 'name' list in zip_dict is used directly."""
        zip_dict = {
            "name": ["job-tp4", "job-tp8"],
            "backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}},
        }
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        assert variants[0][1]["name"] == "job-tp4"
        assert variants[1][1]["name"] == "job-tp8"

    def test_base_not_mutated(self):
        """expand_zip_override does not mutate the base dict."""
        base = {"name": "test", "resources": {"decode_nodes": 8}}
        zip_dict = {"resources": {"decode_nodes": [4, 2]}}
        expand_zip_override("size", zip_dict, base)
        assert base["resources"]["decode_nodes"] == 8

    def test_incompatible_lengths_raises(self):
        """Lists of different lengths (neither being 1) raise ValueError."""
        zip_dict = {
            "backend": {
                "sglang_config": {
                    "prefill": {"tensor-parallel-size": [4, 8]},
                    "decode": {"tensor-parallel-size": [4, 8, 16]},
                }
            }
        }
        with pytest.raises(ValueError, match="Incompatible zip lengths"):
            expand_zip_override("bad_sweep", zip_dict, _ZIP_BASE)

    def test_no_lists_raises(self):
        """zip_override with no list values raises ValueError."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": 4}}}}
        with pytest.raises(ValueError, match="no list values"):
            expand_zip_override("empty", zip_dict, _ZIP_BASE)

    def test_suffix_format(self):
        """Suffix is always '{group}_{i}'."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8, 16]}}}}
        variants = expand_zip_override("my_group", zip_dict, _ZIP_BASE)
        assert [s for s, _ in variants] == ["my_group_0", "my_group_1", "my_group_2"]

    def test_deep_merge_preserves_base_keys(self):
        """Keys present in base but absent from the zip slice are kept."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}}}
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)
        # trust-remote-code is in base but not in zip_dict — must survive
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True
        assert variants[1][1]["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True


# =============================================================================
# TestGenerateOverrideConfigsZip
# =============================================================================


class TestGenerateOverrideConfigsZip:
    RAW = {
        "base": {"name": "base-job"},
        "override_single": {"name": "single"},
        "zip_override_tp": {
            "backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}},
        },
    }

    def test_no_selector_includes_base_override_and_zip(self):
        """selector=None returns base + overrides + all zip variants."""
        variants = generate_override_configs(self.RAW)
        suffixes = [s for s, _ in variants]
        assert suffixes == ["base", "single", "tp_0", "tp_1"]

    def test_zip_selector_all_variants(self):
        """selector='zip_override_tp' returns all N zip variants."""
        variants = generate_override_configs(self.RAW, selector="zip_override_tp")
        assert len(variants) == 2
        assert variants[0][0] == "tp_0"
        assert variants[1][0] == "tp_1"

    def test_zip_selector_index_zero(self):
        """selector='zip_override_tp[0]' returns exactly the first variant."""
        variants = generate_override_configs(self.RAW, selector="zip_override_tp[0]")
        assert len(variants) == 1
        assert variants[0][0] == "tp_0"
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 4

    def test_zip_selector_index_one(self):
        """selector='zip_override_tp[1]' returns exactly the second variant."""
        variants = generate_override_configs(self.RAW, selector="zip_override_tp[1]")
        assert len(variants) == 1
        assert variants[0][0] == "tp_1"
        assert variants[0][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 8

    def test_zip_selector_index_out_of_range(self):
        """Index beyond N raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            generate_override_configs(self.RAW, selector="zip_override_tp[5]")

    def test_zip_selector_missing_group(self):
        """Selector for non-existent zip group raises ValueError."""
        with pytest.raises(ValueError):
            generate_override_configs(self.RAW, selector="zip_override_nonexistent")

    def test_override_selector_still_works(self):
        """Existing override_ selector is unaffected by zip support."""
        variants = generate_override_configs(self.RAW, selector="override_single")
        assert len(variants) == 1
        assert variants[0][0] == "single"
        assert variants[0][1]["name"] == "base-job_single"

    def test_multiple_zip_groups_all_expanded(self):
        """Multiple zip_override_* groups are all expanded with selector=None."""
        raw = {
            "base": {"name": "base"},
            "zip_override_tp": {"backend": {"tp": [4, 8]}},
            "zip_override_mem": {"backend": {"mem": [0.7, 0.8, 0.9]}},
        }
        variants = generate_override_configs(raw)
        suffixes = [s for s, _ in variants]
        assert suffixes == ["base", "mem_0", "mem_1", "mem_2", "tp_0", "tp_1"]

    def test_zip_variants_inherit_base_fields(self):
        """zip variants include all fields from base via deep_merge."""
        raw = {
            "base": {"name": "base", "resources": {"decode_nodes": 8}},
            "zip_override_tp": {"backend": {"tp": [4, 8]}},
        }
        variants = generate_override_configs(raw)
        for suffix, cfg in variants[1:]:  # skip base
            assert cfg["resources"]["decode_nodes"] == 8


# =============================================================================
# TestParseConfigArgZip
# =============================================================================


class TestParseConfigArgZip:
    def test_zip_selector_all(self):
        """zip_override_<name> selector is accepted."""
        path, selector = parse_config_arg("config.yaml:zip_override_tp_sweep")
        assert path == Path("config.yaml")
        assert selector == "zip_override_tp_sweep"

    def test_zip_selector_index(self):
        """zip_override_<name>[N] selector is accepted."""
        path, selector = parse_config_arg("config.yaml:zip_override_tp_sweep[0]")
        assert path == Path("config.yaml")
        assert selector == "zip_override_tp_sweep[0]"

    def test_zip_selector_large_index(self):
        """Multi-digit index is accepted."""
        _, selector = parse_config_arg("config.yaml:zip_override_foo[12]")
        assert selector == "zip_override_foo[12]"

    def test_invalid_selector_still_rejected(self):
        """Non-override, non-zip selector still raises ValueError."""
        with pytest.raises(ValueError, match="Invalid selector"):
            parse_config_arg("config.yaml:foobar")


# =============================================================================
# TestSubmitOverrideZipE2E
# =============================================================================


class TestSubmitOverrideZipE2E:
    """Integration tests for submit_override with zip_override_ variants."""

    def _write_zip_config(self, tmp_path, extra=None):
        raw = {
            "base": {**MINIMAL_CONFIG},
            "zip_override_tp": {
                "resources": {"decode_nodes": [1, 2]},
            },
        }
        if extra:
            raw.update(extra)
        config_file = tmp_path / "zip_test.yaml"
        config_file.write_text(yaml.dump(raw, default_flow_style=False))
        return config_file

    def test_dry_run_shows_zip_variants(self, tmp_path, capsys):
        """Dry-run with zip_override shows correct variant count."""
        config_file = self._write_zip_config(tmp_path)
        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, dry_run=True)
        output = capsys.readouterr().out
        assert "3 variants" in output  # base + tp_0 + tp_1

    def test_dry_run_zip_selector_all(self, tmp_path, capsys):
        """Dry-run with zip_override selector shows N variants."""
        config_file = self._write_zip_config(tmp_path)
        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, selector="zip_override_tp", dry_run=True)
        output = capsys.readouterr().out
        assert "2 variants" in output

    def test_dry_run_zip_selector_index(self, tmp_path, capsys):
        """Dry-run with zip_override[N] selector shows 1 variant."""
        config_file = self._write_zip_config(tmp_path)
        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            submit_override(config_file, selector="zip_override_tp[0]", dry_run=True)
        output = capsys.readouterr().out
        assert "1 variant" in output

    def test_submit_zip_calls_sbatch_per_variant(self, tmp_path):
        """Real submit calls sbatch once per zip variant (plus base = 3 total)."""
        config_file = self._write_zip_config(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0
        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, output_dir=tmp_path)
        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 3  # base + tp_0 + tp_1

    def test_submit_zip_selector_index_calls_sbatch_once(self, tmp_path):
        """zip_override[N] selector submits exactly one job."""
        config_file = self._write_zip_config(tmp_path)
        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0
        with (
            patch("srtctl.cli.submit.load_cluster_config", return_value=None),
            patch("subprocess.run", return_value=mock_result) as mock_sbatch,
            patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
            patch("srtctl.cli.submit.create_job_record"),
        ):
            submit_override(config_file, selector="zip_override_tp[1]", output_dir=tmp_path)
        sbatch_calls = [c for c in mock_sbatch.call_args_list if c[0][0][0] == "sbatch"]
        assert len(sbatch_calls) == 1
