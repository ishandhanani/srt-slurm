# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config override (base + override_* + zip_override_*) functionality."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.submit import is_override_config, parse_config_arg, submit_override
from srtctl.core.config import deep_merge, expand_zip_override, generate_override_configs


# =============================================================================
# TestDeepMerge
# =============================================================================


class TestDeepMerge:
    def test_merge_rules(self) -> None:
        """Scalar override, list replace, nested merge, null delete, new key — all in one."""
        base = {
            "name": "old",
            "resources": {"decode_nodes": 8},
            "benchmark": {"concurrencies": [8192, 10240]},
            "backend": {"sglang_config": {"prefill": {"tp-size": 32, "trust-remote-code": True}}},
            "extra_mount": ["/data:/data"],
        }
        override = {
            "resources": {"decode_nodes": 4},                          # scalar override
            "benchmark": {"concurrencies": [4096]},                    # list replace
            "backend": {"sglang_config": {"prefill": {"tp-size": 64}}},  # nested merge
            "extra_mount": None,                                        # null → delete
            "environment": {"NEW_VAR": "value"},                        # new key
        }
        result = deep_merge(base, override)
        assert result["name"] == "old"                                             # untouched
        assert result["resources"]["decode_nodes"] == 4                            # scalar
        assert result["benchmark"]["concurrencies"] == [4096]                      # list replace
        assert result["backend"]["sglang_config"]["prefill"]["tp-size"] == 64      # nested
        assert result["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is True  # preserved
        assert "extra_mount" not in result                                         # null delete
        assert result["environment"]["NEW_VAR"] == "value"                         # new key

    def test_immutability(self) -> None:
        """Neither base nor override is mutated."""
        base = {"resources": {"decode_nodes": 8}}
        override = {"resources": {"decode_nodes": 4, "extra": [1, 2, 3]}}
        result = deep_merge(base, override)
        result["resources"]["extra"].append(4)
        assert base["resources"]["decode_nodes"] == 8
        assert override["resources"]["extra"] == [1, 2, 3]


# =============================================================================
# TestParseConfigArg
# =============================================================================


class TestParseConfigArg:
    def test_valid_selectors(self) -> None:
        """Plain path, base, override_, zip_override_, and zip_override_[N] are all accepted."""
        assert parse_config_arg("config.yaml") == (Path("config.yaml"), None)
        assert parse_config_arg("config.yaml:base") == (Path("config.yaml"), "base")
        assert parse_config_arg("config.yaml:override_tp64") == (Path("config.yaml"), "override_tp64")
        assert parse_config_arg("config.yaml:zip_override_sweep") == (Path("config.yaml"), "zip_override_sweep")
        assert parse_config_arg("config.yaml:zip_override_sweep[0]") == (Path("config.yaml"), "zip_override_sweep[0]")
        assert parse_config_arg("config.yaml:zip_override_sweep[12]") == (Path("config.yaml"), "zip_override_sweep[12]")

    def test_invalid_selector_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid selector"):
            parse_config_arg("config.yaml:foobar")


# =============================================================================
# TestGenerateOverrideConfigs
# =============================================================================

_RAW = {
    "base": {"name": "base-job", "resources": {"decode_nodes": 8}},
    "override_small": {"resources": {"decode_nodes": 2}},
    "zip_override_tp": {
        "backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}},
    },
}

_RAW_MULTI = {
    "base": {"name": "base-job", "resources": {"decode_nodes": 8}},
    "override_maxtpt_1p1d": {"name": "max-1p1d", "resources": {"decode_nodes": 1}},
    "override_maxtpt_1p2d": {"name": "max-1p2d", "resources": {"decode_nodes": 2}},
    "override_lowlat_1d": {"name": "low-1d", "resources": {"decode_nodes": 1}},
    "zip_override_scale": {"resources": {"decode_nodes": [3, 4, 5]}},
    "zip_override_mem": {"resources": {"decode_nodes": [6, 7]}},
}


class TestGenerateOverrideConfigs:
    def test_full_expansion(self) -> None:
        """No selector: overrides + zip groups returned (sorted); base excluded."""
        variants = generate_override_configs(_RAW)
        suffixes = [s for s, _ in variants]
        assert suffixes == ["small", "tp_0", "tp_1"]

        # override auto-name and deep-merge
        assert variants[0][1]["name"] == "base-job_small"
        assert variants[0][1]["resources"]["decode_nodes"] == 2

        # zip variants have correct values and inherit base fields
        assert variants[1][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 4
        assert variants[2][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 8
        assert variants[1][1]["resources"]["decode_nodes"] == 8  # from base

    def test_selectors(self) -> None:
        """All selector forms return the right subset."""
        # base only
        r = generate_override_configs(_RAW, selector="base")
        assert len(r) == 1 and r[0][0] == "base"

        # specific override
        r = generate_override_configs(_RAW, selector="override_small")
        assert len(r) == 1 and r[0][0] == "small" and r[0][1]["resources"]["decode_nodes"] == 2

        # entire zip group
        r = generate_override_configs(_RAW, selector="zip_override_tp")
        assert [s for s, _ in r] == ["tp_0", "tp_1"]

        # single zip variant by index
        r = generate_override_configs(_RAW, selector="zip_override_tp[1]")
        assert len(r) == 1
        assert r[0][0] == "tp_1"
        assert r[0][1]["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 8

    def test_errors(self) -> None:
        """Missing selectors and out-of-range index raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            generate_override_configs(_RAW, selector="override_nope")
        with pytest.raises(ValueError):
            generate_override_configs(_RAW, selector="zip_override_nonexistent")
        with pytest.raises(ValueError, match="out of range"):
            generate_override_configs(_RAW, selector="zip_override_tp[5]")

    def test_override_explicit_name(self) -> None:
        """An explicit 'name' in override_* is used instead of auto-generated name."""
        raw = {
            "base": {"name": "base-job", "resources": {"decode_nodes": 8}},
            "override_custom": {"name": "my-custom-name", "resources": {"decode_nodes": 4}},
        }
        variants = generate_override_configs(raw)
        assert len(variants) == 1
        assert variants[0][1]["name"] == "my-custom-name"
        assert variants[0][1]["resources"]["decode_nodes"] == 4

        # Selector form also respects explicit name
        r = generate_override_configs(raw, selector="override_custom")
        assert r[0][1]["name"] == "my-custom-name"


# =============================================================================
# TestWildcardSelector
# =============================================================================


class TestWildcardSelector:
    def test_glob_matches_subset(self) -> None:
        """*maxtpt* matches all keys containing 'maxtpt', both override and zip."""
        variants = generate_override_configs(_RAW_MULTI, selector="*maxtpt*")
        assert [s for s, _ in variants] == ["maxtpt_1p1d", "maxtpt_1p2d"]
        assert variants[0][1]["name"] == "max-1p1d"
        assert variants[1][1]["resources"]["decode_nodes"] == 2

    def test_glob_matches_override_and_zip(self) -> None:
        """*scale* matches both override_* and zip_override_* keys uniformly."""
        raw = {
            "base": {"name": "job"},
            "override_scale_big": {"resources": {"decode_nodes": 8}},
            "zip_override_scale": {"resources": {"decode_nodes": [1, 2]}},
        }
        variants = generate_override_configs(raw, selector="*scale*")
        suffixes = [s for s, _ in variants]
        assert "scale_big" in suffixes   # from override_scale_big
        assert "scale_0" in suffixes     # from zip_override_scale
        assert "scale_1" in suffixes

    def test_glob_all_overrides(self) -> None:
        """override_* returns all override variants (base excluded)."""
        variants = generate_override_configs(_RAW_MULTI, selector="override_*")
        assert [s for s, _ in variants] == ["lowlat_1d", "maxtpt_1p1d", "maxtpt_1p2d"]

    def test_glob_all_zip(self) -> None:
        """zip_override_* returns all zip variants across all groups."""
        variants = generate_override_configs(_RAW_MULTI, selector="zip_override_*")
        suffixes = [s for s, _ in variants]
        assert suffixes == ["mem_0", "mem_1", "scale_0", "scale_1", "scale_2"]

    def test_glob_base_excluded(self) -> None:
        """base is never matched by wildcards — only reachable via :base."""
        variants = generate_override_configs(_RAW_MULTI, selector="*")
        suffixes = [s for s, _ in variants]
        assert "base" not in suffixes

    def test_no_match_raises(self) -> None:
        """A pattern that matches nothing raises ValueError."""
        with pytest.raises(ValueError, match="No variants match"):
            generate_override_configs(_RAW_MULTI, selector="*nonexistent*")

    def test_parse_config_arg_accepts_wildcard(self) -> None:
        """parse_config_arg passes any wildcard selector through."""
        _, sel = parse_config_arg("config.yaml:*mtp*")
        assert sel == "*mtp*"
        _, sel = parse_config_arg("config.yaml:override_maxtpt*")
        assert sel == "override_maxtpt*"


# =============================================================================
# TestIsOverrideConfig
# =============================================================================


class TestIsOverrideConfig:
    def test_detection(self, tmp_path: Path) -> None:
        """Normal, override, base-only, and empty files are correctly classified."""
        (tmp_path / "normal.yaml").write_text(yaml.dump({"name": "test"}))
        (tmp_path / "override.yaml").write_text(yaml.dump({"base": {"name": "test"}, "override_a": {}}))
        (tmp_path / "base_only.yaml").write_text(yaml.dump({"base": {"name": "test"}}))
        (tmp_path / "empty.yaml").write_text("")

        assert is_override_config(tmp_path / "normal.yaml") is False
        assert is_override_config(tmp_path / "override.yaml") is True
        assert is_override_config(tmp_path / "base_only.yaml") is True
        assert is_override_config(tmp_path / "empty.yaml") is False


# =============================================================================
# TestExpandZipOverride
# =============================================================================

_ZIP_BASE = {
    "name": "base-job",
    "resources": {"decode_nodes": 8},
    "backend": {"sglang_config": {"prefill": {"tensor-parallel-size": 4, "trust-remote-code": True}}},
}


class TestExpandZipOverride:
    def test_zip_semantics(self) -> None:
        """Equal-length lists zip, length-1 broadcasts, list-of-list becomes literal, scalar passes through."""
        zip_dict = {
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "tensor-parallel-size": [4, 8],
                        "mem-fraction-static": [0.85],    # broadcast
                        "trust-remote-code": False,        # scalar passthrough
                    }
                }
            },
            "benchmark": {"concurrencies": [[4, 8], [4, 8, 16]]},  # list-of-list
        }
        variants = expand_zip_override("tp_sweep", zip_dict, _ZIP_BASE)

        assert len(variants) == 2
        assert [s for s, _ in variants] == ["tp_sweep_0", "tp_sweep_1"]

        v0, v1 = variants[0][1], variants[1][1]
        assert v0["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 4
        assert v1["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == 8
        assert v0["backend"]["sglang_config"]["prefill"]["mem-fraction-static"] == 0.85  # broadcast
        assert v1["backend"]["sglang_config"]["prefill"]["mem-fraction-static"] == 0.85
        assert v0["backend"]["sglang_config"]["prefill"]["trust-remote-code"] is False   # scalar
        assert v0["benchmark"]["concurrencies"] == [4, 8]    # literal list
        assert v1["benchmark"]["concurrencies"] == [4, 8, 16]
        assert v0["resources"]["decode_nodes"] == 8          # base key preserved

    def test_naming_and_immutability(self) -> None:
        """Auto-name, explicit name list, and base not mutated."""
        zip_dict = {"backend": {"sglang_config": {"prefill": {"tensor-parallel-size": [4, 8]}}}}
        base = {"name": "base-job", "resources": {"decode_nodes": 8}}

        # auto-name
        variants = expand_zip_override("tp_sweep", zip_dict, base)
        assert variants[0][1]["name"] == "base-job_tp_sweep_0"
        assert variants[1][1]["name"] == "base-job_tp_sweep_1"
        assert base["resources"]["decode_nodes"] == 8  # not mutated

        # explicit name list
        named_dict = {"name": ["job-tp4", "job-tp8"], **zip_dict}
        variants = expand_zip_override("tp_sweep", named_dict, base)
        assert variants[0][1]["name"] == "job-tp4"
        assert variants[1][1]["name"] == "job-tp8"

    def test_errors(self) -> None:
        """Incompatible lengths and no lists both raise ValueError."""
        with pytest.raises(ValueError, match="Incompatible zip lengths"):
            expand_zip_override("bad", {
                "backend": {
                    "sglang_config": {
                        "prefill": {"tensor-parallel-size": [4, 8]},
                        "decode": {"tensor-parallel-size": [4, 8, 16]},
                    }
                }
            }, _ZIP_BASE)

        with pytest.raises(ValueError, match="no list values"):
            expand_zip_override("empty", {"backend": {"tp": 4}}, _ZIP_BASE)

        with pytest.raises(ValueError, match="empty list"):
            expand_zip_override("zero", {"backend": {"tp": []}}, _ZIP_BASE)


# =============================================================================
# TestSubmitOverride (E2E with mocked SLURM)
# =============================================================================

MINIMAL_CONFIG = {
    "name": "test-job",
    "model": {"path": "/models/test-model", "container": "test-container.sqsh", "precision": "fp8"},
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


def _write_config(tmp_path: Path, extra: dict[str, Any] | None = None, filename: str = "test.yaml") -> Path:
    raw = {"base": {**MINIMAL_CONFIG}, **(extra or {})}
    path = tmp_path / filename
    path.write_text(yaml.dump(raw, default_flow_style=False))
    return path


class TestSubmitOverride:
    def test_dry_run(self, tmp_path: Path, capsys: Any) -> None:
        """Dry-run shows correct variant counts for base-only, overrides, zip, and selectors."""
        cfg = _write_config(tmp_path, {
            "override_small": {"resources": {"decode_nodes": 2}},
            "zip_override_tp": {"resources": {"decode_nodes": [1, 2]}},
        })

        with patch("srtctl.cli.submit.load_cluster_config", return_value=None):
            # all variants (no selector): override_small + zip_tp_0 + zip_tp_1 = 3 (base excluded)
            submit_override(cfg, dry_run=True)
            assert "3 variants" in capsys.readouterr().out

            # zip group only
            submit_override(cfg, selector="zip_override_tp", dry_run=True)
            assert "2 variants" in capsys.readouterr().out

            # single zip variant
            submit_override(cfg, selector="zip_override_tp[0]", dry_run=True)
            assert "1 variant" in capsys.readouterr().out

            # single override
            submit_override(cfg, selector="override_small", dry_run=True)
            out = capsys.readouterr().out
            assert "1 variant" in out
            assert "test-job_small" in out

    def test_sbatch_call_counts(self, tmp_path: Path) -> None:
        """submit_override calls sbatch the right number of times for each selector."""
        cfg = _write_config(tmp_path, {
            "override_small": {"resources": {"decode_nodes": 2}},
            "zip_override_tp": {"resources": {"decode_nodes": [1, 2]}},
        })

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"

        def sbatch_count(selector: str | None = None) -> int:
            with (
                patch("srtctl.cli.submit.load_cluster_config", return_value=None),
                patch("subprocess.run", return_value=mock_result) as mock_run,
                patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
                patch("srtctl.cli.submit.create_job_record"),
            ):
                submit_override(cfg, selector=selector, output_dir=tmp_path)
            return sum(1 for c in mock_run.call_args_list if c[0][0][0] == "sbatch")

        assert sbatch_count() == 3               # small + tp_0 + tp_1 (base excluded by default)
        assert sbatch_count("base") == 1
        assert sbatch_count("override_small") == 1
        assert sbatch_count("zip_override_tp") == 2
        assert sbatch_count("zip_override_tp[1]") == 1
