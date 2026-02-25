# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parameter sweep functionality."""

import pytest

from srtctl.core.sweep import expand_template, generate_sweep_configs


class TestExpandTemplate:
    """Tests for expand_template function."""

    def test_simple_string_replacement(self):
        """Test basic placeholder replacement in a string."""
        result = expand_template("{foo}", {"foo": "bar"})
        assert result == "bar"

    def test_string_with_surrounding_text(self):
        """Test placeholder embedded in text."""
        result = expand_template("prefix-{val}-suffix", {"val": "123"})
        assert result == "prefix-123-suffix"

    def test_multiple_placeholders_in_string(self):
        """Test multiple placeholders in same string."""
        result = expand_template("{a}-{b}", {"a": "x", "b": "y"})
        assert result == "x-y"

    def test_numeric_value(self):
        """Test that numeric values are converted to strings."""
        result = expand_template("{num}", {"num": 42})
        assert result == "42"

    def test_float_value(self):
        """Test float value conversion."""
        result = expand_template("{val}", {"val": 0.85})
        assert result == "0.85"

    def test_nested_dict(self):
        """Test placeholder replacement in nested dicts."""
        template = {
            "outer": {
                "inner": "{val}",
                "other": "static",
            }
        }
        result = expand_template(template, {"val": "replaced"})
        assert result == {
            "outer": {
                "inner": "replaced",
                "other": "static",
            }
        }

    def test_list_of_strings(self):
        """Test placeholder replacement in lists."""
        template = ["{a}", "{b}", "static"]
        result = expand_template(template, {"a": "1", "b": "2"})
        assert result == ["1", "2", "static"]

    def test_deeply_nested_structure(self):
        """Test deeply nested dict/list structures."""
        template = {
            "level1": {
                "level2": {
                    "items": ["{x}", "{y}"],
                    "value": "{z}",
                }
            }
        }
        result = expand_template(template, {"x": "a", "y": "b", "z": "c"})
        assert result["level1"]["level2"]["items"] == ["a", "b"]
        assert result["level1"]["level2"]["value"] == "c"

    def test_list_value_as_whole_placeholder(self):
        """Test that list values replace entire placeholder."""
        result = expand_template("{items}", {"items": [1, 2, 3]})
        assert result == [1, 2, 3]

    def test_list_value_embedded_in_string(self):
        """Test that list values become comma-separated when embedded."""
        result = expand_template("values: {items}", {"items": [1, 2, 3]})
        assert result == "values: 1,2,3"

    def test_no_placeholder_unchanged(self):
        """Test that strings without placeholders are unchanged."""
        result = expand_template("no placeholders here", {"foo": "bar"})
        assert result == "no placeholders here"

    def test_non_string_passthrough(self):
        """Test that non-string primitives pass through unchanged."""
        assert expand_template(42, {"x": "y"}) == 42
        assert expand_template(3.14, {"x": "y"}) == 3.14
        assert expand_template(True, {"x": "y"}) is True
        assert expand_template(None, {"x": "y"}) is None

    def test_unused_values_ignored(self):
        """Test that extra values in dict are ignored."""
        result = expand_template("{a}", {"a": "1", "b": "2", "c": "3"})
        assert result == "1"

    def test_missing_placeholder_unchanged(self):
        """Test that unmatched placeholders remain as-is."""
        result = expand_template("{missing}", {"other": "value"})
        assert result == "{missing}"


class TestGenerateSweepConfigs:
    """Tests for generate_sweep_configs function."""

    def test_missing_sweep_section_raises(self):
        """Test that missing sweep section raises ValueError."""
        config = {"name": "test", "model": {}}
        with pytest.raises(ValueError, match="must have 'sweep' section"):
            generate_sweep_configs(config)

    def test_sweep_must_be_list(self):
        """Test that sweep section must be a list."""
        config = {
            "name": "test",
            "model": {"path": "model", "container": "c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "prefill_nodes": 1, "decode_nodes": 1},
            "sweep": {"tokens": [1024, 2048]},
        }
        with pytest.raises(ValueError, match="non-empty list"):
            generate_sweep_configs(config)

    def test_sweep_entries_must_be_dicts(self):
        """Test that each sweep entry must be a dict."""
        config = {
            "name": "test",
            "model": {"path": "model", "container": "c.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "prefill_nodes": 1, "decode_nodes": 1},
            "sweep": ["not_a_dict"],
        }
        with pytest.raises(ValueError, match="must be a dict"):
            generate_sweep_configs(config)

    def test_single_param_sweep(self):
        """Test sweep with single varying parameter."""
        config = {
            "name": "test",
            "model": {
                "path": "model",
                "container": "container.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "prefill_nodes": 1,
                "decode_nodes": 1,
            },
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "max-total-tokens": "{tokens}",
                    },
                    "decode": {},
                }
            },
            "sweep": [
                {"tokens": 1024},
                {"tokens": 2048},
                {"tokens": 4096},
            ],
        }
        results = generate_sweep_configs(config)

        assert len(results) == 3

        params = [r[1] for r in results]
        assert params[0] == {"tokens": 1024}
        assert params[1] == {"tokens": 2048}
        assert params[2] == {"tokens": 4096}

    def test_correlated_params(self):
        """Test sweep with correlated parameters (the whole point)."""
        config = {
            "name": "test",
            "model": {
                "path": "model",
                "container": "container.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "gb200",
                "prefill_nodes": "{prefill_nodes}",
                "decode_nodes": 1,
                "gpus_per_node": 4,
            },
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "tensor-parallel-size": "{tp_size}",
                        "expert-parallel-size": "{ep_size}",
                    },
                    "decode": {},
                }
            },
            "sweep": [
                {"prefill_nodes": 1, "tp_size": 4, "ep_size": 1},
                {"prefill_nodes": 2, "tp_size": 8, "ep_size": 8},
            ],
        }
        results = generate_sweep_configs(config)

        assert len(results) == 2

        # Job 1: 1 node, TP=4, EP=1
        cfg1 = results[0][0]
        assert cfg1["resources"]["prefill_nodes"] == 1
        assert cfg1["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == "4"
        assert cfg1["backend"]["sglang_config"]["prefill"]["expert-parallel-size"] == "1"

        # Job 2: 2 nodes, TP=8, EP=8
        cfg2 = results[1][0]
        assert cfg2["resources"]["prefill_nodes"] == 2
        assert cfg2["backend"]["sglang_config"]["prefill"]["tensor-parallel-size"] == "8"
        assert cfg2["backend"]["sglang_config"]["prefill"]["expert-parallel-size"] == "8"

    def test_sweep_removes_sweep_section(self):
        """Test that generated configs don't have sweep section."""
        config = {
            "name": "test",
            "model": {
                "path": "model",
                "container": "container.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "prefill_nodes": 1,
                "decode_nodes": 1,
            },
            "backend": {
                "sglang_config": {
                    "prefill": {},
                    "decode": {},
                }
            },
            "sweep": [
                {"x": 1},
            ],
        }
        results = generate_sweep_configs(config)
        generated_config = results[0][0]

        assert "sweep" not in generated_config

    def test_unique_names_generated(self):
        """Test that each config gets a unique name."""
        config = {
            "name": "base",
            "model": {
                "path": "model",
                "container": "container.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "prefill_nodes": 1,
                "decode_nodes": 1,
            },
            "backend": {
                "sglang_config": {
                    "prefill": {},
                    "decode": {},
                }
            },
            "sweep": [
                {"val": 100},
                {"val": 200},
            ],
        }
        results = generate_sweep_configs(config)

        names = [r[0]["name"] for r in results]
        assert len(names) == len(set(names)), "Names should be unique"
        assert "base_val100" in names
        assert "base_val200" in names

    def test_placeholder_substitution_in_generated_config(self):
        """Test that placeholders are actually replaced in output."""
        config = {
            "name": "test",
            "model": {
                "path": "model",
                "container": "container.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "prefill_nodes": 1,
                "decode_nodes": 1,
            },
            "backend": {
                "sglang_config": {
                    "prefill": {
                        "mem-fraction-static": "{mem}",
                    },
                    "decode": {},
                }
            },
            "sweep": [
                {"mem": 0.85},
                {"mem": 0.90},
            ],
        }
        results = generate_sweep_configs(config)

        config1 = results[0][0]
        config2 = results[1][0]

        prefill1 = config1["backend"]["sglang_config"]["prefill"]
        prefill2 = config2["backend"]["sglang_config"]["prefill"]

        assert prefill1["mem-fraction-static"] == "0.85"
        assert prefill2["mem-fraction-static"] == "0.9"
