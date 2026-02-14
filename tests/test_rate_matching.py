"""Tests for the tools/rate_matching module.

Covers: CTX processing, GEN processing, rate-matching math, Pareto frontier,
schema validation, config generation, state management, orchestrator
resilience (signal handling, reconciliation, atomic saves, detach mode),
and engine parser registration (TRT-LLM, vLLM, SGLang).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure the rate_matching package is importable
_RATE_MATCHING_DIR = Path(__file__).resolve().parent.parent / "tools" / "rate_matching"
if str(_RATE_MATCHING_DIR) not in sys.path:
    sys.path.insert(0, str(_RATE_MATCHING_DIR))

# Import parser modules so they self-register via decorators (same as run_sweep.py)
import process_ctx_results as _ctx_mod  # noqa: F401
import process_gen_results as _gen_mod  # noqa: F401
import process_ctx_results_vllm as _vllm_ctx_mod  # noqa: F401
import process_gen_results_vllm as _vllm_gen_mod  # noqa: F401
import process_ctx_results_sglang as _sglang_ctx_mod  # noqa: F401
import process_gen_results_sglang as _sglang_gen_mod  # noqa: F401


# =====================================================================
# 7A. CTX processing tests
# =====================================================================

class TestCTXProcessing:
    """Tests for process_ctx_results.process_ctx_data."""

    @staticmethod
    def _make_entry(
        iter_num: int,
        global_rank: int = 0,
        rank: int = 0,
        num_ctx_requests: int = 8,
        num_ctx_tokens: int = 8192,
        num_generation_tokens: int = 0,
        prev_device_step_time_ms: float | None = 50.0,
        num_scheduled_requests: int = 8,
    ) -> dict:
        return {
            "iter": iter_num,
            "global_rank": global_rank,
            "rank": rank,
            "num_ctx_requests": num_ctx_requests,
            "num_ctx_tokens": num_ctx_tokens,
            "num_generation_tokens": num_generation_tokens,
            "prev_device_step_time_ms": prev_device_step_time_ms,
            "num_scheduled_requests": num_scheduled_requests,
            "current_requests": 8,
            "total_requests": 100,
            "host_step_time_ms": 55.0,
            "timestamp": "2025-01-01T00:00:00",
        }

    def _make_data(self, n: int = 25, **overrides) -> list[dict]:
        """Create *n* clean CTX iteration entries."""
        return [self._make_entry(iter_num=i, **overrides) for i in range(n)]

    def test_pure_prefill_filtering(self):
        """Only iterations with num_generation_tokens == 0 are kept."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(25)
        # Inject some non-prefill rows
        data[5]["num_generation_tokens"] = 10
        data[10]["num_generation_tokens"] = 5

        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" not in result

    def test_warmup_cooldown_trim(self):
        """First 2 and last 2 iterations are trimmed."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(25)
        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        # After warmup/cooldown: 25 - 4 = 21 remaining (before outlier filter)
        assert "error" not in result
        # The final count should reflect trimming
        assert result["num_iterations"] <= 23  # at most 25-2 (after trim + outlier)

    def test_num_ctx_requests_threshold(self):
        """Only fully-packed iterations pass when max_batch_size is set."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(25, num_ctx_requests=8)
        # Set some rows below threshold
        for i in range(3):
            data[i + 5]["num_ctx_requests"] = 4

        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" not in result
        assert result["threshold_used"] == 8

    def test_outlier_filtering(self):
        """Rows outside median +/-20% are removed."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(30, prev_device_step_time_ms=50.0)
        # Add outliers that are clearly outside 20%
        data[10]["prev_device_step_time_ms"] = 200.0
        data[11]["prev_device_step_time_ms"] = 10.0

        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" not in result
        # Outliers should have been removed
        assert result["num_iterations"] < 28

    def test_request_rate_calculation(self):
        """request_rate = sum(num_ctx_requests) / sum(prev_device_step_time_s)."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(25, num_ctx_requests=8, prev_device_step_time_ms=100.0)
        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" not in result
        # request_rate should be 8 / 0.1 = 80 req/s (approximately)
        assert result["request_rate_req_per_s"] == pytest.approx(80.0, rel=0.1)

    def test_iter_restart_detection(self):
        """If iter numbers restart (decrease), only the last run is kept."""
        from process_ctx_results import process_ctx_data

        # First run: iters 0-24, then restart: iters 0-24
        data = self._make_data(25)
        data += [self._make_entry(iter_num=i) for i in range(25)]

        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        # Should use only the second run (25 entries)
        assert "error" not in result

    def test_minimum_iterations_threshold_error(self):
        """Error when fewer than 15 iterations remain after filtering."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(10)  # Too few
        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" in result
        assert "minimum 15" in result["error"].lower() or "insufficient" in result["error"].lower()

    def test_na_prev_device_step_time(self):
        """Rows with N/A (None) prev_device_step_time are dropped."""
        from process_ctx_results import process_ctx_data

        data = self._make_data(25)
        data[0]["prev_device_step_time_ms"] = None
        data[1]["prev_device_step_time_ms"] = None

        result = process_ctx_data(data, isl=1024, max_batch_size=8)
        assert "error" not in result

    def test_multi_rank_dep_scaling(self):
        """In DEP mode (ctx_dep=True), request_rate is multiplied by num_ranks."""
        from process_ctx_results import process_ctx_data

        # Multi-rank data: each iter has entries from both ranks, with iter
        # values strictly increasing to avoid the restart detector.
        # Real TRT-LLM logs interleave ranks per iter like:
        #   iter=0 rank=0, iter=1 rank=0, iter=1 rank=1, iter=2 rank=0, ...
        # We simulate this by pairing rank 0 and rank 1 for each iter
        # but ensuring iter values are monotonically increasing per rank.
        data = []
        for i in range(30):
            # rank 0 always present
            data.append(self._make_entry(iter_num=i, global_rank=0,
                                         num_ctx_requests=4,
                                         num_ctx_tokens=4096,
                                         prev_device_step_time_ms=100.0))
        # Rank 1 as a separate run (not interleaved, higher iters)
        for i in range(30, 60):
            data.append(self._make_entry(iter_num=i, global_rank=1,
                                         num_ctx_requests=4,
                                         num_ctx_tokens=4096,
                                         prev_device_step_time_ms=100.0))

        result = process_ctx_data(data, isl=1024, ctx_dep=True, max_batch_size=4)
        assert "error" not in result
        assert result["num_ranks"] == 2
        # With DEP scaling, request_rate = base_rate * 2


# =====================================================================
# 7B. GEN processing tests
# =====================================================================

class TestGENProcessing:
    """Tests for process_gen_results.process_gen_data."""

    @staticmethod
    def _make_entry(
        iter_num: int,
        global_rank: int = 0,
        rank: int = 0,
        num_scheduled_requests: int = 32,
        num_ctx_tokens: int = 0,
        num_generation_tokens: int = 32,
        prev_device_step_time_ms: float | None = 5.0,
    ) -> dict:
        return {
            "iter": iter_num,
            "global_rank": global_rank,
            "rank": rank,
            "num_scheduled_requests": num_scheduled_requests,
            "num_ctx_tokens": num_ctx_tokens,
            "num_ctx_requests": 0,
            "num_generation_tokens": num_generation_tokens,
            "prev_device_step_time_ms": prev_device_step_time_ms,
            "current_requests": 32,
            "total_requests": 200,
            "host_step_time_ms": 6.0,
            "timestamp": "2025-01-01T00:00:00",
        }

    def _make_data(self, n: int = 120, **overrides) -> list[dict]:
        """Create *n* clean GEN iteration entries."""
        return [self._make_entry(iter_num=i, **overrides) for i in range(n)]

    def test_pure_decode_filtering(self):
        """Only iterations with num_ctx_tokens == 0 are kept."""
        from process_gen_results import process_gen_data

        data = self._make_data(120)
        data[60]["num_ctx_tokens"] = 100  # inject non-decode
        result = process_gen_data(data, concurrency=32, mode="tep")
        assert "error" not in result

    def test_duplicate_merge(self):
        """Duplicate (iter, global_rank) rows are deduplicated."""
        from process_gen_results import process_gen_data

        data = self._make_data(120)
        # Duplicate iter=50
        data.append(self._make_entry(iter_num=50, prev_device_step_time_ms=5.5))
        result = process_gen_data(data, concurrency=32, mode="tep")
        assert "error" not in result

    def test_warmup_cooldown_trim(self):
        """First 50 and last 10 iterations are trimmed."""
        from process_gen_results import process_gen_data

        data = self._make_data(120)
        result = process_gen_data(data, concurrency=32, mode="tep")
        assert "error" not in result
        # 120 - 50 - 10 = 60 remaining (before outlier filter)
        assert result["num_iterations"] <= 60

    def test_tep_concurrency_matching(self):
        """TEP: match num_scheduled_requests == concurrency."""
        from process_gen_results import process_gen_data

        data = self._make_data(120, num_scheduled_requests=32, num_generation_tokens=32)
        result = process_gen_data(data, concurrency=32, mode="tep", mtp=0)
        assert "error" not in result
        assert result["concurrency"] == 32

    def test_dep_concurrency_matching(self):
        """DEP: match num_scheduled_requests == concurrency / ep_rank."""
        from process_gen_results import process_gen_data

        # DEP with tp=8: ep_rank=8, so expected_scheduled=256/8=32
        data = self._make_data(120, num_scheduled_requests=32, num_generation_tokens=32)
        result = process_gen_data(data, concurrency=256, mode="dep", tp=8, ep_rank=8, mtp=0)
        assert "error" not in result
        assert result["concurrency"] == 256

    def test_mtp_accept_rate_stp(self):
        """STP (mtp=0) returns accept_rate of 1.0."""
        from process_gen_results import get_mtp_accept_rate

        assert get_mtp_accept_rate(1024, 0) == 1.0

    def test_mtp_accept_rate_with_overrides(self):
        """MTP with overrides returns the overridden value."""
        from process_gen_results import get_mtp_accept_rate

        overrides = {1: 1.8, 2: 2.28, 3: 2.56}
        assert get_mtp_accept_rate(1024, 3, overrides=overrides) == 2.56
        assert get_mtp_accept_rate(1024, 1, overrides=overrides) == 1.8

    def test_mtp_accept_rate_missing_override_raises(self):
        """MTP without overrides raises ValueError."""
        from process_gen_results import get_mtp_accept_rate

        with pytest.raises(ValueError, match="mtp_accept_rates"):
            get_mtp_accept_rate(1024, 3)

    def test_throughput_formulas(self):
        """Verify throughput_per_user, tpot, output_throughput formulas."""
        from process_gen_results import process_gen_data

        step_ms = 5.0
        data = self._make_data(120, prev_device_step_time_ms=step_ms,
                               num_scheduled_requests=32, num_generation_tokens=32)
        result = process_gen_data(data, concurrency=32, mode="tep", mtp=0)
        assert "error" not in result

        # STP: accept_rate=1.0
        # elapsed_time_avg ~= 5ms = 0.005s
        # throughput_per_user = 1/0.005 * 1.0 = 200
        # tpot = 0.005 / 1.0 = 0.005s = 5ms
        # output_throughput = 200 * 32 = 6400
        assert result["avg_step_time_ms"] == pytest.approx(step_ms, rel=0.1)
        assert result["tpot_ms"] == pytest.approx(step_ms, rel=0.1)
        expected_tpu = 1.0 / (step_ms / 1000.0)
        assert result["throughput_per_user"] == pytest.approx(expected_tpu, rel=0.1)
        assert result["output_throughput"] == pytest.approx(expected_tpu * 32, rel=0.1)

    def test_outlier_filtering_gen(self):
        """Rows outside median +/-20% are removed."""
        from process_gen_results import process_gen_data

        data = self._make_data(120, prev_device_step_time_ms=5.0)
        data[70]["prev_device_step_time_ms"] = 50.0  # outlier
        result = process_gen_data(data, concurrency=32, mode="tep")
        assert "error" not in result


# =====================================================================
# 7C. Rate-matching math tests
# =====================================================================

class TestRateMatchingMath:
    """Tests for metrics.compute_rate_matching and compare_sol_vs_e2e."""

    def test_compute_rate_matching_basic(self):
        """Basic rate-matching with known inputs."""
        from metrics import compute_rate_matching

        ctx_result = {
            "request_rate_req_per_s": 10.0,
            "ctx_throughput_tokens_per_s": 10240.0,
            "avg_prev_device_step_time_ms": 100.0,
        }
        gen_result = {
            "output_throughput": 6400.0,
            "throughput_per_user": 200.0,
            "avg_step_time_ms": 5.0,
            "tpot_ms": 5.0,
            "concurrency": 32,
            "mode": "tep",
            "mtp_accept_rate": 1.0,
            "mtp": 0,
            "tp_size": 8,
            "batch_size": 128,
        }

        result = compute_rate_matching(
            ctx_result, gen_result, osl=1024, random_ratio=1.0,
            gpus_per_ctx_instance=8, gpus_per_gen_instance=8, max_total_gpus=64,
        )
        assert "gen_req_rate" in result
        assert "ctx_gen_inst_ratio" in result
        assert "output_tput_per_gpu" in result
        assert result["gen_req_rate"] > 0
        assert result["ctx_gen_inst_ratio"] > 0

    def test_find_best_allocation(self):
        """Known ratio should yield expected CTX:GEN split."""
        from metrics import _find_best_allocation

        # ratio=0.5 means 1 CTX per 2 GEN.  With 8 GPUs each and 64 total:
        # ctx=1, gen=2 => 8+16=24 GPUs (or could be larger)
        result = _find_best_allocation(0.5, gpus_per_ctx=8, gpus_per_gen=8, max_total_gpus=64)
        assert result["ctx_instances"] >= 1
        assert result["gen_instances"] >= 2
        assert result["total_gpus"] <= 64

    def test_find_best_allocation_1to1(self):
        """Ratio 1.0 should yield equal CTX and GEN."""
        from metrics import _find_best_allocation

        result = _find_best_allocation(1.0, gpus_per_ctx=8, gpus_per_gen=8, max_total_gpus=64)
        assert result["ctx_instances"] == result["gen_instances"]

    def test_zero_osl(self):
        """Zero osl produces zero gen_req_rate."""
        from metrics import compute_rate_matching

        ctx_result = {"request_rate_req_per_s": 10.0, "ctx_throughput_tokens_per_s": 10000.0,
                      "avg_prev_device_step_time_ms": 100.0}
        gen_result = {"output_throughput": 6400.0, "throughput_per_user": 200.0,
                      "avg_step_time_ms": 5.0, "tpot_ms": 5.0, "concurrency": 32,
                      "mode": "tep", "mtp_accept_rate": 1.0, "mtp": 0, "tp_size": 8,
                      "batch_size": 128}

        result = compute_rate_matching(ctx_result, gen_result, osl=0)
        assert result["gen_req_rate"] == 0

    def test_compare_sol_vs_e2e_basic(self):
        """Comparison returns expected structure."""
        from metrics import compare_sol_vs_e2e

        sol = {
            "tpot_ms": 5.0,
            "interactivity": 200.0,
            "output_tput_per_gpu": 100.0,
            "output_throughput": 6400.0,
        }
        e2e = {
            "output_throughput": 6200.0,
            "median_tpot_ms": 5.2,
        }
        result = compare_sol_vs_e2e(sol, e2e, total_gpus=64, gen_instances=1, per_worker_conc=32)
        assert "diff_pct" in result
        assert "pass" in result
        assert isinstance(result["pass"]["overall"], bool)

    def test_compare_sol_vs_e2e_pass(self):
        """Within tolerance should pass."""
        from metrics import compare_sol_vs_e2e

        sol = {"tpot_ms": 5.0, "interactivity": 200.0, "output_tput_per_gpu": 100.0,
               "output_throughput": 6400.0}
        e2e = {"output_throughput": 6300.0, "median_tpot_ms": 5.05}
        result = compare_sol_vs_e2e(sol, e2e, total_gpus=64)
        assert result["pass"]["overall"] is True

    def test_compare_sol_vs_e2e_fail(self):
        """Outside tolerance should fail."""
        from metrics import compare_sol_vs_e2e

        sol = {"tpot_ms": 5.0, "interactivity": 200.0, "output_tput_per_gpu": 100.0,
               "output_throughput": 6400.0}
        e2e = {"output_throughput": 3000.0, "median_tpot_ms": 10.0}
        result = compare_sol_vs_e2e(sol, e2e, total_gpus=64)
        assert result["pass"]["overall"] is False


# =====================================================================
# 7D. Pareto frontier tests
# =====================================================================

class TestParetoFrontier:
    """Tests for pareto.extract_pareto_frontier."""

    def test_basic_pareto(self):
        """Extract Pareto-optimal points from a simple set."""
        from pareto import extract_pareto_frontier

        results = [
            {"interactivity": 200, "output_tput_per_gpu": 100, "mode": "tep", "concurrency": 32},
            {"interactivity": 150, "output_tput_per_gpu": 120, "mode": "tep", "concurrency": 64},
            {"interactivity": 100, "output_tput_per_gpu": 130, "mode": "dep", "concurrency": 128},
            {"interactivity": 50, "output_tput_per_gpu": 110, "mode": "dep", "concurrency": 256},  # dominated
        ]
        frontier = extract_pareto_frontier(results)

        # Point at interactivity=50, tput=110 is dominated by interactivity=100, tput=130
        assert len(frontier) == 3
        ranks = [p["pareto_rank"] for p in frontier]
        assert ranks == [1, 2, 3]

    def test_ranking_order(self):
        """Pareto rank 1 has highest interactivity."""
        from pareto import extract_pareto_frontier

        results = [
            {"interactivity": 50, "output_tput_per_gpu": 200, "mode": "dep", "concurrency": 128},
            {"interactivity": 200, "output_tput_per_gpu": 50, "mode": "tep", "concurrency": 8},
        ]
        frontier = extract_pareto_frontier(results)
        assert frontier[0]["pareto_rank"] == 1
        assert frontier[0]["interactivity"] == 200

    def test_empty_input(self):
        """Empty input returns empty frontier."""
        from pareto import extract_pareto_frontier

        assert extract_pareto_frontier([]) == []

    def test_all_dominated(self):
        """Single dominant point when others are all dominated."""
        from pareto import extract_pareto_frontier

        results = [
            {"interactivity": 200, "output_tput_per_gpu": 200, "mode": "tep", "concurrency": 32},
            {"interactivity": 100, "output_tput_per_gpu": 100, "mode": "dep", "concurrency": 64},
            {"interactivity": 50, "output_tput_per_gpu": 50, "mode": "dep", "concurrency": 128},
        ]
        frontier = extract_pareto_frontier(results)
        # Only the first point is Pareto-optimal (highest in both dimensions)
        assert len(frontier) == 1
        assert frontier[0]["interactivity"] == 200

    def test_error_results_filtered(self):
        """Results with 'error' key are excluded from Pareto."""
        from pareto import extract_pareto_frontier

        results = [
            {"error": "insufficient data"},
            {"interactivity": 200, "output_tput_per_gpu": 100, "mode": "tep", "concurrency": 32},
        ]
        frontier = extract_pareto_frontier(results)
        assert len(frontier) == 1


# =====================================================================
# 7E. Schema validation tests
# =====================================================================

class TestSchemaValidation:
    """Tests for schema.py validation logic."""

    def test_gen_sweep_group_zip_expansion(self):
        """Zip expansion produces one item per parameter list entry."""
        from schema import GenSweepGroup

        group = GenSweepGroup(
            expansion="zip",
            parameters={
                "concurrency": [[8, 16, 32]],
                "batch_size": [128],
            },
            defaults={"mode": "tep", "tp_size": 8, "mtp_num": 0},
        )
        items = group.expand()
        assert len(items) == 1  # zip of length-1 lists
        assert items[0].batch_size == 128

    def test_gen_sweep_group_grid_expansion(self):
        """Grid expansion produces the Cartesian product."""
        from schema import GenSweepGroup

        group = GenSweepGroup(
            expansion="grid",
            parameters={
                "batch_size": [64, 128],
                "concurrency": [8, 32],
            },
            defaults={"mode": "tep", "tp_size": 8, "mtp_num": 0},
        )
        items = group.expand()
        assert len(items) == 4  # 2 x 2

    def test_validate_mtp_accept_rates_required(self):
        """mtp_accept_rates must be present when MTP is used."""
        from schema import RateMatchingSweepConfig

        config_data = {
            "name": "test",
            "model": {"path": "m", "container": "c", "precision": "fp8"},
            "workload": {"isl": 1024, "osl": 1024},
            "resources": {"gpu_type": "h200", "gpus_per_node": 8,
                          "ctx_gpus_per_instance": 8, "gen_gpus_per_instance": 8},
            "gen_sweep": [
                {"mode": "tep", "batch_size": 128, "concurrency": 32, "tp_size": 8, "mtp_num": 3},
            ],
        }
        with pytest.raises(ValueError, match="mtp_accept_rates"):
            RateMatchingSweepConfig(**config_data)

    def test_gen_sweep_item_defaults_mtp(self):
        """max_num_tokens defaults to batch_size * (1 + mtp_num) for MTP."""
        from schema import GenSweepItem

        item = GenSweepItem(mode="tep", batch_size=128, concurrency=32, tp_size=8, mtp_num=3)
        assert item.max_num_tokens == 128 * 4  # 128 * (1+3)

    def test_gen_sweep_item_defaults_stp(self):
        """max_num_tokens defaults to batch_size for STP."""
        from schema import GenSweepItem

        item = GenSweepItem(mode="tep", batch_size=128, concurrency=32, tp_size=8, mtp_num=0)
        assert item.max_num_tokens == 128

    def test_load_sweep_config_example(self):
        """The example YAML loads and validates successfully."""
        from schema import load_sweep_config

        yaml_path = _RATE_MATCHING_DIR / "h200_1k1k_mtp_sweep.yaml"
        if not yaml_path.exists():
            pytest.skip("Example YAML not found")
        cfg = load_sweep_config(str(yaml_path))
        assert cfg.name == "dsr1_1k1k_mtp"
        assert len(cfg.gen_sweep) > 0
        assert cfg.workload.isl == 1024
        assert cfg.workload.osl == 1024


# =====================================================================
# 7F. Config generation tests
# =====================================================================

class TestConfigGeneration:
    """Tests for generate_configs.py."""

    @pytest.fixture
    def sweep_cfg(self):
        """Minimal RateMatchingSweepConfig for config generation tests."""
        from schema import RateMatchingSweepConfig

        return RateMatchingSweepConfig(**{
            "name": "test_sweep",
            "model": {"path": "test_model", "container": "test:1.0", "precision": "fp8"},
            "workload": {
                "isl": 1024,
                "osl": 1024,
                "mtp_accept_rates": {1: 1.8, 2: 2.28, 3: 2.56},
            },
            "resources": {
                "gpu_type": "h200",
                "gpus_per_node": 8,
                "ctx_gpus_per_instance": 8,
                "gen_gpus_per_instance": 8,
            },
            "gen_sweep": [
                {"mode": "tep", "batch_size": 128, "concurrency": 32, "tp_size": 8, "mtp_num": 0},
            ],
        })

    def test_ctx_sol_has_random_range_ratio_1(self, sweep_cfg):
        """CTX SOL config must set random_range_ratio=1.0 for exact ISL."""
        from generate_configs import generate_ctx_sol_config

        config = generate_ctx_sol_config(sweep_cfg)
        assert config["benchmark"]["random_range_ratio"] == 1.0

    def test_gen_sol_has_req_queues_size(self, sweep_cfg):
        """GEN SOL config must set TLLM_BENCHMARK_REQ_QUEUES_SIZE matching concurrency."""
        from generate_configs import generate_gen_sol_config
        from schema import GenSweepItem

        gen_item = GenSweepItem(mode="tep", batch_size=128, concurrency=32, tp_size=8, mtp_num=0)
        config = generate_gen_sol_config(sweep_cfg, gen_item)
        decode_env = config["backend"]["decode_environment"]
        assert decode_env["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] == "32"

    def test_e2e_system_concurrency(self, sweep_cfg):
        """E2E system_concurrency = per_worker * gen_instances * multiplier."""
        from generate_configs import generate_e2e_config

        pareto_point = {
            "ctx_instances": 1,
            "gen_instances": 4,
            "concurrency": 32,
            "batch_size": 128,
            "mode": "tep",
            "tp_size": 8,
            "mtp_num": 0,
        }
        config = generate_e2e_config(sweep_cfg, pareto_point, concurrency_multiplier=1.05)
        sys_conc = int(32 * 4 * 1.05)
        assert config["benchmark"]["concurrencies"] == str(sys_conc)

    def test_cuda_graph_batch_sizes(self):
        """_cuda_graph_batch_sizes produces expected sequences."""
        from generate_configs import _cuda_graph_batch_sizes

        # max_bs=128
        sizes = _cuda_graph_batch_sizes(128)
        assert 1 in sizes
        assert 2 in sizes
        assert 4 in sizes
        assert 8 in sizes
        assert 16 in sizes
        assert 128 in sizes
        assert sizes == sorted(sizes)

        # max_bs=8 (all powers of 2)
        sizes = _cuda_graph_batch_sizes(8)
        assert sizes == [1, 2, 4, 8]

        # max_bs=32
        sizes = _cuda_graph_batch_sizes(32)
        assert 1 in sizes
        assert 16 in sizes
        assert 24 in sizes
        assert 32 in sizes


# =====================================================================
# 7G. State machine tests
# =====================================================================

class TestStateMachine:
    """Tests for state.SweepState save/load and phase transitions."""

    def test_save_load_round_trip(self):
        """State serialises and deserialises correctly."""
        from state import SweepState

        state = SweepState()
        state.sweep_name = "test_sweep"
        state.phase = "gen"
        state.ctx_result = {"request_rate_req_per_s": 10.0}
        state.gen_results = [{"concurrency": 32, "tpot_ms": 5.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            state.save(path)

            loaded = SweepState.load(path)
            assert loaded.sweep_name == "test_sweep"
            assert loaded.phase == "gen"
            assert loaded.ctx_result == {"request_rate_req_per_s": 10.0}
            assert len(loaded.gen_results) == 1

    def test_phase_transitions(self):
        """SweepState can transition through all expected phases."""
        from state import SweepState

        state = SweepState()
        phases = ["init", "ctx", "gen", "rate_match", "pareto", "e2e", "complete"]

        with tempfile.TemporaryDirectory() as tmpdir:
            state.output_dir = tmpdir

            for phase in phases:
                state.phase = phase
                state.save()
                loaded = SweepState.load(str(Path(tmpdir) / "sweep_state.json"))
                assert loaded.phase == phase

    def test_srtctl_root_persists(self):
        """srtctl_root is saved and loaded correctly."""
        from state import SweepState

        state = SweepState()
        state.srtctl_root = "/custom/path"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            state.save(path)

            loaded = SweepState.load(path)
            assert loaded.srtctl_root == "/custom/path"

    def test_job_records_persist(self):
        """Job records (ctx, gen, e2e) survive round-trip."""
        from state import SweepState

        state = SweepState()
        state.ctx_job = {"config_path": "/a.yaml", "status": "completed", "job_id": 123}
        state.gen_jobs = [{"config_path": "/b.yaml", "status": "running", "job_id": 456}]
        state.e2e_jobs = [{"config_path": "/c.yaml", "status": "pending",
                           "pareto_rank": 1, "multiplier": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "state.json")
            state.save(path)
            loaded = SweepState.load(path)
            assert loaded.ctx_job["job_id"] == 123
            assert loaded.gen_jobs[0]["status"] == "running"
            assert loaded.e2e_jobs[0]["pareto_rank"] == 1


# =====================================================================
# Parser registry tests
# =====================================================================

class TestParserRegistry:
    """Tests for the parser registry in parser_base.py."""

    def test_get_ctx_parser_trtllm(self):
        """get_ctx_parser('trtllm') returns a TrtllmCTXLogParser."""
        from parser_base import get_ctx_parser
        import process_ctx_results  # noqa: F401 — triggers registration

        parser = get_ctx_parser("trtllm")
        assert parser.__class__.__name__ == "TrtllmCTXLogParser"

    def test_get_gen_parser_trtllm(self):
        """get_gen_parser('trtllm') returns a TrtllmGENLogParser."""
        from parser_base import get_gen_parser
        import process_gen_results  # noqa: F401 — triggers registration

        parser = get_gen_parser("trtllm")
        assert parser.__class__.__name__ == "TrtllmGENLogParser"

    def test_unknown_engine_raises(self):
        """Requesting an unknown engine raises KeyError."""
        from parser_base import get_ctx_parser, get_gen_parser

        with pytest.raises(KeyError, match="vllm"):
            get_ctx_parser("vllm")
        with pytest.raises(KeyError, match="vllm"):
            get_gen_parser("vllm")


# =====================================================================
# 9B. Signal handling tests
# =====================================================================

class TestSignalHandling:
    """Tests for SIGHUP/SIGTERM/SIGINT graceful shutdown."""

    def test_graceful_shutdown_saves_state(self, tmp_path):
        """_graceful_shutdown persists state before exiting."""
        import run_sweep

        state = _make_state(tmp_path)
        state.phase = "gen"
        run_sweep._active_state = state

        with pytest.raises(SystemExit) as exc_info:
            run_sweep._graceful_shutdown(signal.SIGTERM, None)

        # Exit code = 128 + signum
        assert exc_info.value.code == 128 + signal.SIGTERM
        # State file should exist
        saved = json.loads((tmp_path / "sweep_state.json").read_text())
        assert saved["phase"] == "gen"

    def test_graceful_shutdown_no_state(self):
        """_graceful_shutdown exits cleanly even without an active state."""
        import run_sweep

        run_sweep._active_state = None

        with pytest.raises(SystemExit) as exc_info:
            run_sweep._graceful_shutdown(signal.SIGINT, None)
        assert exc_info.value.code == 128 + signal.SIGINT

    def test_install_signal_handlers(self):
        """_install_signal_handlers registers handlers for expected signals."""
        import run_sweep

        old_handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.getsignal(sig)
        if hasattr(signal, "SIGHUP"):
            old_handlers[signal.SIGHUP] = signal.getsignal(signal.SIGHUP)

        try:
            run_sweep._install_signal_handlers()

            assert signal.getsignal(signal.SIGTERM) is run_sweep._graceful_shutdown
            assert signal.getsignal(signal.SIGINT) is run_sweep._graceful_shutdown
            if hasattr(signal, "SIGHUP"):
                assert signal.getsignal(signal.SIGHUP) is run_sweep._graceful_shutdown
        finally:
            # Restore original handlers
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


# =====================================================================
# 9C/D. Reconciliation tests
# =====================================================================

def _make_state(tmp_path):
    """Helper to build a SweepState rooted in tmp_path."""
    from state import SweepState

    state = SweepState()
    state.output_dir = str(tmp_path)
    state.sweep_name = "test"
    state.srtctl_root = str(tmp_path)
    return state


class _FakeParser:
    """Minimal parser stub for reconciliation tests."""

    def __init__(self, log_dirs_with_files: set[str] | None = None):
        self._has_logs = log_dirs_with_files or set()

    def find_log(self, logs_dir):
        return Path(logs_dir) / "fake.log" if str(logs_dir) in self._has_logs else None


class TestReconciliation:
    """Tests for _reconcile_stale_jobs covering CTX, GEN, and E2E jobs."""

    def test_reconcile_ctx_job(self, tmp_path):
        """A stale CTX job with logs on disk is reconciled to 'completed'."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)
        # Simulate a CTX job stuck at "running" with job_id and output_dir
        out_dir = tmp_path / "outputs" / "111"
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "prefill.log").write_text("data")

        state.ctx_job = {
            "status": "running", "job_id": 111,
            "output_dir": str(out_dir), "config_path": "/fake.yaml",
        }

        ctx_parser = _FakeParser({str(logs_dir)})
        n = _reconcile_stale_jobs(state, ctx_parser=ctx_parser, verbose=False)

        assert n == 1
        assert state.ctx_job["status"] == "completed"

    def test_reconcile_gen_jobs(self, tmp_path):
        """Stale GEN jobs with logs on disk are reconciled."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)

        out1 = tmp_path / "outputs" / "201"
        (out1 / "logs").mkdir(parents=True)
        out2 = tmp_path / "outputs" / "202"
        (out2 / "logs").mkdir(parents=True)

        state.gen_jobs = [
            {"status": "running", "job_id": 201, "output_dir": str(out1),
             "config_path": "/g1.yaml"},
            {"status": "submitted", "job_id": 202, "output_dir": str(out2),
             "config_path": "/g2.yaml"},
            {"status": "completed", "job_id": 200, "output_dir": "/done",
             "config_path": "/g0.yaml"},
        ]

        gen_parser = _FakeParser({str(out1 / "logs"), str(out2 / "logs")})
        n = _reconcile_stale_jobs(state, gen_parser=gen_parser, verbose=False)

        assert n == 2
        assert state.gen_jobs[0]["status"] == "completed"
        assert state.gen_jobs[1]["status"] == "completed"
        assert state.gen_jobs[2]["status"] == "completed"  # unchanged

    def test_reconcile_e2e_jobs(self, tmp_path):
        """Stale E2E jobs with sa-bench results on disk are reconciled."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)

        out = tmp_path / "outputs" / "301"
        sa_dir = out / "logs" / "sa-bench_c4"
        sa_dir.mkdir(parents=True)
        (sa_dir / "results_1.json").write_text('{"throughput": 100}')

        state.e2e_jobs = [
            {"status": "running", "job_id": 301, "output_dir": str(out),
             "config_path": "/e.yaml", "pareto_rank": 1, "multiplier": 1.0},
        ]

        n = _reconcile_stale_jobs(state, verbose=False)
        assert n == 1
        assert state.e2e_jobs[0]["status"] == "completed"

    def test_no_reconciliation_without_results(self, tmp_path):
        """Jobs without results on disk are NOT reconciled."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)

        # CTX job: running, but no logs exist
        state.ctx_job = {
            "status": "running", "job_id": 999,
            "output_dir": str(tmp_path / "no_such_dir"),
            "config_path": "/fake.yaml",
        }
        state.gen_jobs = [
            {"status": "pending", "job_id": 998,
             "output_dir": str(tmp_path / "nope"),
             "config_path": "/g.yaml"},
        ]

        ctx_parser = _FakeParser()  # no dirs have logs
        gen_parser = _FakeParser()
        n = _reconcile_stale_jobs(
            state, ctx_parser=ctx_parser, gen_parser=gen_parser, verbose=False,
        )
        assert n == 0
        assert state.ctx_job["status"] == "running"
        assert state.gen_jobs[0]["status"] == "pending"

    def test_reconcile_skips_completed_jobs(self, tmp_path):
        """Already-completed jobs are not touched by reconciliation."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)
        state.ctx_job = {
            "status": "completed", "job_id": 100,
            "output_dir": "/done", "config_path": "/c.yaml",
        }
        n = _reconcile_stale_jobs(state, verbose=False)
        assert n == 0

    def test_reconcile_missing_job_id_skipped(self, tmp_path):
        """Jobs without a job_id (never submitted) are not reconciled."""
        from run_sweep import _reconcile_stale_jobs

        state = _make_state(tmp_path)
        state.e2e_jobs = [
            {"status": "pending", "config_path": "/e.yaml",
             "pareto_rank": 1, "multiplier": 1.0},
        ]
        n = _reconcile_stale_jobs(state, verbose=False)
        assert n == 0


# =====================================================================
# 9E. Atomic state save tests
# =====================================================================

class TestAtomicSave:
    """Tests for atomic state persistence."""

    def test_save_creates_valid_json(self, tmp_path):
        """state.save() produces a valid JSON file."""
        state = _make_state(tmp_path)
        state.phase = "gen"
        state.save()

        data = json.loads((tmp_path / "sweep_state.json").read_text())
        assert data["phase"] == "gen"

    def test_save_is_atomic_no_leftover_tmp(self, tmp_path):
        """After save(), no .tmp files remain in the directory."""
        state = _make_state(tmp_path)
        state.save()

        tmp_files = list(tmp_path.glob(".sweep_state_*.tmp"))
        assert tmp_files == []

    def test_save_overwrites_existing(self, tmp_path):
        """Successive saves overwrite the same file."""
        state = _make_state(tmp_path)
        state.phase = "init"
        state.save()
        state.phase = "complete"
        state.save()

        data = json.loads((tmp_path / "sweep_state.json").read_text())
        assert data["phase"] == "complete"

    def test_save_cleans_up_on_error(self, tmp_path):
        """If writing fails, no temp file is left behind."""
        state = _make_state(tmp_path)

        # Force a write failure by making the target directory read-only.
        # mkstemp will fail because the directory is not writable.
        tmp_path.chmod(0o444)
        try:
            with pytest.raises(OSError):
                state.save()

            tmp_files = list(tmp_path.glob(".sweep_state_*.tmp"))
            assert tmp_files == []
        finally:
            tmp_path.chmod(0o755)

    def test_save_to_explicit_path(self, tmp_path):
        """save(path=...) writes to the specified path atomically."""
        state = _make_state(tmp_path)
        target = tmp_path / "custom_state.json"
        state.save(path=str(target))

        assert target.exists()
        data = json.loads(target.read_text())
        assert data["sweep_name"] == "test"


# =====================================================================
# 9F. Detach mode / CLI tests
# =====================================================================

class TestCLI:
    """Tests for CLI helpers (multiplexer detection, detach mode)."""

    def test_inside_multiplexer_tmux(self):
        """_inside_multiplexer returns True when TMUX is set."""
        from cli import _inside_multiplexer

        with mock.patch.dict(os.environ, {"TMUX": "/tmp/tmux-1000/default,12345,0"}):
            assert _inside_multiplexer() is True

    def test_inside_multiplexer_screen(self):
        """_inside_multiplexer returns True when STY (screen) is set."""
        from cli import _inside_multiplexer

        with mock.patch.dict(os.environ, {"STY": "12345.pts-0.host"}, clear=False):
            assert _inside_multiplexer() is True

    def test_outside_multiplexer(self):
        """_inside_multiplexer returns False in a plain terminal."""
        from cli import _inside_multiplexer

        env = {k: v for k, v in os.environ.items()
               if k not in ("TMUX", "STY")}
        with mock.patch.dict(os.environ, env, clear=True):
            # Also need stdin to be a tty for this to return False
            with mock.patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                assert _inside_multiplexer() is False

    def test_detach_flag_exists(self):
        """The 'run' subcommand accepts --detach."""
        from cli import main

        # Parse known args to verify --detach is accepted
        with mock.patch("sys.argv", ["srtctl-rate-match", "run",
                                     "-f", "sweep.yaml", "--detach"]):
            with mock.patch("cli.cmd_run") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.detach is True

    def test_add_e2e_subcommand_exists(self):
        """The 'add-e2e' subcommand accepts --multipliers and -o."""
        from cli import main

        with mock.patch("sys.argv", ["srtctl-rate-match", "add-e2e",
                                     "-o", "/tmp/sweep", "--multipliers",
                                     "0.95", "1.10"]):
            with mock.patch("cli.cmd_add_e2e") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.multipliers == [0.95, 1.10]
                assert args.output == "/tmp/sweep"


# =====================================================================
# 10A. State backup tests
# =====================================================================

class TestStateBackup:
    """Tests for SweepState.save_backup()."""

    def test_backup_creates_timestamped_file(self, tmp_path):
        """save_backup() creates a .bak file with timestamp."""
        state = _make_state(tmp_path)
        state.phase = "complete"
        state.save()

        backup = state.save_backup()
        assert Path(backup).exists()
        assert "sweep_state.json.bak." in backup
        # Backup content matches original
        original = json.loads((tmp_path / "sweep_state.json").read_text())
        backed_up = json.loads(Path(backup).read_text())
        assert original["phase"] == backed_up["phase"]

    def test_backup_fails_without_state_file(self, tmp_path):
        """save_backup() raises if there's no state file to back up."""
        state = _make_state(tmp_path)
        with pytest.raises(FileNotFoundError):
            state.save_backup()


# =====================================================================
# 10B. Overwrite guard tests
# =====================================================================

class TestOverwriteGuard:
    """Tests for run_sweep() refusing to overwrite existing state."""

    def test_run_sweep_refuses_overwrite(self, tmp_path):
        """run_sweep() raises if state exists and resume=False."""
        from run_sweep import run_sweep

        # Create an existing state file
        state = _make_state(tmp_path)
        state.phase = "complete"
        state.save()

        # Need a valid config path — use the example sweep YAML
        config_path = str(_RATE_MATCHING_DIR / "h200_1k1k_mtp_sweep.yaml")

        with pytest.raises(RuntimeError, match="Sweep state already exists"):
            run_sweep(
                config_path=config_path,
                output_dir=str(tmp_path),
                resume=False,
                dry_run=False,
            )

    def test_run_sweep_allows_dry_run_over_existing(self, tmp_path):
        """run_sweep() allows dry_run=True even with existing state (no-op guard)."""
        from run_sweep import run_sweep

        # dry_run should not raise — it's safe
        state = _make_state(tmp_path)
        state.phase = "init"
        state.save()

        config_path = str(_RATE_MATCHING_DIR / "h200_1k1k_mtp_sweep.yaml")

        # Should not raise (dry_run bypasses the guard)
        result = run_sweep(
            config_path=config_path,
            output_dir=str(tmp_path),
            resume=False,
            dry_run=True,
        )
        assert result is not None


# =====================================================================
# 10C. add_e2e_jobs tests
# =====================================================================

def _make_completed_sweep_state(tmp_path):
    """Build a realistic completed sweep state with a Pareto frontier."""
    state = _make_state(tmp_path)
    state.sweep_name = "test_1k1k"
    state.phase = "complete"
    state.sweep_config_path = str(_RATE_MATCHING_DIR / "h200_1k1k_mtp_sweep.yaml")

    # Minimal Pareto frontier (2 points)
    state.pareto_frontier = [
        {
            "pareto_rank": 1,
            "concurrency": 64,
            "ctx_instances": 1,
            "gen_instances": 3,
            "mode": "dep",
            "tp_size": 8,
            "batch_size": 512,
            "mtp_num": 3,
            "eplb_num_slots": 0,
            "ratio_str": "1:3",
            "interactivity": 2.5,
            "output_tput_per_gpu": 150.0,
            "tpot_ms": 12.0,
            "avg_step_time_ms": 24.0,
            "mtp_accept_rate": 0.85,
            "max_num_tokens": 512,
            "gpu_memory_fraction": 0.9,
            "total_gpus": 32,
        },
        {
            "pareto_rank": 2,
            "concurrency": 128,
            "ctx_instances": 1,
            "gen_instances": 7,
            "mode": "dep",
            "tp_size": 8,
            "batch_size": 512,
            "mtp_num": 3,
            "eplb_num_slots": 0,
            "ratio_str": "1:7",
            "interactivity": 5.0,
            "output_tput_per_gpu": 200.0,
            "tpot_ms": 20.0,
            "avg_step_time_ms": 40.0,
            "mtp_accept_rate": 0.85,
            "max_num_tokens": 512,
            "gpu_memory_fraction": 0.9,
            "total_gpus": 64,
        },
    ]

    # Existing E2E configs/jobs for multiplier 1.0
    state.e2e_configs = [
        {"config_path": "/fake/r1_1.0x.yaml", "pareto_rank": 1,
         "multiplier": 1.0, "per_worker_concurrency": 64,
         "system_concurrency": 192, "config_name": "r1_1.0x"},
        {"config_path": "/fake/r2_1.0x.yaml", "pareto_rank": 2,
         "multiplier": 1.0, "per_worker_concurrency": 128,
         "system_concurrency": 896, "config_name": "r2_1.0x"},
    ]
    state.e2e_jobs = [
        {"config_path": "/fake/r1_1.0x.yaml", "pareto_rank": 1,
         "multiplier": 1.0, "status": "completed", "job_id": 500,
         "config_name": "r1_1.0x"},
        {"config_path": "/fake/r2_1.0x.yaml", "pareto_rank": 2,
         "multiplier": 1.0, "status": "completed", "job_id": 501,
         "config_name": "r2_1.0x"},
    ]

    state.save()
    return state


class TestAddE2E:
    """Tests for add_e2e_jobs() — incremental E2E addition."""

    def test_add_new_multiplier_dry_run(self, tmp_path):
        """Dry-run adds configs to state without submitting."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        state = add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[0.95],
            dry_run=True,
            verbose=False,
        )

        # Should have 2 original + 2 new configs (1 per Pareto point)
        assert len(state.e2e_configs) == 4
        assert len(state.e2e_jobs) == 4

        # New jobs should be pending
        new_jobs = [j for j in state.e2e_jobs if j["multiplier"] == 0.95]
        assert len(new_jobs) == 2
        assert all(j["status"] == "pending" for j in new_jobs)

    def test_duplicate_multiplier_skipped(self, tmp_path):
        """Requesting an existing multiplier adds nothing."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        state = add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[1.0],  # already exists
            dry_run=True,
            verbose=False,
        )

        # No new configs added
        assert len(state.e2e_configs) == 2
        assert len(state.e2e_jobs) == 2

    def test_mixed_new_and_existing_multipliers(self, tmp_path):
        """Only new multipliers are added; existing ones are skipped."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        state = add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[1.0, 0.95, 1.05],
            dry_run=True,
            verbose=False,
        )

        # 1.0 exists (2 jobs), 0.95 new (2 jobs), 1.05 new (2 jobs)
        assert len(state.e2e_configs) == 6
        assert len(state.e2e_jobs) == 6

    def test_backup_created_before_mutation(self, tmp_path):
        """A backup file is created before adding configs."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[0.95],
            dry_run=True,
            verbose=False,
        )

        backups = list(tmp_path.glob("sweep_state.json.bak.*"))
        assert len(backups) == 1

    def test_no_pareto_raises(self, tmp_path):
        """add_e2e_jobs raises if the sweep has no Pareto frontier."""
        from run_sweep import add_e2e_jobs

        state = _make_state(tmp_path)
        state.phase = "gen"
        state.save()

        with pytest.raises(RuntimeError, match="no Pareto frontier"):
            add_e2e_jobs(
                output_dir=str(tmp_path),
                multipliers=[0.95],
                dry_run=True,
                verbose=False,
            )

    def test_configs_written_to_disk(self, tmp_path):
        """New E2E config YAML files are generated on disk."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        state = add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[0.95],
            dry_run=True,
            verbose=False,
        )

        new_configs = [c for c in state.e2e_configs if c["multiplier"] == 0.95]
        for c in new_configs:
            assert Path(c["config_path"]).exists(), f"Missing: {c['config_path']}"

    def test_idempotent_double_add(self, tmp_path):
        """Calling add_e2e_jobs twice with the same multiplier is idempotent."""
        from run_sweep import add_e2e_jobs

        _make_completed_sweep_state(tmp_path)

        add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[0.95],
            dry_run=True,
            verbose=False,
        )
        state = add_e2e_jobs(
            output_dir=str(tmp_path),
            multipliers=[0.95],
            dry_run=True,
            verbose=False,
        )

        # Should still have 4 (not 6)
        assert len(state.e2e_configs) == 4
        assert len(state.e2e_jobs) == 4


# =====================================================================
# Engine parser registration and stub tests
# =====================================================================

class TestParserRegistry:
    """Verify all engine parsers are registered and have correct interfaces."""

    def test_trtllm_ctx_parser_registered(self):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("trtllm")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")

    def test_trtllm_gen_parser_registered(self):
        from parser_base import get_gen_parser
        parser = get_gen_parser("trtllm")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")
        assert hasattr(parser, "get_mtp_accept_rate")
        assert hasattr(parser, "process_all_concurrencies")

    def test_vllm_ctx_parser_registered(self):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("vllm")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")

    def test_vllm_gen_parser_registered(self):
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")
        assert hasattr(parser, "get_mtp_accept_rate")

    def test_sglang_ctx_parser_registered(self):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("sglang")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")

    def test_sglang_gen_parser_registered(self):
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        assert parser is not None
        assert hasattr(parser, "find_log")
        assert hasattr(parser, "parse")
        assert hasattr(parser, "process")
        assert hasattr(parser, "get_mtp_accept_rate")

    def test_unknown_engine_raises(self):
        from parser_base import get_ctx_parser, get_gen_parser
        with pytest.raises(KeyError, match="No CTX parser registered"):
            get_ctx_parser("nonexistent_engine")
        with pytest.raises(KeyError, match="No GEN parser registered"):
            get_gen_parser("nonexistent_engine")

    def test_all_engines_listed(self):
        """All three engines should be in the registry."""
        from parser_base import _CTX_PARSERS, _GEN_PARSERS
        for engine in ("trtllm", "vllm", "sglang"):
            assert engine in _CTX_PARSERS, f"CTX parser missing for {engine}"
            assert engine in _GEN_PARSERS, f"GEN parser missing for {engine}"


class TestVllmParserStubs:
    """Verify vLLM parser stubs raise NotImplementedError with helpful messages."""

    def test_ctx_parse_raises(self, tmp_path):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("vllm")
        with pytest.raises(NotImplementedError, match="vLLM CTX log parsing"):
            parser.parse(tmp_path / "dummy.log")

    def test_ctx_process_raises(self):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("vllm")
        with pytest.raises(NotImplementedError, match="vLLM CTX result processing"):
            parser.process([], isl=1024)

    def test_gen_parse_raises(self, tmp_path):
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        with pytest.raises(NotImplementedError, match="vLLM GEN log parsing"):
            parser.parse(tmp_path / "dummy.log")

    def test_gen_process_raises(self):
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        with pytest.raises(NotImplementedError, match="vLLM GEN result processing"):
            parser.process([], concurrency=32, mode="tep")

    def test_ctx_find_log_returns_none_on_empty_dir(self, tmp_path):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("vllm")
        assert parser.find_log(tmp_path) is None

    def test_gen_find_log_returns_none_on_empty_dir(self, tmp_path):
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        assert parser.find_log(tmp_path) is None

    def test_gen_mtp_accept_rate_stp(self):
        """STP (mtp_num=0) always returns 1.0."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        assert parser.get_mtp_accept_rate(1024, 0) == 1.0

    def test_gen_mtp_accept_rate_with_override(self):
        """MTP accept rate uses overrides from sweep YAML."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        rate = parser.get_mtp_accept_rate(1024, 3, overrides={3: 2.56})
        assert rate == 2.56

    def test_gen_mtp_accept_rate_no_data_raises(self):
        """MTP without overrides or built-in data raises ValueError."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("vllm")
        with pytest.raises(ValueError, match="No vLLM MTP accept rate"):
            parser.get_mtp_accept_rate(1024, 3)


class TestSglangParserStubs:
    """Verify SGLang parser stubs raise NotImplementedError with helpful messages."""

    def test_ctx_parse_raises(self, tmp_path):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("sglang")
        with pytest.raises(NotImplementedError, match="SGLang CTX log parsing"):
            parser.parse(tmp_path / "dummy.log")

    def test_ctx_process_raises(self):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("sglang")
        with pytest.raises(NotImplementedError, match="SGLang CTX result processing"):
            parser.process([], isl=1024)

    def test_gen_parse_raises(self, tmp_path):
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        with pytest.raises(NotImplementedError, match="SGLang GEN log parsing"):
            parser.parse(tmp_path / "dummy.log")

    def test_gen_process_raises(self):
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        with pytest.raises(NotImplementedError, match="SGLang GEN result processing"):
            parser.process([], concurrency=32, mode="tep")

    def test_ctx_find_log_returns_none_on_empty_dir(self, tmp_path):
        from parser_base import get_ctx_parser
        parser = get_ctx_parser("sglang")
        assert parser.find_log(tmp_path) is None

    def test_gen_find_log_returns_none_on_empty_dir(self, tmp_path):
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        assert parser.find_log(tmp_path) is None

    def test_gen_mtp_accept_rate_stp(self):
        """STP (mtp_num=0) always returns 1.0."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        assert parser.get_mtp_accept_rate(1024, 0) == 1.0

    def test_gen_mtp_accept_rate_with_override(self):
        """MTP accept rate uses overrides from sweep YAML."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        rate = parser.get_mtp_accept_rate(1024, 3, overrides={3: 2.56})
        assert rate == 2.56

    def test_gen_mtp_accept_rate_no_data_raises(self):
        """MTP without overrides or built-in data raises ValueError."""
        from parser_base import get_gen_parser
        parser = get_gen_parser("sglang")
        with pytest.raises(ValueError, match="No SGLang MTP accept rate"):
            parser.get_mtp_accept_rate(1024, 3)
