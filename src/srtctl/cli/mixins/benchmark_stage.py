# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark stage mixin for SweepOrchestrator.

Handles benchmark execution and profiling.
"""

import logging
import shlex
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.health import wait_for_model
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter

if TYPE_CHECKING:
    from srtctl.benchmarks.base import BenchmarkRunner
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint

logger = logging.getLogger(__name__)


class BenchmarkStageMixin:
    """Mixin for benchmark execution stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.endpoints: list[Endpoint]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        ...

    def run_benchmark(
        self, registry: "ProcessRegistry", stop_event: threading.Event, reporter: StatusReporter | None = None
    ) -> int:
        """Run the benchmark."""
        logger.info("Waiting for workers to be ready...")

        r = self.config.resources
        num_workers = r.num_prefill + r.num_decode + r.num_agg

        # Build descriptive worker count string
        worker_desc = f"{r.num_agg} agg" if r.num_agg > 0 else f"{r.num_prefill}P + {r.num_decode}D"

        logger.info("Waiting for server health (expecting %d workers: %s)...", num_workers, worker_desc)

        # For aggregated mode: expect 0 prefill, N decode (backend workers count as decode)
        # For disaggregated mode: expect N prefill, M decode
        if r.num_agg > 0:
            n_prefill = 0
            n_decode = r.num_agg
        else:
            n_prefill = r.num_prefill
            n_decode = r.num_decode

        # Determine if we're in direct worker mode (bypassing the router)
        is_direct = (
            r.num_agg == 1
            and self.config.frontend.type == "sglang"
        )
        health_port = self.runtime.frontend_port
        health_frontend_type = "direct" if is_direct else self.config.frontend.type

        hc = self.config.health_check
        if not wait_for_model(
            host=self.runtime.nodes.head,
            port=health_port,
            n_prefill=n_prefill,
            n_decode=n_decode,
            poll_interval=float(hc.interval_seconds),
            timeout=float(hc.max_attempts * hc.interval_seconds),
            report_every=60.0,
            frontend_type=health_frontend_type,
            stop_event=stop_event,
        ):
            logger.error("Server did not become healthy")
            if reporter:
                reporter.report(JobStatus.FAILED, JobStage.BENCHMARK, "Workers failed health check")
            return 1

        logger.info("Server is healthy - starting benchmark")
        if reporter:
            reporter.report(JobStatus.BENCHMARK, JobStage.BENCHMARK, "Running benchmark")

        # When profiling is enabled, use the dedicated profiling runner (profile.sh)
        # which sends controlled traffic and triggers cudaProfilerStart/Stop via
        # the SGLang /start_profile API.
        benchmark_type = self.config.benchmark.type
        if self.config.profiling.enabled:
            benchmark_type = "profiling"
            logger.info(
                "Profiling enabled (type=%s) - using dedicated profiling benchmark",
                self.config.profiling.type,
            )
            logger.info(
                "Profiling config: isl=%s, osl=%s, concurrency=%s",
                self.config.profiling.isl,
                self.config.profiling.osl,
                self.config.profiling.concurrency,
            )

        if benchmark_type == "manual":
            logger.info("Benchmark type is 'manual' - server is ready for testing")
            logger.info("Frontend URL: http://%s:%d", self.runtime.nodes.head, self.runtime.frontend_port)
            logger.info("Press Ctrl+C to stop the job")

            while not stop_event.is_set():
                if registry.check_failures():
                    logger.error("Worker failure detected during manual mode")
                    return 1
                time.sleep(5)
            return 0

        # Get the appropriate benchmark runner
        from srtctl.benchmarks import get_runner

        try:
            runner = get_runner(benchmark_type)
        except ValueError as e:
            logger.error("%s", e)
            return 1

        # Validate config
        errors = runner.validate_config(self.config)
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            return 1

        logger.info("Running %s benchmark", runner.name)

        # Run the benchmark script
        benchmark_log = self.runtime.log_dir / "benchmark.out"
        exit_code = self._run_benchmark_script(runner, benchmark_log, stop_event)

        if exit_code != 0:
            logger.error("Benchmark failed with exit code %d", exit_code)
        else:
            logger.info("Benchmark completed successfully")

        return exit_code

    def _run_benchmark_script(
        self,
        runner: "BenchmarkRunner",
        log_file: Path,
        stop_event: threading.Event,
    ) -> int:
        """Run the actual benchmark script."""

        cmd = runner.build_command(self.config, self.runtime)
        env_to_set = self._get_benchmark_profiling_env(runner)

        logger.info("Script: %s", runner.script_path)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", log_file)

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(log_file),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        # Wait for benchmark to complete
        while proc.poll() is None:
            if stop_event.is_set():
                logger.info("Stop requested, terminating benchmark")
                proc.terminate()
                return 1
            time.sleep(1)

        return proc.returncode or 0

    def _get_benchmark_profiling_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        env: dict[str, str] = {}

        # Add profiling-specific env vars
        if runner.name == "Profiling" and self.config.profiling.enabled:
            p = self.config.profiling

            # Traffic generator params
            if p.isl is not None:
                env["PROFILE_ISL"] = str(p.isl)
            if p.osl is not None:
                env["PROFILE_OSL"] = str(p.osl)
            if p.concurrency is not None:
                env["PROFILE_CONCURRENCY"] = str(p.concurrency)

            # Model name
            env["PROFILE_MODEL_NAME"] = self.config.served_model_name

            # Head node for traffic
            env["HEAD_NODE"] = self.runtime.nodes.head
            env["HEAD_PORT"] = str(self.runtime.frontend_port)

            # Collect worker leader IPs by mode
            prefill_ips = []
            decode_ips = []
            agg_ips = []

            for endpoint in self.endpoints:
                leader_ip = get_hostname_ip(endpoint.leader_node)
                if endpoint.mode == "prefill":
                    prefill_ips.append(leader_ip)
                elif endpoint.mode == "decode":
                    decode_ips.append(leader_ip)
                elif endpoint.mode == "agg":
                    agg_ips.append(leader_ip)

            if prefill_ips:
                env["PROFILE_PREFILL_IPS"] = ",".join(prefill_ips)
            if decode_ips:
                env["PROFILE_DECODE_IPS"] = ",".join(decode_ips)
            if agg_ips:
                env["PROFILE_AGG_IPS"] = ",".join(agg_ips)

            # Phase-specific step configs
            if p.prefill:
                if p.prefill.start_step is not None:
                    env["PROFILE_PREFILL_START_STEP"] = str(p.prefill.start_step)
                if p.prefill.stop_step is not None:
                    env["PROFILE_PREFILL_STOP_STEP"] = str(p.prefill.stop_step)
            if p.decode:
                if p.decode.start_step is not None:
                    env["PROFILE_DECODE_START_STEP"] = str(p.decode.start_step)
                if p.decode.stop_step is not None:
                    env["PROFILE_DECODE_STOP_STEP"] = str(p.decode.stop_step)
            if p.aggregated:
                if p.aggregated.start_step is not None:
                    env["PROFILE_AGG_START_STEP"] = str(p.aggregated.start_step)
                if p.aggregated.stop_step is not None:
                    env["PROFILE_AGG_STOP_STEP"] = str(p.aggregated.stop_step)

            # Torch profiler directory
            if p.is_torch:
                env["SGLANG_TORCH_PROFILER_DIR"] = str(self.runtime.log_dir / "profiles")

            # The profile.sh script only generates traffic when PROFILING_MODE=prefill
            env["PROFILING_MODE"] = "prefill"

            # Profiler type so profile.sh knows whether to call /start_profile API
            # (nsys uses time-based capture, torch uses API-based)
            env["PROFILER_TYPE"] = p.type

            # Backend type for benchmark_serving.py
            # With Dynamo frontend, always use "dynamo" backend
            # (trtllm/sglang are internal backend types, not benchmark_serving.py backends)
            if self.config.frontend.type == "dynamo":
                env["BACKEND_TYPE"] = "dynamo"
            elif self.config.backend_type == "sglang":
                env["BACKEND_TYPE"] = "sglang"
            else:
                # tensorrt-llm is the benchmark_serving.py name for TRT-LLM
                env["BACKEND_TYPE"] = "tensorrt-llm"

            # Internal backend type for profile.sh to distinguish sglang vs trtllm
            # (needed for nsys profiling - sglang uses /start_profile API, trtllm uses env var)
            env["SRTCTL_BACKEND_TYPE"] = self.config.backend_type

            # Configurable num_prompts
            env["PROFILE_NUM_PROMPTS"] = str(p.num_prompts)

        return env
