# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom benchmark runner for user-specified commands.

Allows users to specify their own benchmark command with placeholders
that get expanded at runtime.

Example config:
    benchmark:
      type: "custom"
      container_image: "/containers/aiperf.sqsh"
      command: |
        uvx aiperf profile \
          --url {nginx_url} \
          --concurrency 128 \
          --artifact-dir {log_dir}/results
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("custom")
class CustomBenchmarkRunner(BenchmarkRunner):
    """Custom benchmark runner for user-specified commands.

    Required config fields:
        - benchmark.command: Command template with placeholders

    Optional:
        - benchmark.container_image: Custom container to run in

    Available placeholders in command:
        - {nginx_url}: URL of the nginx load balancer (e.g., http://localhost:8000)
        - {log_dir}: Log directory path
        - {job_id}: SLURM job ID
        - {run_name}: Job name + job ID
        - {model_path}: Model path (mounted at /model for local models)
        - {head_node_ip}: IP address of head node
        - {gpus_per_node}: GPUs per node
    """

    @property
    def name(self) -> str:
        return "Custom Benchmark"

    @property
    def script_path(self) -> str:
        """Custom benchmarks don't use a script path - command is fully specified."""
        return ""

    def validate_config(self, config: SrtConfig) -> list[str]:
        """Validate that command is specified for custom benchmarks."""
        errors = []
        if config.benchmark.command is None:
            errors.append("benchmark.command is required for type: custom")
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        """Build command by expanding placeholders in the command template.

        Args:
            config: Full job configuration
            runtime: Runtime context with resolved paths

        Returns:
            Command as list of strings
        """
        if config.benchmark.command is None:
            raise ValueError("benchmark.command is required for type: custom")

        # Build nginx_url for the placeholder
        nginx_url = f"http://localhost:{runtime.frontend_port}"

        # Expand placeholders using FormattableString.get_string()
        # Pass nginx_url as extra kwarg
        cmd_str = config.benchmark.command.get_string(runtime, nginx_url=nginx_url)

        # Strip and handle multiline commands
        cmd_str = cmd_str.strip()

        # If command contains shell operators (&&, ||, |, ;, etc.), use bash -c
        # This ensures proper shell interpretation of these operators
        shell_operators = ["&&", "||", "|", ";", ">", "<", "$(", "`"]
        if any(op in cmd_str for op in shell_operators):
            return ["bash", "-c", cmd_str]

        # Use shlex to properly parse the command into a list
        # This handles quoted strings, escapes, etc.
        try:
            return shlex.split(cmd_str)
        except ValueError:
            # If shlex fails (e.g., unclosed quotes), fall back to bash -c
            return ["bash", "-c", cmd_str]
