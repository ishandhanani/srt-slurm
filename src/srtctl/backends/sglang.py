#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend support.
"""

import logging
import os
import tempfile
import yaml
from datetime import datetime
from jinja2 import Template
from pathlib import Path

import srtctl
from srtctl.core.config import get_srtslurm_setting
from srtctl.core.sweep import expand_template

from .base import Backend


class SGLangBackend(Backend):
    """SGLang backend for distributed serving."""

    def __init__(self, config: dict, setup_script: str = None):
        """Initialize SGLang backend.

        Args:
            config: Full user configuration dict
            setup_script: Optional custom setup script name in configs directory
        """
        super().__init__(config)
        self.setup_script = setup_script

    def generate_config_file(self, params: dict = None) -> Path | None:
        """Generate SGLang YAML config file.

        Args:
            params: Optional sweep parameters for template expansion

        Returns:
            Path to generated config file
        """
        if "sglang_config" not in self.backend_config:
            return None

        sglang_cfg = self.backend_config["sglang_config"]

        # Expand templates if sweeping
        if params:
            sglang_cfg = expand_template(sglang_cfg, params)
            logging.info(f"Expanded config with params: {params}")

        # Validate that all keys use dashes, not underscores
        for mode in ["prefill", "decode", "aggregated"]:
            if mode in sglang_cfg and sglang_cfg[mode]:
                for key in sglang_cfg[mode].keys():
                    if "_" in key:
                        raise ValueError(
                            f"Invalid key '{key}' in sglang_config.{mode}: "
                            f"Keys must use dashes (kebab-case), not underscores. "
                            f"Use '{key.replace('_', '-')}' instead."
                        )

        # Extract prefill, decode, and aggregated configs (no conversion needed - already using dashes)
        result = {}
        for mode in ["prefill", "decode", "aggregated"]:
            if mode in sglang_cfg:
                result[mode] = sglang_cfg[mode]

        # Add environment variables as top-level keys
        for mode in ["prefill", "decode", "aggregated"]:
            env_vars = self.get_environment_vars(mode)
            if env_vars:
                result[f"{mode}_environment"] = env_vars

        # Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="sglang_config_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(result, f, default_flow_style=False)

        logging.info(f"Generated SGLang config: {temp_path}")
        return Path(temp_path)

    def render_command(self, mode: str, config_path: Path = None) -> str:
        """Render full SGLang command with all flags inlined.

        Args:
            mode: "prefill" or "decode"
            config_path: Path to generated SGLang config file

        Returns:
            Multi-line bash command string
        """
        lines = []

        # Environment variables
        env_vars = self.get_environment_vars(mode) or {}
        for key, val in env_vars.items():
            lines.append(f"{key}={val} \\")

        # Python command - use sglang.launch_server when profiler != none, dynamo.sglang otherwise
        profiling_type = (self.config.get("profiling") or {}).get("type") or "none"
        nsys_prefix = "nsys profile -t cuda,nvtx --cuda-graph-trace=node -c cudaProfilerApi --capture-range-end stop --force-overwrite true"
        if profiling_type == "nsys":
            lines.append(f"{nsys_prefix} python3 -m sglang.launch_server \\")
        elif profiling_type == "torch":
            lines.append("python3 -m sglang.launch_server \\")
        else:
            lines.append("python3 -m dynamo.sglang \\")

        # Inline all SGLang flags from config file
        if config_path:
            with open(config_path) as f:
                sglang_config = yaml.load(f, Loader=yaml.FullLoader)

            mode_config = sglang_config.get(mode, {})
            flag_lines = self._config_to_flags(mode_config)
            lines.extend(flag_lines)

        # Add coordination flags
        coord_flags = self._get_coordination_flags(mode)
        lines.extend(coord_flags)

        return "\n".join(lines)

    def _config_to_flags(self, config: dict) -> list[str]:
        """Convert config dict to CLI flags.

        Args:
            config: SGLang config dict for this mode

        Returns:
            List of flag strings with backslash continuations
        """
        lines = []
        profiling_type = (self.config.get("profiling") or {}).get("type") or "none"

        for key, value in sorted(config.items()):
            # Convert underscores to hyphens
            flag_name = key.replace("_", "-")

            # Always pass disaggregation-mode so profiling runs in PD mode

            if isinstance(value, bool):
                if value:
                    lines.append(f"    --{flag_name} \\")
            elif isinstance(value, list):
                values_str = " ".join(str(v) for v in value)
                lines.append(f"    --{flag_name} {values_str} \\")
            else:
                lines.append(f"    --{flag_name} {value} \\")

        return lines

    def _get_coordination_flags(self, mode: str) -> list[str]:
        """Get multi-node coordination flags.

        Args:
            mode: "prefill" or "decode"

        Returns:
            List of coordination flag strings
        """
        lines = []

        # Determine nnodes based on mode
        if self.is_disaggregated():
            nnodes = self.resources["prefill_nodes"] if mode == "prefill" else self.resources["decode_nodes"]
        else:
            nnodes = self.resources["agg_nodes"]

        # Coordination flags
        lines.append("    --dist-init-addr $HOST_IP_MACHINE:$PORT \\")
        lines.append(f"    --nnodes {nnodes} \\")
        lines.append("    --node-rank $RANK \\")

        return lines

    def _get_enable_config_dump(self) -> bool:
        """Get enable_config_dump value, handling profiling mode.

        Returns:
            True if config dump should be enabled, False otherwise
        """
        # Get value from config (defaults to True in schema)
        enable_config_dump = self.config.get("enable_config_dump", True)

        # Auto-disable when profiling is enabled (unless explicitly set to True)
        profiling_type = (self.config.get("profiling") or {}).get("type") or "none"
        if profiling_type != "none":
            # When profiling, disable config dump by default
            # User can explicitly set enable_config_dump: true to override
            return False

        return enable_config_dump

    def generate_slurm_script(self, config_path: Path = None, timestamp: str = None) -> tuple[Path, str]:
        """Generate SLURM job script from Jinja template.

        Args:
            config_path: Path to SGLang config file
            timestamp: Timestamp for job submission

        Returns:
            Tuple of (script_path, rendered_script_content)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine mode and node counts
        is_aggregated = not self.is_disaggregated()

        if is_aggregated:
            agg_nodes = self.resources["agg_nodes"]
            agg_workers = self.resources["agg_workers"]
            prefill_nodes = 0
            decode_nodes = 0
            prefill_workers = 0
            decode_workers = 0
            total_nodes = agg_nodes
        else:
            prefill_nodes = self.resources["prefill_nodes"]
            decode_nodes = self.resources["decode_nodes"]
            prefill_workers = self.resources["prefill_workers"]
            decode_workers = self.resources["decode_workers"]
            agg_nodes = 0
            agg_workers = 0
            total_nodes = prefill_nodes + decode_nodes

        # Get SLURM settings
        job_name = self.config.get("name", "srtctl-job")
        account = self.slurm.get("account") or get_srtslurm_setting("default_account")
        partition = self.slurm.get("partition") or get_srtslurm_setting("default_partition")
        time_limit = self.slurm.get("time_limit") or get_srtslurm_setting("default_time_limit", "04:00:00")

        # Get resource settings from srtslurm.yaml if available
        gpus_per_node = get_srtslurm_setting("gpus_per_node", self.resources.get("gpus_per_node"))
        network_interface = get_srtslurm_setting("network_interface", None)

        # Get backend settings
        gpu_type = self.backend_config.get("gpu_type", "h100")

        # Benchmark config
        benchmark_config = self.config.get("benchmark", {})
        bench_type = benchmark_config.get("type", "manual")
        do_benchmark = bench_type != "manual"

        # Parse benchmark args if applicable
        parsable_config = ""
        if bench_type == "sa-bench":
            isl = benchmark_config.get("isl")
            osl = benchmark_config.get("osl")
            concurrencies = benchmark_config.get("concurrencies")
            req_rate = benchmark_config.get("req_rate", "inf")

            if isinstance(concurrencies, list):
                concurrency_str = "x".join(str(c) for c in concurrencies)
            else:
                concurrency_str = str(concurrencies)

            parsable_config = f"{isl} {osl} {concurrency_str} {req_rate}"
        elif bench_type == "mmlu":
            num_examples = benchmark_config.get("num_examples", 200)
            max_tokens = benchmark_config.get("max_tokens", 2048)
            repeat = benchmark_config.get("repeat", 8)
            num_threads = benchmark_config.get("num_threads", 512)
            parsable_config = f"{num_examples} {max_tokens} {repeat} {num_threads}"
        elif bench_type == "gpqa":
            num_examples = benchmark_config.get("num_examples", 198)
            max_tokens = benchmark_config.get("max_tokens", 32768)
            repeat = benchmark_config.get("repeat", 8)
            num_threads = benchmark_config.get("num_threads", 128)
            parsable_config = f"{num_examples} {max_tokens} {repeat} {num_threads}"
        elif bench_type == "longbenchv2":
            num_examples = benchmark_config.get("num_examples", None)
            max_tokens = benchmark_config.get("max_tokens", 16384)
            max_context_length = benchmark_config.get("max_context_length", 128000)
            num_threads = benchmark_config.get("num_threads", 16)
            categories = benchmark_config.get("categories", None)
            parsable_config = f"{num_examples} {max_tokens} {max_context_length} {num_threads} {categories}"

        # Config directory should point to where deepep_config.json lives
        # This is typically the configs/ directory in the yaml-config repo
        yaml_config_root = Path(srtctl.__file__).parent.parent.parent

        # Log directory - check srtslurm.yaml first, then fall back to default
        srtctl_root_setting = get_srtslurm_setting("srtctl_root")
        if srtctl_root_setting:
            srtctl_root = Path(srtctl_root_setting)
        else:
            # Fall back to default: current yaml-config directory (which contains scripts/)
            srtctl_root = yaml_config_root

        # Use srtctl_root for config_dir_path so it respects srtslurm.yaml setting
        config_dir_path = srtctl_root / "configs"
        log_dir_path = srtctl_root / "logs"

        # Build profiling env injections 
        profiling_cfg = self.config.get("profiling") or {}

        def build_env_str(cfg: dict) -> str:
            parts: list[str] = []
            if "isl" in cfg and cfg["isl"] is not None:
                parts.append(f"PROFILE_ISL={cfg['isl']}")
            if "osl" in cfg and cfg["osl"] is not None:
                parts.append(f"PROFILE_OSL={cfg['osl']}")
            if "concurrency" in cfg and cfg["concurrency"] is not None:
                parts.append(f"PROFILE_CONCURRENCY={cfg['concurrency']}")
            if "start_step" in cfg and cfg["start_step"] is not None:
                parts.append(f"PROFILE_START_STEP={cfg['start_step']}")
            if "stop_step" in cfg and cfg["stop_step"] is not None:
                parts.append(f"PROFILE_STOP_STEP={cfg['stop_step']}")
            return " ".join(parts)

        # Use the same profiling spec for both prefill and decode; in PD
        # disaggregation mode this single spec drives both sides.
        prefill_profile_env = build_env_str(profiling_cfg)
        decode_profile_env = build_env_str(profiling_cfg)

        profiler_mode = profiling_cfg.get("type") or "none"
        # Template variables
        template_vars = {
            "job_name": job_name,
            "total_nodes": total_nodes,
            "account": account,
            "time_limit": time_limit,
            "prefill_nodes": prefill_nodes,
            "decode_nodes": decode_nodes,
            "prefill_workers": prefill_workers,
            "decode_workers": decode_workers,
            "agg_nodes": agg_nodes,
            "agg_workers": agg_workers,
            "is_aggregated": is_aggregated,
            "model_dir": self.model.get("path"),
            "config_dir": str(config_dir_path),
            "container_image": self.model.get("container"),
            "gpus_per_node": gpus_per_node,
            "network_interface": network_interface,
            "gpu_type": gpu_type,
            "partition": partition,
            "enable_multiple_frontends": self.backend_config.get("enable_multiple_frontends", True),
            "num_additional_frontends": self.backend_config.get("num_additional_frontends", 9),
            "use_sglang_router": self.backend_config.get("use_sglang_router", False),
            "do_benchmark": do_benchmark,
            "benchmark_type": bench_type,
            "benchmark_arg": parsable_config,
            "timestamp": timestamp,
            # Config dump enabled by default (True in schema)
            # Auto-disabled when profiling unless explicitly enabled
            "enable_config_dump": self._get_enable_config_dump(),
            "log_dir_prefix": str(log_dir_path),  # Absolute path to logs directory
            "profiler": profiler_mode,
            "prefill_profile_env": prefill_profile_env,
            "decode_profile_env": decode_profile_env,
            "setup_script": self.setup_script,
            "use_gpus_per_node_directive": get_srtslurm_setting("use_gpus_per_node_directive", True),
            "use_segment_sbatch_directive": get_srtslurm_setting("use_segment_sbatch_directive", True),
            "extra_container_mounts": ",".join(self.config.get("extra_mount") or []),
        }

        # Select template based on mode
        if is_aggregated:
            template_name = "job_script_template_agg.j2"
        else:
            template_name = "job_script_template_disagg.j2"

        # Find template path - check srtslurm.yaml first, then fall back to default location
        srtctl_root = get_srtslurm_setting("srtctl_root")

        if srtctl_root:
            # User specified srtctl_root in srtslurm.yaml
            template_path = Path(srtctl_root) / "scripts" / "templates" / template_name
        else:
            # Fall back to default: current yaml-config directory (which contains scripts/)
            yaml_config_root = Path(srtctl.__file__).parent.parent.parent
            template_path = yaml_config_root / "scripts" / "templates" / template_name

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Set 'srtctl_root' in srtslurm.yaml to point to your srtctl repo.\n"
                f"Example: srtctl_root: /mnt/lustre01/users/slurm-shared/ishan/sweepr"
            )

        # Render template
        with open(template_path) as f:
            template = Template(f.read())

        rendered_script = template.render(**template_vars)

        # Write to temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".sh", prefix="slurm_job_")
        with os.fdopen(fd, "w") as f:
            f.write(rendered_script)

        logging.info(f"Generated SLURM job script: {temp_path}")
        return Path(temp_path), rendered_script
