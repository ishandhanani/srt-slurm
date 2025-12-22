#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang backend for SLURM job generation."""

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


class SGLangBackend:
    """SGLang backend for distributed serving."""

    def __init__(self, config: dict, setup_script: str = None):
        self.config = config
        self.backend_config = config.get("backend", {})
        self.resources = config.get("resources", {})
        self.model = config.get("model", {})
        self.slurm = config.get("slurm", {})
        self.setup_script = setup_script

    def is_disaggregated(self) -> bool:
        return self.resources.get("prefill_nodes") is not None

    def get_environment_vars(self, mode: str) -> dict[str, str]:
        return self.backend_config.get(f"{mode}_environment", {})

    def _profiling_type(self) -> str:
        return (self.config.get("profiling") or {}).get("type") or "none"

    def _config_to_flags(self, config: dict) -> list[str]:
        lines = []
        for key, value in sorted(config.items()):
            flag = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    lines.append(f"    --{flag} \\")
            elif isinstance(value, list):
                lines.append(f"    --{flag} {' '.join(str(v) for v in value)} \\")
            else:
                lines.append(f"    --{flag} {value} \\")
        return lines

    def generate_config_file(self, params: dict = None) -> Path | None:
        """Generate SGLang YAML config file."""
        if "sglang_config" not in self.backend_config:
            return None

        sglang_cfg = self.backend_config["sglang_config"]
        if params:
            sglang_cfg = expand_template(sglang_cfg, params)
            logging.info(f"Expanded config with params: {params}")

        # Validate kebab-case keys
        for mode in ["prefill", "decode", "aggregated"]:
            if mode in sglang_cfg and sglang_cfg[mode]:
                for key in sglang_cfg[mode].keys():
                    if "_" in key:
                        raise ValueError(f"Invalid key '{key}': use '{key.replace('_', '-')}' (kebab-case)")

        result = {mode: sglang_cfg[mode] for mode in ["prefill", "decode", "aggregated"] if mode in sglang_cfg}
        for mode in ["prefill", "decode", "aggregated"]:
            if env := self.get_environment_vars(mode):
                result[f"{mode}_environment"] = env

        fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="sglang_config_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(result, f, default_flow_style=False)
        logging.info(f"Generated SGLang config: {temp_path}")
        return Path(temp_path)

    def render_command(self, mode: str, config_path: Path = None) -> str:
        """Render full SGLang command with all flags inlined."""
        lines = [f"{k}={v} \\" for k, v in (self.get_environment_vars(mode) or {}).items()]

        prof = self._profiling_type()
        use_sglang = prof != "none" or self.backend_config.get("use_sglang_router", False)
        if prof == "nsys":
            lines.append(
                "nsys profile -t cuda,nvtx --cuda-graph-trace=node -c cudaProfilerApi --capture-range-end stop --force-overwrite true python3 -m sglang.launch_server \\"
            )
        elif use_sglang:
            lines.append("python3 -m sglang.launch_server \\")
        else:
            lines.append("python3 -m dynamo.sglang \\")

        if config_path:
            with open(config_path) as f:
                sglang_config = yaml.safe_load(f)
            lines.extend(self._config_to_flags(sglang_config.get(mode, {})))

        nnodes = (
            (self.resources["prefill_nodes"] if mode == "prefill" else self.resources["decode_nodes"])
            if self.is_disaggregated()
            else self.resources["agg_nodes"]
        )
        lines.extend(
            [
                "    --dist-init-addr $HOST_IP_MACHINE:$PORT \\",
                f"    --nnodes {nnodes} \\",
                "    --node-rank $RANK \\",
            ]
        )
        return "\n".join(lines)

    def generate_slurm_script(self, config_path: Path = None, timestamp: str = None) -> tuple[Path, str]:
        """Generate SLURM job script from Jinja template."""
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        is_aggregated = not self.is_disaggregated()

        if is_aggregated:
            agg_nodes, agg_workers = self.resources["agg_nodes"], self.resources["agg_workers"]
            prefill_nodes = decode_nodes = prefill_workers = decode_workers = 0
            total_nodes = agg_nodes
        else:
            prefill_nodes, decode_nodes = self.resources["prefill_nodes"], self.resources["decode_nodes"]
            prefill_workers, decode_workers = self.resources["prefill_workers"], self.resources["decode_workers"]
            agg_nodes = agg_workers = 0
            total_nodes = prefill_nodes + decode_nodes

        # SLURM settings
        job_name = self.config.get("name", "srtctl-job")
        account = self.slurm.get("account") or get_srtslurm_setting("default_account")
        partition = self.slurm.get("partition") or get_srtslurm_setting("default_partition")
        time_limit = self.slurm.get("time_limit") or get_srtslurm_setting("default_time_limit", "04:00:00")
        gpus_per_node = get_srtslurm_setting("gpus_per_node", self.resources.get("gpus_per_node"))

        # Benchmark config
        benchmark_config = self.config.get("benchmark", {})
        bench_type = benchmark_config.get("type", "manual")
        parsable_config = ""
        if bench_type == "sa-bench":
            conc = benchmark_config.get("concurrencies")
            conc_str = "x".join(str(c) for c in conc) if isinstance(conc, list) else str(conc)
            parsable_config = f"{benchmark_config.get('isl')} {benchmark_config.get('osl')} {conc_str} {benchmark_config.get('req_rate', 'inf')}"
        elif bench_type == "mmlu":
            num_examples = benchmark_config.get("num_examples", 200)
            max_tokens = benchmark_config.get("max_tokens", 8192)
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

        # Paths
        srtctl_root = Path(get_srtslurm_setting("srtctl_root") or Path(srtctl.__file__).parent.parent.parent)

        # Profiling env
        profiling_cfg = self.config.get("profiling") or {}
        env_map = {
            "isl": "PROFILE_ISL",
            "osl": "PROFILE_OSL",
            "concurrency": "PROFILE_CONCURRENCY",
            "start_step": "PROFILE_START_STEP",
            "stop_step": "PROFILE_STOP_STEP",
        }
        profile_env = " ".join(
            f"{env}={profiling_cfg[k]}" for k, env in env_map.items() if profiling_cfg.get(k) is not None
        )
        profiler_mode = self._profiling_type()

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
            "config_dir": str(srtctl_root / "configs"),
            "container_image": self.model.get("container"),
            "gpus_per_node": gpus_per_node,
            "network_interface": get_srtslurm_setting("network_interface"),
            "gpu_type": self.backend_config.get("gpu_type", "h100"),
            "partition": partition,
            "enable_multiple_frontends": self.backend_config.get("enable_multiple_frontends", True),
            "num_additional_frontends": self.backend_config.get("num_additional_frontends", 9),
            "use_sglang_router": self.backend_config.get("use_sglang_router", False),
            "do_benchmark": bench_type != "manual",
            "benchmark_type": bench_type,
            "benchmark_arg": parsable_config,
            "timestamp": timestamp,
            "enable_config_dump": profiler_mode == "none" and self.config.get("enable_config_dump", True),
            "log_dir_prefix": str(srtctl_root / "logs"),
            "profiler": profiler_mode,
            "prefill_profile_env": profile_env,
            "decode_profile_env": profile_env,
            "setup_script": self.setup_script,
            "use_gpus_per_node_directive": get_srtslurm_setting("use_gpus_per_node_directive", True),
            "use_segment_sbatch_directive": get_srtslurm_setting("use_segment_sbatch_directive", True),
            "extra_container_mounts": ",".join(self.config.get("extra_mount") or []),
        }

        template_name = "job_script_template_agg.j2" if is_aggregated else "job_script_template_disagg.j2"
        template_path = srtctl_root / "scripts" / "templates" / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}\nSet 'srtctl_root' in srtslurm.yaml")

        with open(template_path) as f:
            rendered_script = Template(f.read()).render(**template_vars)

        fd, temp_path = tempfile.mkstemp(suffix=".sh", prefix="slurm_job_")
        with os.fdopen(fd, "w") as f:
            f.write(rendered_script)
        logging.info(f"Generated SLURM job script: {temp_path}")
        return Path(temp_path), rendered_script
