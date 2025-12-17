# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command building functions for SGLang workers."""

import logging
import os
import subprocess


def build_sglang_command_from_yaml(
    worker_type: str,
    sglang_config_path: str,
    host_ip: str,
    port: int,
    total_nodes: int,
    rank: int,
    profiler: str = "none",
    dump_config_path: str | None = None,
) -> str:
    """Build SGLang command using native YAML config support.

    dynamo.sglang supports reading config from YAML:
        python3 -m dynamo.sglang --config file.yaml --config-key prefill

    sglang.launch_server (profiling mode) requires explicit flags:
        python3 -m sglang.launch_server --model-path /model/ --tp 4 ...

    Args:
        worker_type: "prefill", "decode", or "aggregated"
        sglang_config_path: Path to generated sglang_config.yaml
        host_ip: Host IP for distributed coordination
        port: Port for distributed coordination
        total_nodes: Total number of nodes
        rank: Node rank (0-indexed)
        profiler: Profiling method: "none", "torch", or "nsys"

    Returns:
        Full command string ready to execute
    """
    import yaml

    # Load config to extract environment variables and mode config
    with open(sglang_config_path) as f:
        sglang_config = yaml.safe_load(f)

    config_key = worker_type if worker_type != "aggregated" else "aggregated"

    # Environment variables are stored at top level as {mode}_environment
    env_key = f"{config_key}_environment"
    env_vars = sglang_config.get(env_key, {})

    # Build environment variable exports
    env_exports = []
    for key, value in env_vars.items():
        env_exports.append(f"export {key}={value}")
    if profiler == "torch":
        env_exports.append(f"export SGLANG_TORCH_PROFILER_DIR=/logs/profiles/{config_key}")

    # Determine Python module based on profiling mode
    python_module = "sglang.launch_server" if profiler != "none" else "dynamo.sglang"
    nsys_prefix = f"nsys profile -t cuda,nvtx --cuda-graph-trace=node -c cudaProfilerApi --capture-range-end stop --force-overwrite true -o /logs/profiles/{config_key}_{rank}"

    if profiler != "none":
        # Profiling mode: inline all flags (sglang.launch_server doesn't support --config)
        mode_config = sglang_config.get(config_key, {})
        # Wrap with NSYS on all ranks; outputs are isolated per-rank
        if profiler == "nsys":
            cmd_parts = [f"{nsys_prefix} python3 -m {python_module}"]
        else:
            cmd_parts = [f"python3 -m {python_module}"]

        # Add all SGLang flags from config
        for key, value in sorted(mode_config.items()):
            flag_name = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{flag_name}")
            elif isinstance(value, list):
                values_str = " ".join(str(v) for v in value)
                cmd_parts.append(f"--{flag_name} {values_str}")
            else:
                cmd_parts.append(f"--{flag_name} {value}")

        # Add coordination flags
        cmd_parts.extend(
            [
                f"--dist-init-addr {host_ip}:{port}",
                f"--nnodes {total_nodes}",
                f"--node-rank {rank}",
                "--host 0.0.0.0",
            ]
        )
    else:
        # Normal mode: use --config and --config-key (dynamo.sglang supports this)
        cmd_parts = [
            f"python3 -m {python_module}",
            f"--config {sglang_config_path}",
            f"--config-key {config_key}",
            f"--dist-init-addr {host_ip}:{port}",
            f"--nnodes {total_nodes}",
            f"--node-rank {rank}",
            "--host 0.0.0.0",
        ]

    # Add dump-config-to flag if provided
    if dump_config_path:
        cmd_parts.append(f"--dump-config-to {dump_config_path}")

    # Combine environment exports and command
    full_command = " && ".join(env_exports + [" ".join(cmd_parts)]) if env_exports else " ".join(cmd_parts)

    return full_command


def install_dynamo_wheels(gpu_type: str) -> None:
    """Install dynamo from PyPI.

    Args:
        gpu_type: GPU type (unused - pip auto-selects correct architecture)
    """
    logging.info("Installing dynamo 0.7.0 from PyPI")

    # Install ai-dynamo-runtime (pip auto-selects x86_64 or aarch64 wheel)
    runtime_package = "ai-dynamo-runtime==0.7.0"
    logging.info(f"Installing {runtime_package}")
    result = subprocess.run(["python3", "-m", "pip", "install", runtime_package], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Failed to install runtime package: {result.stderr}")
        raise RuntimeError(f"Failed to install {runtime_package}")

    # Install ai-dynamo
    dynamo_package = "ai-dynamo==0.7.0"
    logging.info(f"Installing {dynamo_package}")
    result = subprocess.run(["python3", "-m", "pip", "install", dynamo_package], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Failed to install dynamo package: {result.stderr}")
        raise RuntimeError(f"Failed to install {dynamo_package}")

    logging.info("Successfully installed dynamo from PyPI")


def get_gpu_command(
    worker_type: str,
    sglang_config_path: str,
    host_ip: str,
    port: int,
    total_nodes: int,
    rank: int,
    profiler: str = "none",
    dump_config_path: str | None = None,
) -> str:
    """Generate command to run SGLang worker using YAML config.

    Args:
        worker_type: "prefill", "decode", or "aggregated"
        sglang_config_path: Path to sglang_config.yaml
        host_ip: Host IP for distributed coordination
        port: Port for distributed coordination
        total_nodes: Total number of nodes
        rank: Node rank (0-indexed)
        profiler: Profiling method: "none", "torch", or "nsys"

    Returns:
        Command string to execute
    """
    if not sglang_config_path or not os.path.exists(sglang_config_path):
        raise ValueError(f"SGLang config path required but not found: {sglang_config_path}")

    logging.info(f"Building command from YAML config: {sglang_config_path}")
    return build_sglang_command_from_yaml(
        worker_type, sglang_config_path, host_ip, port, total_nodes, rank, profiler, dump_config_path
    )
