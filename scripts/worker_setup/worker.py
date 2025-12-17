# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker setup functions for prefill, decode, and aggregated workers."""

import logging
import subprocess
import os

from .command import get_gpu_command, install_dynamo_wheels
from .environment import DIST_INIT_PORT, ETCD_CLIENT_PORT
from .infrastructure import setup_head_prefill_node
from .utils import run_command, wait_for_etcd


def _get_sglang_version() -> str | None:
    """Get the installed sglang version."""
    try:
        result = subprocess.run(
            ["python", "-c", "import sglang; print(sglang.__version__)"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logging.warning(f"Failed to get sglang version: {e}")
    return None


# TODO: this can be removed once we sync GB200 to > 0.5.5.post2
def _patch_sglang_engine():
    """Temporary patch to fix send_to_rpc initialization. Only applies to 0.5.5.post2."""
    version = _get_sglang_version()
    if version != "0.5.5.post2":
        logging.info(f"Skipping patch - sglang version {version} != 0.5.5.post2")
        return

    logging.info("Applying temporary patch to engine.py (sglang 0.5.5.post2)")
    sed_cmd = (
        "sed -i '/self.send_to_rpc = get_zmq_socket(/,/^        )/c\\"
        "        if self.server_args.node_rank == 0:\\n"
        "            self.send_to_rpc = get_zmq_socket(\\n"
        "                context, zmq.DEALER, self.port_args.rpc_ipc_name, True\\n"
        "            )\\n"
        "        else:\\n"
        "            self.send_to_rpc = None' "
        "/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py"
    )
    result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning(f"Failed to apply patch: {result.stderr}")
    else:
        logging.info("Patch applied successfully")


def _run_setup_script(setup_script: str | None = None):
    """
    Run a setup script in the /configs directory if explicitly provided.

    Args:
        setup_script: Custom setup script name (e.g., 'custom-setup.sh').
                     If None, no setup script runs.
    """
    if not setup_script:
        return

    script_path = f"/configs/{setup_script}"

    if os.path.exists(script_path):
        logging.info(f"Running setup script: {script_path}")
        run_command(f"bash {script_path}")
    else:
        logging.warning(f"Setup script not found: {script_path}")


def setup_prefill_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    profiler: str = "none",
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
    use_sglang_router: bool = False,
) -> int:
    """Setup the prefill worker."""
    # Setup infrastructure first (if traditional mode)
    need_frontend = not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0

    if not use_sglang_router:
        if need_frontend:
            setup_head_prefill_node(master_ip)
            if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
                raise RuntimeError("Failed to connect to etcd")
        else:
            logging.info(f"Setting up prefill worker {worker_idx}, local rank {local_rank}")
            if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
                raise RuntimeError("Failed to connect to etcd")

    # Install dynamo from PyPI
    install_dynamo_wheels(gpu_type)

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Start frontend AFTER installing dynamo (traditional mode only)
    if need_frontend:
        logging.info("Starting frontend in traditional mode (after dynamo installation)")

        # Open log files for frontend
        frontend_stdout = open("/logs/frontend.out", "w")
        frontend_stderr = open("/logs/frontend.err", "w")

        frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
        frontend_process = run_command(frontend_cmd, background=True, stdout=frontend_stdout, stderr=frontend_stderr)
        if not frontend_process:
            raise RuntimeError("Failed to start frontend")
        logging.info(f"Frontend started in background (PID: {frontend_process.pid})")
        logging.info("Frontend logs: /logs/frontend.out and /logs/frontend.err")

    # Apply temporary patch (for gb200 (not gb300) and h100)
    if (gpu_type.startswith("gb200") and not gpu_type.startswith("gb300")) or gpu_type.startswith("h100"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="prefill",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        profiler=profiler,
        dump_config_path=dump_config_path,
    )
    return run_command(cmd_to_run)


def setup_decode_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    profiler: str = "none",
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
    use_sglang_router: bool = False,
) -> int:
    """Setup the decode worker."""
    logging.info(f"Setting up decode worker {worker_idx}, local rank {local_rank}")

    if not use_sglang_router:
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Install dynamo from PyPI
    install_dynamo_wheels(gpu_type)

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Apply temporary patch (for gb200 (not gb300) and h100)
    if (gpu_type.startswith("gb200") and not gpu_type.startswith("gb300")) or gpu_type.startswith("h100"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="decode",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        profiler=profiler,
        dump_config_path=dump_config_path,
    )
    return run_command(cmd_to_run)


def setup_aggregated_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    profiler: str = "none",
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
    use_sglang_router: bool = False,
) -> int:
    """Setup the aggregated worker."""
    # Setup infrastructure first (if traditional mode)
    need_frontend = not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0

    if not use_sglang_router:
        if need_frontend:
            setup_head_prefill_node(master_ip)
            if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
                raise RuntimeError("Failed to connect to etcd")
        else:
            logging.info(f"Setting up aggregated worker {worker_idx}, local rank {local_rank}")
            if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
                raise RuntimeError("Failed to connect to etcd")

    # Install dynamo from PyPI
    install_dynamo_wheels(gpu_type)

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Start frontend AFTER installing dynamo (traditional mode only)
    if need_frontend:
        logging.info("Starting frontend in traditional mode (after dynamo installation)")

        # Open log files for frontend
        frontend_stdout = open("/logs/frontend.out", "w")
        frontend_stderr = open("/logs/frontend.err", "w")

        frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
        frontend_process = run_command(frontend_cmd, background=True, stdout=frontend_stdout, stderr=frontend_stderr)
        if not frontend_process:
            raise RuntimeError("Failed to start frontend")
        logging.info(f"Frontend started in background (PID: {frontend_process.pid})")
        logging.info("Frontend logs: /logs/frontend.out and /logs/frontend.err")

    # Apply temporary patch (for gb200 (not gb300) and h100)
    if (gpu_type.startswith("gb200") and not gpu_type.startswith("gb300")) or gpu_type.startswith("h100"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="aggregated",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        profiler=profiler,
        dump_config_path=dump_config_path,
    )
    return run_command(cmd_to_run)
