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


# TODO: this can be removed once we sync GB200 to > 0.5.5.post2
def _patch_sglang_engine():
    """Temporary patch to fix send_to_rpc initialization."""
    logging.info("Applying temporary patch to engine.py")
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
    Run a setup script in the /configs directory if it exists.
    
    Args:
        setup_script: Custom setup script name (e.g., 'custom-setup.sh'). 
                     If None, defaults to 'setup-script.sh'.
    """
    script_name = setup_script if setup_script else "setup-script.sh"
    script_path = f"/configs/{script_name}"
    
    if os.path.exists(script_path):
        logging.info(f"Running setup script: {script_path}")
        run_command(f"bash {script_path}")
    elif setup_script:
        # Only warn if a custom script was explicitly provided
        logging.warning(f"Setup script not found: {script_path}")

def setup_prefill_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
) -> int:
    """Setup the prefill worker."""
    # Setup infrastructure first (if traditional mode)
    need_frontend = not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0

    if need_frontend:
        setup_head_prefill_node(master_ip)
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")
    else:
        logging.info(f"Setting up prefill worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Start frontend AFTER installing wheels (traditional mode only)
    if need_frontend:
        logging.info("Starting frontend in traditional mode (after wheel installation)")

        # Open log files for frontend
        frontend_stdout = open("/logs/frontend.out", "w")
        frontend_stderr = open("/logs/frontend.err", "w")

        frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
        frontend_process = run_command(frontend_cmd, background=True, stdout=frontend_stdout, stderr=frontend_stderr)
        if not frontend_process:
            raise RuntimeError("Failed to start frontend")
        logging.info(f"Frontend started in background (PID: {frontend_process.pid})")
        logging.info("Frontend logs: /logs/frontend.out and /logs/frontend.err")

    # Apply temporary patch (only for gb200, not gb300)
    if gpu_type.startswith("gb200") and not gpu_type.startswith("gb300"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="prefill",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
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
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
) -> int:
    """Setup the decode worker."""
    logging.info(f"Setting up decode worker {worker_idx}, local rank {local_rank}")

    if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
        raise RuntimeError("Failed to connect to etcd")

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Apply temporary patch (only for gb200, not gb300)
    if gpu_type.startswith("gb200") and not gpu_type.startswith("gb300"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="decode",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
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
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
    dump_config_path: str | None = None,
    setup_script: str | None = None,
) -> int:
    """Setup the aggregated worker."""
    # Setup infrastructure first (if traditional mode)
    need_frontend = not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0

    if need_frontend:
        setup_head_prefill_node(master_ip)
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")
    else:
        logging.info(f"Setting up aggregated worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Run custom setup script if provided
    _run_setup_script(setup_script)

    # Install dynamo wheels
    install_dynamo_wheels(gpu_type)

    # Start frontend AFTER installing wheels (traditional mode only)
    if need_frontend:
        logging.info("Starting frontend in traditional mode (after wheel installation)")

        # Open log files for frontend
        frontend_stdout = open("/logs/frontend.out", "w")
        frontend_stderr = open("/logs/frontend.err", "w")

        frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
        frontend_process = run_command(frontend_cmd, background=True, stdout=frontend_stdout, stderr=frontend_stderr)
        if not frontend_process:
            raise RuntimeError("Failed to start frontend")
        logging.info(f"Frontend started in background (PID: {frontend_process.pid})")
        logging.info("Frontend logs: /logs/frontend.out and /logs/frontend.err")

    # Apply temporary patch (only for gb200, not gb300)
    if gpu_type.startswith("gb200") and not gpu_type.startswith("gb300"):
        _patch_sglang_engine()

    # Build and execute SGLang command from YAML config
    cmd_to_run = get_gpu_command(
        worker_type="aggregated",
        sglang_config_path=sglang_config_path,
        host_ip=leader_ip,
        port=DIST_INIT_PORT,
        total_nodes=nodes_per_worker,
        rank=local_rank,
        use_profiling=sglang_torch_profiler,
        dump_config_path=dump_config_path,
    )
    return run_command(cmd_to_run)
