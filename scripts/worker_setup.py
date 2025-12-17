#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker setup script for Slurm nodes.

This script will be running on the prefill and decode nodes, and will be called by the
benchmark_dynamo.sh script.

The script will:
- Setup the environment
- Generate the python3 command to run the prefill or decode worker
- Start dynamo (or sglang)
"""

import argparse
import logging
import socket

from worker_setup import (
    setup_aggregated_worker,
    setup_decode_worker,
    setup_env,
    setup_frontend_worker,
    setup_logging,
    setup_nginx_worker,
    setup_prefill_worker,
)


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker setup script for Dynamo distributed training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--leader_ip",
        type=str,
        required=False,
        help="IP address of the leader node for this worker group",
    )
    parser.add_argument(
        "--master_ip",
        type=str,
        required=True,
        help="IP address of the master node (first prefill node) for NATS/ETCD",
    )
    parser.add_argument(
        "--worker_idx",
        type=int,
        required=False,
        help="Index of the worker group (0-based)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=False,
        help="Local rank within the worker group (0 for leader)",
    )
    parser.add_argument(
        "--nodes_per_worker",
        type=int,
        required=False,
        help="Number of nodes per worker",
    )
    parser.add_argument(
        "--worker_type",
        choices=["decode", "prefill", "frontend", "nginx", "aggregated"],
        required=True,
        help="Type of worker to run",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="Number of GPUs per node (default: 8)",
    )

    parser.add_argument(
        "--gpu_type",
        type=str,
        default="gb200-fp8",
        help="Type of GPU to use (script will be validated at runtime)",
    )
    parser.add_argument(
        "--nginx_config",
        type=str,
        help="Path to nginx configuration file (required for nginx worker type)",
    )

    parser.add_argument(
        "--multiple-frontends-enabled",
        action="store_true",
        help="Whether multiple frontend architecture is enabled (affects infrastructure setup)",
    )

    parser.add_argument(
        "--use-sglang-router",
        action="store_true",
        help="Whether this job uses sglang router (PD disaggregation); skips NATS/ETCD/frontend bootstrap in workers.",
    )

    parser.add_argument(
        "--dump-config-path",
        type=str,
        default=None,
        help="Path to dump config file (e.g., /logs/node_config.json)",
    )

    parser.add_argument(
        "--profiler",
        type=str,
        choices=["none", "torch", "nsys"],
        default="none",
        help="Profiling method for workers",
    )

    parser.add_argument(
        "--sglang-config-path",
        type=str,
        default=None,
        help="Path to sglang_config.yaml for YAML-based configs (enables direct command execution)",
    )

    parser.add_argument(
        "--setup-script",
        type=str,
        default=None,
        help="Custom setup script name in /configs directory (e.g., 'custom-setup.sh')",
    )

    return parser.parse_args(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.worker_type in ["prefill", "decode"]:
        if args.worker_idx is None or args.worker_idx < 0:
            raise ValueError("Worker index must be provided and non-negative for prefill/decode")

    if args.worker_type in ["prefill", "decode"]:
        if args.local_rank is None or args.local_rank < 0:
            raise ValueError("Local rank must be non-negative")

        if args.nodes_per_worker is None or args.nodes_per_worker < 1:
            raise ValueError("Nodes per worker must be at least 1")

        if args.gpus_per_node < 1:
            raise ValueError("GPUs per node must be at least 1")

        if args.local_rank >= args.nodes_per_worker:
            raise ValueError(
                f"Local rank ({args.local_rank}) must be less than nodes per worker ({args.nodes_per_worker})"
            )

    # Validate nginx-specific arguments
    if args.worker_type == "nginx" and not args.nginx_config:
        raise ValueError("--nginx_config is required for nginx worker type")


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)
    _validate_args(args)

    logging.info(f"{args.worker_type.capitalize()} worker setup started")
    logging.info(f"Hostname: {socket.gethostname()}")
    logging.info(f"Worker type: {args.worker_type}")
    logging.info(f"Worker index: {args.worker_idx}")
    logging.info(f"Local rank: {args.local_rank}")
    logging.info(f"Leader IP: {args.leader_ip}")
    logging.info(f"Master IP: {args.master_ip}")
    logging.info(f"Nodes per worker: {args.nodes_per_worker}")

    setup_env(args.master_ip)

    if args.worker_type == "nginx":
        if not args.nginx_config:
            raise ValueError("--nginx_config is required for nginx worker type")
        setup_nginx_worker(args.nginx_config)
    elif args.worker_type == "frontend":
        setup_frontend_worker(args.worker_idx, args.master_ip, args.gpu_type)
    elif args.worker_type == "prefill":
        setup_prefill_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpu_type,
            args.multiple_frontends_enabled,
            args.profiler,
            args.sglang_config_path,
            args.dump_config_path,
            args.setup_script,
            args.use_sglang_router,
        )
    elif args.worker_type == "decode":
        setup_decode_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpu_type,
            args.profiler,
            args.sglang_config_path,
            args.dump_config_path,
            args.setup_script,
            args.use_sglang_router,
        )
    elif args.worker_type == "aggregated":
        setup_aggregated_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpu_type,
            args.multiple_frontends_enabled,
            args.profiler,
            args.sglang_config_path,
            args.dump_config_path,
            args.setup_script,
            args.use_sglang_router,
        )

    logging.info(f"{args.worker_type.capitalize()} worker setup complete")


if __name__ == "__main__":
    main()
