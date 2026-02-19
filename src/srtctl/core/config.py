#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config loading and resolution with srtslurm.yaml integration.

This module provides:
- load_config(): Load YAML config, apply cluster defaults, return typed SrtConfig
- get_srtslurm_setting(): Get cluster-wide settings
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .schema import ClusterConfig, SrtConfig

logger = logging.getLogger(__name__)


def load_cluster_config() -> dict[str, Any] | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Searches for srtslurm.yaml in order:
    1. SRTSLURM_CONFIG environment variable (if set)
    2. Current working directory
    3. Parent directories up to 3 levels

    Returns None if file doesn't exist (graceful degradation).
    """
    # Check env var first (highest priority)
    env_config = os.environ.get("SRTSLURM_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            cluster_config_path = env_path
            logger.debug(f"Using srtslurm.yaml from SRTSLURM_CONFIG: {cluster_config_path}")
        else:
            logger.warning(f"SRTSLURM_CONFIG set but file not found: {env_config}")
            return None
    else:
        # Search paths
        search_paths = [
            Path.cwd() / "srtslurm.yaml",
            Path.cwd().parent / "srtslurm.yaml",
            Path.cwd().parent.parent / "srtslurm.yaml",
        ]

        cluster_config_path = None
        for path in search_paths:
            if path.exists():
                cluster_config_path = path
                break

        if not cluster_config_path:
            logger.debug("No srtslurm.yaml found - using config as-is")
            return None

    try:
        with open(cluster_config_path) as f:
            raw_config = yaml.safe_load(f)

        # Validate with marshmallow schema
        schema = ClusterConfig.Schema()
        validated = schema.load(raw_config)
        logger.debug(f"Loaded cluster config from {cluster_config_path}")

        # Dump back to dict for compatibility
        return schema.dump(validated)
    except Exception as e:
        logger.warning(f"Failed to load or validate srtslurm.yaml: {e}")
        return None


def resolve_config_with_defaults(user_config: dict[str, Any], cluster_config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Resolve user config by applying cluster defaults and aliases.

    This applies:
    1. Default SLURM settings (account, partition, time_limit)
    2. Model path alias resolution
    3. Container alias resolution

    Args:
        user_config: User's YAML config as dict
        cluster_config: Cluster defaults from srtslurm.yaml (or None)

    Returns:
        Resolved config dict with all defaults applied
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    if cluster_config is None:
        return config

    # Apply SLURM defaults
    slurm = config.setdefault("slurm", {})
    if "account" not in slurm and cluster_config.get("default_account"):
        slurm["account"] = cluster_config["default_account"]
        logger.debug(f"Applied default account: {slurm['account']}")

    if "partition" not in slurm and cluster_config.get("default_partition"):
        slurm["partition"] = cluster_config["default_partition"]
        logger.debug(f"Applied default partition: {slurm['partition']}")

    if "time_limit" not in slurm and cluster_config.get("default_time_limit"):
        slurm["time_limit"] = cluster_config["default_time_limit"]
        logger.debug(f"Applied default time_limit: {slurm['time_limit']}")

    # Resolve model path alias
    model = config.get("model", {})
    model_path = model.get("path", "")

    model_paths = cluster_config.get("model_paths")
    if model_paths and model_path in model_paths:
        resolved_path = model_paths[model_path]
        model["path"] = resolved_path
        logger.debug(f"Resolved model alias '{model_path}' -> '{resolved_path}'")

    # Resolve container alias
    container = model.get("container", "")

    containers = cluster_config.get("containers")
    if containers and container in containers:
        resolved_container = containers[container]
        model["container"] = resolved_container
        logger.debug(f"Resolved container alias '{container}' -> '{resolved_container}'")

    # Apply reporting defaults (if not specified in user config)
    if "reporting" not in config and cluster_config.get("reporting"):
        config["reporting"] = cluster_config["reporting"]
        logger.debug("Applied cluster reporting config")

    # Resolve frontend nginx_container alias
    frontend = config.get("frontend", {})
    nginx_container = frontend.get("nginx_container", "")

    if containers and nginx_container in containers:
        resolved_nginx = containers[nginx_container]
        frontend["nginx_container"] = resolved_nginx
        config["frontend"] = frontend
        logger.debug(f"Resolved nginx_container alias '{nginx_container}' -> '{resolved_nginx}'")

    return config


def get_srtslurm_setting(key: str, default: Any = None) -> Any:
    """
    Get a setting from srtslurm.yaml cluster config.

    Args:
        key: Setting key (e.g., 'gpus_per_node', 'network_interface')
        default: Default value if not found

    Returns:
        Setting value or default if not found
    """
    cluster_config = load_cluster_config()
    if cluster_config and key in cluster_config:
        return cluster_config[key]
    return default


def load_config(path: Path | str, config_overrides: dict | None = None) -> SrtConfig:
    """
    Load and validate YAML config, applying cluster defaults.

    Returns a fully typed, frozen SrtConfig dataclass ready for use.

    Args:
        path: Path to the YAML configuration file
        config_overrides: Optional dict to deep-merge into the raw config before
            schema validation. Used by CLI flags (e.g., --nsys) to inject
            profiling config without modifying the YAML file.

    Returns:
        SrtConfig frozen dataclass

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load raw user config
    with open(path) as f:
        user_config = yaml.safe_load(f)

    # Apply CLI overrides (e.g., profiling flags) before resolving defaults
    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict) and isinstance(user_config.get(key), dict):
                user_config[key].update(value)
            else:
                user_config[key] = value

    # Load cluster defaults (optional)
    cluster_config = load_cluster_config()

    # Resolve with defaults (applies aliases and default values)
    resolved_config = resolve_config_with_defaults(user_config, cluster_config)

    # Parse with marshmallow schema to get typed SrtConfig
    try:
        schema = SrtConfig.Schema()
        config = schema.load(resolved_config)
        assert isinstance(config, SrtConfig)
        logger.info(f"Loaded config: {config.name}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid config in {path}: {e}") from e
