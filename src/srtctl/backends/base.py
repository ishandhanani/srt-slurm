# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base types and protocols for backend configurations.
"""

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


class BackendType(str, Enum):
    """Supported backend types."""

    SGLANG = "sglang"


class BackendProtocol(Protocol):
    """Protocol that all backend configurations must implement.

    This allows frozen dataclasses to act as backends by implementing these methods.
    Each backend is responsible for:
    1. Allocating logical endpoints (serving units)
    2. Converting endpoints to physical processes
    3. Building commands to start those processes
    """

    @property
    def type(self) -> str:
        """Backend type identifier."""
        ...

    def get_config_for_mode(self, mode: str) -> dict[str, Any]:
        """Get config dict for a worker mode (prefill/decode/agg)."""
        ...

    def get_environment_for_mode(self, mode: str) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        ...

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> list["Endpoint"]:
        """Allocate logical endpoints based on resource requirements."""
        ...

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_sys_port: int = 8081,
        port_allocator: "NodePortAllocator | None" = None,
    ) -> list["Process"]:
        """Convert logical endpoints to physical processes."""
        ...

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: list["Process"],
        runtime: "RuntimeContext",
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Optional["Path"] = None,
    ) -> list[str]:
        """Build command to start a worker process."""
        ...
