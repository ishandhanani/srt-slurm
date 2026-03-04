# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comment-aware YAML utilities using ruamel.yaml.

Provides load/dump that preserve comments and a merge that keeps base field
order with override fields appended at the end.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import IO, Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def _make_yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 120
    y.best_sequence_indent = 2
    y.best_map_flow_style = False
    return y


def load_yaml_with_comments(path: Path) -> CommentedMap:
    """Load a YAML file preserving comments and key insertion order."""
    y = _make_yaml()
    with open(path) as f:
        result = y.load(f)
    if not isinstance(result, CommentedMap):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(result).__name__}")
    return result


def dump_yaml_with_comments(data: Any, stream: IO[str] | None = None) -> str | None:
    """Dump YAML preserving comments. Returns str when stream is None."""
    y = _make_yaml()
    if stream is None:
        buf = io.StringIO()
        y.dump(data, buf)
        return buf.getvalue()
    y.dump(data, stream)
    return None


def comment_aware_merge(base: CommentedMap, override: CommentedMap | dict[str, Any]) -> CommentedMap:
    """Merge *override* into *base*, preserving base field order and comments.

    Rules:
    - Base keys: kept in their original order, comments preserved, values updated from override.
    - ``None`` value in override → key removed from result.
    - Nested dicts: recursively merged with the same rules.
    - New keys in override (absent from base): appended at end in override order,
      with their override comments when the override is a CommentedMap.
    """
    result = CommentedMap()

    # Pass 1 — base keys in base order
    for key in list(base.keys()):
        if key in override:
            val = override[key]
            if val is None:
                continue  # None → delete key
            if isinstance(base[key], CommentedMap) and isinstance(val, (dict, CommentedMap)):
                result[key] = comment_aware_merge(base[key], val)
            else:
                result[key] = copy.deepcopy(val)
        else:
            result[key] = copy.deepcopy(base[key])

        # Carry over comment tokens attached to this key in the base
        if key in base.ca.items:
            result.ca.items[key] = base.ca.items[key]

    # Pass 2 — new keys from override not present in base
    for key in override:
        if key not in base and override[key] is not None:
            result[key] = copy.deepcopy(override[key])
            # Carry over comment from override (only available for CommentedMap)
            if isinstance(override, CommentedMap) and key in override.ca.items:
                result.ca.items[key] = override.ca.items[key]

    # Block comment before the first key (e.g. "# section header")
    if base.ca.comment is not None:
        result.ca.comment = base.ca.comment

    return result
