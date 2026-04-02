#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import yaml


def _default_libero_config_root() -> Path:
    return Path(tempfile.gettempdir()) / "lerobot_libero"


def ensure_libero_runtime_ready(config_root: str | os.PathLike[str] | None = None) -> Path:
    """Prepare a non-interactive LIBERO runtime configuration.

    LIBERO creates its config file lazily and prompts on stdin the first time it is
    imported if `LIBERO_CONFIG_PATH/config.yaml` does not exist. This helper creates a
    valid config ahead of time so CLI commands can run unattended in local smoke tests.
    """

    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    resolved_root = Path(config_root) if config_root is not None else Path(
        os.environ.get("LIBERO_CONFIG_PATH", _default_libero_config_root())
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)

    config_path = resolved_root / "config.yaml"
    if config_path.exists():
        return config_path

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise ImportError("LIBERO is not installed. Install the `lerobot[libero]` extra to use LIBERO.")

    package_root = Path(spec.origin).resolve().parent
    default_config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str((package_root / "../datasets").resolve()),
        "assets": str(package_root / "assets"),
    }
    config_path.write_text(yaml.safe_dump(default_config), encoding="utf-8")
    return config_path
