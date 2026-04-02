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

from pathlib import Path

from lerobot.envs.libero_bootstrap import ensure_libero_runtime_ready


def test_ensure_libero_runtime_ready_creates_noninteractive_config(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    config_path = ensure_libero_runtime_ready(tmp_path)

    assert config_path == Path(tmp_path) / "config.yaml"
    assert config_path.exists()
    assert "benchmark_root" in config_path.read_text(encoding="utf-8")
