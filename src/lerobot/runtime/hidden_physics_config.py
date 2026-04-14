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

"""Configuration dataclasses for the Hidden-Physics Diagnostic Benchmark v1.

This module defines the top-level benchmark configuration that drives the
evaluation runner.  It deliberately stays thin: it selects *what* to run
(families, templates, levels) and *how* to run it (episodes, output paths),
but does not define environment internals or policy codecs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Benchmark families & templates — canonical names
# ---------------------------------------------------------------------------

BENCHMARK_FAMILIES = ("F", "M", "C", "P", "R")
DIAGNOSTIC_FAMILIES = ("F", "M", "C", "P")
CHALLENGE_FAMILY = "R"

FAMILY_FULL_NAMES: dict[str, str] = {
    "F": "Effective Contact Friction",
    "M": "Object Mass",
    "C": "Object CoM / Inertia",
    "P": "Robot Passive Dynamics",
    "R": "Physics Reasoning / Multi-Strategy",
}

BENCHMARK_TEMPLATES = ("T1", "T2", "T3", "T4", "T5")
TEMPLATE_FULL_NAMES: dict[str, str] = {
    "T1": "Open Tabletop Transfer",
    "T2": "Container Deposit",
    "T3": "Elevation-Change Transfer",
    "T4": "Small-Target Placement",
    "T5": "Narrow-Opening Transfer",
}

IID_OOD_LEVELS = ("nominal", "iid_low", "iid_high", "ood_low", "ood_high")


# ---------------------------------------------------------------------------
# Policy mode
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PolicyModeConfig:
    """Selects between LeRobot-native and adapter-backed policy execution."""

    mode: str = "native"  # "native" | "openpi_adapter"
    native_policy_path: str = ""
    native_policy_overrides: dict[str, Any] = field(default_factory=dict)
    adapter_endpoint: str = ""
    adapter_spec: str = ""


# ---------------------------------------------------------------------------
# Runtime controls
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HiddenPhysicsRuntimeConfig:
    """Controls for a single benchmark run."""

    n_episodes_per_task: int = 10
    max_steps: int = 400
    write_trace: bool = True
    write_video: bool = False
    output_dir: str = "outputs/hidden_physics"
    run_name: str = ""
    fail_fast: bool = False
    seed: int = 42
    parallel_tasks: int = 1  # Number of tasks to evaluate in parallel (1 = sequential)


# ---------------------------------------------------------------------------
# Top-level benchmark config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HiddenPhysicsBenchmarkConfig:
    """Top-level entry point for a Hidden-Physics Diagnostic Benchmark run.

    This config is the single object that ``lerobot_hidden_physics_eval.py``
    reads to know what to run.
    """

    benchmark_name: str = "hidden_physics_v1"
    catalog_path: str = ""

    # Selection filters — empty means "all"
    families: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    iid_ood_levels: list[str] = field(default_factory=list)
    seed_groups: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)

    # Sub-configs
    policy: PolicyModeConfig = field(default_factory=PolicyModeConfig)
    runtime: HiddenPhysicsRuntimeConfig = field(default_factory=HiddenPhysicsRuntimeConfig)

    # Env overrides (passed through to make_env)
    env_type: str = "libero"
    env_overrides: dict[str, Any] = field(default_factory=dict)
