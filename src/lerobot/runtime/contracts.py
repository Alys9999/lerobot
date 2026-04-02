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

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ObservationPacket:
    timestamp: float
    episode_id: str
    step_id: int
    images: dict[str, np.ndarray]
    robot_state: dict[str, np.ndarray]
    task_text: str
    task_id: str
    robot_id: str
    embodiment_id: str
    backend_id: str
    prev_action: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    privileged_state: dict[str, Any] | None = None


@dataclass(slots=True)
class ActionCommand:
    action_space: str
    values: np.ndarray
    gripper: float | np.ndarray | None = None
    control_dt: float = 0.0
    horizon: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RobotSpec:
    robot_id: str
    action_dim: int
    camera_keys: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSpec:
    task_suite: str
    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeSpec:
    control_dt: float
    max_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyRequest:
    observation: ObservationPacket
    robot_spec: RobotSpec
    task_spec: TaskSpec
    runtime_spec: RuntimeSpec


@dataclass(slots=True)
class PolicyResponse:
    action: ActionCommand
    raw_output: Any
    latency_ms: float
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    observations: list[ObservationPacket] = field(default_factory=list)
    actions: list[ActionCommand] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
