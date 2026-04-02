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

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ActionCommand, EpisodeTrace, ObservationPacket


def summarize_observation(packet: ObservationPacket) -> dict[str, Any]:
    return {
        "step_id": packet.step_id,
        "task_text": packet.task_text,
        "task_id": packet.task_id,
        "image_keys": list(packet.images.keys()),
        "image_shapes": {key: list(np.asarray(value).shape) for key, value in packet.images.items()},
        "state_shapes": {key: list(np.asarray(value).shape) for key, value in packet.robot_state.items()},
    }


def summarize_action(action: ActionCommand) -> dict[str, Any]:
    return {
        "action_space": action.action_space,
        "shape": list(np.asarray(action.values).shape),
        "horizon": action.horizon,
        "metadata": dict(action.metadata),
    }


def episode_trace_to_summary_dict(trace: EpisodeTrace) -> dict[str, Any]:
    return {
        "success": trace.success,
        "metrics": dict(trace.metrics),
        "metadata": dict(trace.metadata),
        "observations": [summarize_observation(packet) for packet in trace.observations],
        "actions": [summarize_action(action) for action in trace.actions],
        "rewards": list(trace.rewards),
        "dones": list(trace.dones),
        "infos": trace.infos,
    }


def write_episode_trace(trace: EpisodeTrace, output_dir: Path, episode_index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{episode_index:03d}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(episode_trace_to_summary_dict(trace), handle, indent=2)
    return output_path


def read_episode_trace_summary(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
