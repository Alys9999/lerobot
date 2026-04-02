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

import numpy as np

from lerobot.runtime.contracts import ActionCommand, EpisodeTrace, ObservationPacket
from lerobot.runtime.trace import read_episode_trace_summary, write_episode_trace


def test_write_and_read_episode_trace_summary_round_trip(tmp_path):
    trace = EpisodeTrace(
        observations=[
            ObservationPacket(
                timestamp=0.0,
                episode_id="ep0",
                step_id=0,
                images={"third_person": np.zeros((8, 8, 3), dtype=np.uint8)},
                robot_state={"libero_state_8d": np.zeros(8, dtype=np.float32)},
                task_text="pick up the bowl",
                task_id="3",
                robot_id="franka_panda",
                embodiment_id="libero",
                backend_id="mujoco",
            )
        ],
        actions=[
            ActionCommand(
                action_space="env_native_7d",
                values=np.zeros((10, 7), dtype=np.float32),
                horizon=10,
            )
        ],
        success=True,
        metrics={"avg_latency_ms": 12.5},
        metadata={
            "variation": {"object_friction": 0.2, "finger_friction": 0.6, "object_weight": 0.3},
            "action_chunk_shapes": [[10, 7]],
        },
    )

    trace_path = write_episode_trace(trace, tmp_path, episode_index=0)
    loaded = read_episode_trace_summary(trace_path)

    assert trace_path.exists()
    assert loaded["success"] is True
    assert loaded["metadata"]["variation"]["object_weight"] == 0.3
    assert loaded["metadata"]["action_chunk_shapes"] == [[10, 7]]
    assert loaded["actions"][0]["shape"] == [10, 7]
