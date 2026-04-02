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
from lerobot.runtime.trace import episode_trace_to_summary_dict


def test_episode_trace_summary_contains_observations_and_actions():
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
                values=np.zeros((2, 7), dtype=np.float32),
                horizon=2,
            )
        ],
        rewards=[1.0],
        dones=[False],
        success=True,
    )

    summary = episode_trace_to_summary_dict(trace)

    assert summary["success"] is True
    assert summary["observations"][0]["image_keys"] == ["third_person"]
    assert summary["actions"][0]["shape"] == [2, 7]
