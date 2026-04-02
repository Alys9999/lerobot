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
import pytest

from lerobot.adapters.openpi_jax.output_codec_libero import OpenPIJaxLiberoOutputCodec
from lerobot.runtime.contracts import ObservationPacket, PolicyRequest, RobotSpec, RuntimeSpec, TaskSpec


def test_openpi_jax_output_codec_libero_slices_to_env_action_dim():
    codec = OpenPIJaxLiberoOutputCodec()
    request = PolicyRequest(
        observation=ObservationPacket(
            timestamp=0.0,
            episode_id="ep0",
            step_id=0,
            images={
                "third_person": np.zeros((8, 8, 3), dtype=np.uint8),
                "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            robot_state={"libero_state_8d": np.zeros(8, dtype=np.float32)},
            task_text="pick up the bowl",
            task_id="3",
            robot_id="franka_panda",
            embodiment_id="libero",
            backend_id="mujoco",
        ),
        robot_spec=RobotSpec(robot_id="franka_panda", action_dim=7),
        task_spec=TaskSpec(task_suite="libero_object", task_id="3", prompt="pick up the bowl"),
        runtime_spec=RuntimeSpec(control_dt=1 / 30, max_steps=300),
    )

    output = codec.decode({"actions": np.ones((codec.spec.action_horizon, 9), dtype=np.float32)}, request)

    assert output.action_space == "env_native_7d"
    assert output.values.shape == (codec.spec.action_horizon, 7)
    assert output.horizon == codec.spec.action_horizon
    assert output.control_dt == 1 / 30


def test_openpi_jax_output_codec_libero_rejects_bad_horizon():
    codec = OpenPIJaxLiberoOutputCodec()
    request = PolicyRequest(
        observation=ObservationPacket(
            timestamp=0.0,
            episode_id="ep0",
            step_id=0,
            images={
                "third_person": np.zeros((8, 8, 3), dtype=np.uint8),
                "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            robot_state={"libero_state_8d": np.zeros(8, dtype=np.float32)},
            task_text="pick up the bowl",
            task_id="3",
            robot_id="franka_panda",
            embodiment_id="libero",
            backend_id="mujoco",
        ),
        robot_spec=RobotSpec(robot_id="franka_panda", action_dim=7),
        task_spec=TaskSpec(task_suite="libero_object", task_id="3", prompt="pick up the bowl"),
        runtime_spec=RuntimeSpec(control_dt=1 / 30, max_steps=300),
    )

    with pytest.raises(ValueError, match="Expected action horizon 10"):
        codec.decode({"actions": np.ones((4, 9), dtype=np.float32)}, request)
