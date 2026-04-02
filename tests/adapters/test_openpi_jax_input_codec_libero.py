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

from lerobot.adapters.openpi_jax.input_codec_libero import OpenPIJaxLiberoInputCodec
from lerobot.runtime.contracts import ObservationPacket, PolicyRequest, RobotSpec, RuntimeSpec, TaskSpec


def test_openpi_jax_input_codec_libero_encodes_expected_keys():
    codec = OpenPIJaxLiberoInputCodec()
    packet = ObservationPacket(
        timestamp=0.0,
        episode_id="ep0",
        step_id=0,
        images={
            "third_person": np.zeros((32, 32, 3), dtype=np.uint8),
            "left_wrist": np.ones((32, 32, 3), dtype=np.uint8),
        },
        robot_state={"libero_state_8d": np.arange(8, dtype=np.float32)},
        task_text="pick up the bowl",
        task_id="3",
        robot_id="franka_panda",
        embodiment_id="libero",
        backend_id="mujoco",
    )
    request = PolicyRequest(
        observation=packet,
        robot_spec=RobotSpec(robot_id="franka_panda", action_dim=7),
        task_spec=TaskSpec(task_suite="libero_object", task_id="3", prompt="pick up the bowl"),
        runtime_spec=RuntimeSpec(control_dt=1 / 30, max_steps=300),
    )

    encoded = codec.encode(request)

    assert sorted(encoded.keys()) == [
        "observation/image",
        "observation/state",
        "observation/wrist_image",
        "prompt",
    ]
    assert encoded["observation/image"].shape == (32, 32, 3)
    assert encoded["observation/image"].dtype == np.uint8
    assert encoded["observation/wrist_image"].shape == (32, 32, 3)
    assert encoded["observation/state"].shape == (8,)
    assert encoded["observation/state"].dtype == np.float32
    assert encoded["prompt"] == "pick up the bowl"
