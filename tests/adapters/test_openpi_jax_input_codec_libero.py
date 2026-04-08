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
from lerobot.adapters.openpi_jax.spec import OpenPIJaxLiberoSpec
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


def test_openpi_jax_input_codec_libero_supports_nested_chw_images():
    codec = OpenPIJaxLiberoInputCodec(
        OpenPIJaxLiberoSpec(
            env_type="aloha",
            robot_id="aloha",
            embodiment_id="aloha",
            backend_id="gym_aloha",
            packet_image_keys={"cam_high": "observation.images.top"},
            remote_image_keys={"cam_high": "cam_high"},
            state_packet_key="state",
            state_remote_key="state",
            state_dim=14,
            remote_image_container_key="images",
            remote_image_layout="chw",
            prompt_required=False,
            action_space="env_native_14d",
            action_dim=14,
            action_horizon=50,
        )
    )
    packet = ObservationPacket(
        timestamp=0.0,
        episode_id="ep0",
        step_id=0,
        images={"cam_high": np.zeros((32, 32, 3), dtype=np.uint8)},
        robot_state={"state": np.arange(14, dtype=np.float32)},
        task_text="transfer cube",
        task_id="0",
        robot_id="aloha",
        embodiment_id="aloha",
        backend_id="gym_aloha",
    )
    request = PolicyRequest(
        observation=packet,
        robot_spec=RobotSpec(robot_id="aloha", action_dim=14),
        task_spec=TaskSpec(task_suite="aloha", task_id="0", prompt="transfer cube"),
        runtime_spec=RuntimeSpec(control_dt=1 / 50, max_steps=200),
    )

    encoded = codec.encode(request)

    assert sorted(encoded.keys()) == ["images", "prompt", "state"]
    assert sorted(encoded["images"].keys()) == ["cam_high"]
    assert encoded["images"]["cam_high"].shape == (3, 32, 32)
    assert encoded["state"].shape == (14,)


def test_openpi_jax_input_codec_libero_supports_split_state_keys():
    codec = OpenPIJaxLiberoInputCodec(
        OpenPIJaxLiberoSpec(
            model_id="openpi_jax_pi05_droid",
            env_type="droid",
            robot_id="franka_panda",
            embodiment_id="droid",
            backend_id="droid",
            packet_image_keys={
                "base": "observation.images.image",
                "left_wrist": "observation.images.image2",
            },
            remote_image_keys={
                "base": "observation/exterior_image_1_left",
                "left_wrist": "observation/wrist_image_left",
            },
            state_packet_key="droid_state_8d",
            state_remote_key=None,
            state_remote_keys={
                "observation/joint_position": [0, 1, 2, 3, 4, 5, 6],
                "observation/gripper_position": [7],
            },
            state_dim=8,
            action_space="env_native_7d",
            action_dim=7,
            action_horizon=15,
            server_action_dim=8,
            output_action_indices=[0, 1, 2, 3, 4, 5, 6],
        )
    )
    packet = ObservationPacket(
        timestamp=0.0,
        episode_id="ep0",
        step_id=0,
        images={
            "base": np.zeros((32, 32, 3), dtype=np.uint8),
            "left_wrist": np.ones((32, 32, 3), dtype=np.uint8),
        },
        robot_state={"droid_state_8d": np.arange(8, dtype=np.float32)},
        task_text="pick up the bowl",
        task_id="0",
        robot_id="franka_panda",
        embodiment_id="droid",
        backend_id="droid",
    )
    request = PolicyRequest(
        observation=packet,
        robot_spec=RobotSpec(robot_id="franka_panda", action_dim=7),
        task_spec=TaskSpec(task_suite="libero_spatial", task_id="0", prompt="pick up the bowl"),
        runtime_spec=RuntimeSpec(control_dt=1 / 30, max_steps=300),
    )

    encoded = codec.encode(request)

    assert sorted(encoded.keys()) == [
        "observation/exterior_image_1_left",
        "observation/gripper_position",
        "observation/joint_position",
        "observation/wrist_image_left",
        "prompt",
    ]
    assert encoded["observation/joint_position"].shape == (7,)
    assert encoded["observation/gripper_position"].shape == (1,)
