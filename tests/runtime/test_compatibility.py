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

from lerobot.adapters.openpi_jax.spec import OpenPIJaxLiberoSpec
from lerobot.runtime.compatibility import (
    CompatibilityError,
    validate_action_command_for_spec,
    validate_openpi_jax_policy_request,
)
from lerobot.runtime.contracts import ActionCommand, ObservationPacket, PolicyRequest, RobotSpec, RuntimeSpec, TaskSpec


def _make_request(*, state_dim: int = 8) -> PolicyRequest:
    return PolicyRequest(
        observation=ObservationPacket(
            timestamp=0.0,
            episode_id="ep0",
            step_id=0,
            images={
                "third_person": np.zeros((8, 8, 3), dtype=np.uint8),
                "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            robot_state={"libero_state_8d": np.zeros(state_dim, dtype=np.float32)},
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


def test_validate_openpi_jax_policy_request_accepts_valid_request():
    validate_openpi_jax_policy_request(_make_request(), OpenPIJaxLiberoSpec())


def test_validate_openpi_jax_policy_request_uses_spec_state_dim():
    request = _make_request(state_dim=6)
    validate_openpi_jax_policy_request(request, OpenPIJaxLiberoSpec(state_dim=6))


def test_validate_openpi_jax_policy_request_rejects_bad_state_dim():
    request = _make_request()
    request.observation.robot_state["libero_state_8d"] = np.zeros(6, dtype=np.float32)

    with pytest.raises(CompatibilityError, match="Expected 8D LIBERO state"):
        validate_openpi_jax_policy_request(request, OpenPIJaxLiberoSpec())


def test_validate_action_command_for_spec_rejects_bad_horizon():
    action = ActionCommand(
        action_space="env_native_7d",
        values=np.zeros((4, 7), dtype=np.float32),
        horizon=4,
    )

    with pytest.raises(CompatibilityError, match="Expected action horizon 10"):
        validate_action_command_for_spec(action, OpenPIJaxLiberoSpec())
