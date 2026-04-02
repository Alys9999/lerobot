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

from typing import Any

import numpy as np

from .contracts import ActionCommand, PolicyRequest


class CompatibilityError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def validate_openpi_jax_policy_request(req: PolicyRequest, spec: Any) -> None:
    obs = req.observation
    for image_key in spec.required_image_keys:
        _require(image_key in obs.images, f"Missing required image key '{image_key}'.")
        image = np.asarray(obs.images[image_key])
        _require(image.ndim in (3, 4), f"Image '{image_key}' must have 3 or 4 dims, got {image.shape}.")

    _require(
        spec.state_packet_key in obs.robot_state,
        f"Missing required robot state key '{spec.state_packet_key}'.",
    )
    state = np.asarray(obs.robot_state[spec.state_packet_key], dtype=np.float32).reshape(-1)
    _require(
        state.shape[0] == spec.state_dim,
        f"Expected {spec.state_dim}D LIBERO state, got {state.shape[0]}.",
    )

    if spec.prompt_required:
        _require(bool(obs.task_text and obs.task_text.strip()), "Prompt is required for OpenPI JAX LIBERO.")

    _require(
        req.robot_spec.action_dim == spec.action_dim,
        f"Robot action dim {req.robot_spec.action_dim} != expected {spec.action_dim}.",
    )


def validate_action_command_for_spec(action: ActionCommand, spec: Any) -> None:
    values = np.asarray(action.values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    _require(values.ndim == 2, f"Expected action values with shape (H, D), got {values.shape}.")
    _require(
        action.horizon == values.shape[0],
        f"ActionCommand horizon {action.horizon} != action values horizon {values.shape[0]}.",
    )
    _require(
        values.shape[1] == spec.action_dim,
        f"Expected action dim {spec.action_dim}, got {values.shape[1]}.",
    )
    _require(
        values.shape[0] == spec.action_horizon,
        f"Expected action horizon {spec.action_horizon}, got {values.shape[0]}.",
    )
    _require(
        action.action_space == spec.action_space,
        f"Expected action space '{spec.action_space}', got '{action.action_space}'.",
    )
