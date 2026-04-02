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

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass(frozen=True, slots=True)
class OpenPIJaxLiberoSpec:
    model_id: str = "openpi_jax_pi05_libero"
    runtime_transport: str = "websocket"
    packet_image_keys: dict[str, str] = field(
        default_factory=lambda: {
            "third_person": f"{OBS_IMAGES}.image",
            "left_wrist": f"{OBS_IMAGES}.image2",
        }
    )
    remote_image_keys: dict[str, str] = field(
        default_factory=lambda: {
            "third_person": "observation/image",
            "left_wrist": "observation/wrist_image",
        }
    )
    state_observation_key: str = OBS_STATE
    state_packet_key: str = "libero_state_8d"
    state_remote_key: str = "observation/state"
    state_dim: int = 8
    prompt_remote_key: str = "prompt"
    prompt_required: bool = True
    action_space: str = "env_native_7d"
    action_dim: int = 7
    action_horizon: int = 10

    @property
    def required_image_keys(self) -> tuple[str, ...]:
        return tuple(self.packet_image_keys.keys())
