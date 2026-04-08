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


@dataclass(slots=True)
class OpenPIJaxLiberoSpec:
    model_id: str = "openpi_jax_pi05_libero"
    env_type: str = "libero"
    runtime_transport: str = "websocket"
    robot_id: str = "franka_panda"
    embodiment_id: str = "libero"
    backend_id: str = "mujoco"
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
    state_remote_key: str | None = "observation/state"
    state_remote_keys: dict[str, list[int]] = field(default_factory=dict)
    state_dim: int = 8
    remote_image_container_key: str | None = None
    remote_image_layout: str = "hwc"
    prompt_remote_key: str = "prompt"
    prompt_required: bool = True
    action_space: str = "env_native_7d"
    action_dim: int = 7
    action_horizon: int = 10
    server_action_dim: int | None = None
    output_action_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        packet_aliases = set(self.packet_image_keys)
        remote_aliases = set(self.remote_image_keys)
        if packet_aliases != remote_aliases:
            raise ValueError(
                "OpenPIJaxLiberoSpec requires packet_image_keys and remote_image_keys to use the same aliases. "
                f"Got packet aliases={sorted(packet_aliases)} remote aliases={sorted(remote_aliases)}."
            )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}.")
        if self.state_remote_key is None and not self.state_remote_keys:
            raise ValueError("Either state_remote_key or state_remote_keys must be configured.")
        if self.state_remote_keys:
            seen_state_indices: set[int] = set()
            for remote_key, indices in self.state_remote_keys.items():
                if not remote_key:
                    raise ValueError("state_remote_keys cannot contain an empty remote key.")
                if not indices:
                    raise ValueError(f"state_remote_keys[{remote_key!r}] must contain at least one state index.")
                for raw_index in indices:
                    index = int(raw_index)
                    if index < 0 or index >= self.state_dim:
                        raise ValueError(
                            "state_remote_keys references an out-of-range state index. "
                            f"Got index={index} for state_dim={self.state_dim}."
                        )
                    if index in seen_state_indices:
                        raise ValueError(
                            "state_remote_keys cannot reference the same packet state index more than once. "
                            f"Got duplicate index={index}."
                        )
                    seen_state_indices.add(index)
            if seen_state_indices != set(range(self.state_dim)):
                raise ValueError(
                    "state_remote_keys must cover every packet state index exactly once. "
                    f"Got indices={sorted(seen_state_indices)} expected={list(range(self.state_dim))}."
                )
        if self.remote_image_layout not in {"hwc", "chw"}:
            raise ValueError(
                "remote_image_layout must be either 'hwc' or 'chw'. "
                f"Got {self.remote_image_layout!r}."
            )
        if self.output_action_indices and len(self.output_action_indices) != self.action_dim:
            raise ValueError(
                "output_action_indices must either be empty or have one entry per env action dimension. "
                f"Got len(output_action_indices)={len(self.output_action_indices)} action_dim={self.action_dim}."
            )

    @property
    def required_image_keys(self) -> tuple[str, ...]:
        return tuple(self.packet_image_keys.keys())
