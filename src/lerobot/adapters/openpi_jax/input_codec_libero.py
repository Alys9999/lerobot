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

from lerobot.runtime.compatibility import validate_openpi_jax_policy_request
from lerobot.runtime.contracts import PolicyRequest

from .spec import OpenPIJaxLiberoSpec


def _to_uint8_hwc_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}.")
    if image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[-1]:
        image = np.moveaxis(image, 0, -1)
    if image.dtype.kind == "f":
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def _to_remote_image(image: np.ndarray, *, layout: str) -> np.ndarray:
    hwc_image = _to_uint8_hwc_image(image)
    if layout == "hwc":
        return hwc_image
    if layout == "chw":
        return np.ascontiguousarray(np.moveaxis(hwc_image, -1, 0))
    raise ValueError(f"Unsupported remote image layout {layout!r}.")


class OpenPIJaxLiberoInputCodec:
    def __init__(self, spec: OpenPIJaxLiberoSpec | None = None):
        self.spec = spec or OpenPIJaxLiberoSpec()

    def encode(self, req: PolicyRequest) -> dict[str, Any]:
        validate_openpi_jax_policy_request(req, self.spec)
        obs = req.observation
        payload: dict[str, Any] = {
            self.spec.prompt_remote_key: obs.task_text,
            self.spec.state_remote_key: np.asarray(
                obs.robot_state[self.spec.state_packet_key], dtype=np.float32
            ).reshape(-1),
        }
        image_payload = {
            remote_key: _to_remote_image(obs.images[alias], layout=self.spec.remote_image_layout)
            for alias, remote_key in self.spec.remote_image_keys.items()
        }
        if self.spec.remote_image_container_key is None:
            payload.update(image_payload)
        else:
            payload[self.spec.remote_image_container_key] = image_payload
        return payload
