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

from lerobot.runtime.contracts import ActionCommand, PolicyRequest

from .spec import OpenPIJaxLiberoSpec


class OpenPIJaxLiberoOutputCodec:
    def __init__(self, spec: OpenPIJaxLiberoSpec | None = None):
        self.spec = spec or OpenPIJaxLiberoSpec()

    def decode(self, model_output: dict[str, Any], req: PolicyRequest) -> ActionCommand:
        if "actions" not in model_output:
            raise ValueError("OpenPI JAX output did not contain the required 'actions' key.")

        values = np.asarray(model_output["actions"], dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"Expected action chunk with shape (H, D), got {values.shape}.")
        if values.shape[1] < self.spec.action_dim:
            raise ValueError(
                f"Expected action dim >= {self.spec.action_dim}, got action chunk with shape {values.shape}."
            )

        values = np.ascontiguousarray(values[:, : self.spec.action_dim])
        return ActionCommand(
            action_space=self.spec.action_space,
            values=values,
            control_dt=req.runtime_spec.control_dt,
            horizon=values.shape[0],
            metadata={
                "model_id": self.spec.model_id,
                "raw_action_shape": list(np.asarray(model_output["actions"]).shape),
            },
        )
