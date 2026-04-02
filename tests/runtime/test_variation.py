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

from lerobot.runtime.variation import (
    VariationConfig,
    VariationVariableConfig,
    VariationVariablesConfig,
    build_variation_profile,
)


class _FakeModel:
    def __init__(self):
        self.ngeom = 3
        self.nbody = 2
        self.geom_friction = np.ones((3, 3), dtype=np.float32)
        self.body_mass = np.array([1.0, 2.0], dtype=np.float32)

    def geom_id2name(self, geom_id: int) -> str:
        return ["bowl_geom", "left_finger_pad", "table_geom"][geom_id]

    def body_id2name(self, body_id: int) -> str:
        return ["bowl_body", "table_body"][body_id]


class _FakeSim:
    def __init__(self):
        self.model = _FakeModel()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeInnerEnv:
    def __init__(self):
        self.sim = _FakeSim()


class _FakeEnv:
    def __init__(self):
        self._env = _FakeInnerEnv()


def test_build_variation_profile_samples_and_applies_values():
    profile = build_variation_profile(
        VariationConfig(
            variables=VariationVariablesConfig(
                object_friction=VariationVariableConfig(enabled=True, target="bowl", range=(0.2, 0.2)),
                finger_friction=VariationVariableConfig(
                    enabled=True, target="gripper_fingers", range=(0.6, 0.6)
                ),
                object_weight=VariationVariableConfig(enabled=True, target="bowl", range=(0.3, 0.3)),
            )
        )
    )
    env = _FakeEnv()

    sampled = profile.sample_all(np.random.default_rng(0))
    profile.apply_all(env, sampled)

    np.testing.assert_allclose(env._env.sim.model.geom_friction[0], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(env._env.sim.model.geom_friction[1], [0.6, 0.6, 0.6])
    assert np.isclose(float(env._env.sim.model.body_mass[0]), 0.3)
    assert env._env.sim.forward_calls == 3
