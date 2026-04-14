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

from lerobot.runtime.variation import (
    EffectiveContactFrictionVariation,
    JointArmatureVariation,
    JointDampingVariation,
    JointFrictionlossVariation,
    ObjectCoMOffsetVariation,
    ObjectInertiaDiagonalVariation,
    ObjectWeightVariation,
    VariationApplicationError,
    VariationConfig,
    VariationVariableConfig,
    VariationVariablesConfig,
    build_family_variation_profile,
    build_variation_profile,
)


# ---------------------------------------------------------------------------
# Fake MuJoCo model covering geoms, bodies, joints and DOFs
# ---------------------------------------------------------------------------


class _FakeModel:
    """Fake MuJoCo model surface used by the variation primitives.

    Exposes a bowl body, a bowl geom and two finger geoms plus a gripper
    hinge joint, which is the minimal shape needed to exercise F/M/C/P.
    """

    def __init__(self) -> None:
        # Geoms: 0=bowl_geom, 1=left_finger_pad, 2=right_finger_pad, 3=table_geom
        self.ngeom = 4
        self.geom_friction = np.ones((4, 3), dtype=np.float32)

        # Bodies: 0=bowl_body, 1=table_body
        self.nbody = 2
        self.body_mass = np.array([1.0, 2.0], dtype=np.float32)
        self.body_ipos = np.zeros((2, 3), dtype=np.float64)
        self.body_inertia = np.array(
            [[0.10, 0.10, 0.10], [1.00, 1.00, 1.00]], dtype=np.float64
        )

        # Joints / DOFs: 0=gripper_right_hinge, 1=gripper_left_hinge, 2=shoulder_joint
        self.njnt = 3
        self.nv = 3
        # Every joint is a hinge (type 3) with 1 DOF, in order.
        self.jnt_type = np.array([3, 3, 3], dtype=np.int32)
        self.jnt_dofadr = np.array([0, 1, 2], dtype=np.int32)
        self.dof_damping = np.array([0.10, 0.10, 0.10], dtype=np.float64)
        self.dof_frictionloss = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        self.dof_armature = np.array([0.00, 0.00, 0.00], dtype=np.float64)

    def geom_id2name(self, geom_id: int) -> str:
        return ["bowl_geom", "left_finger_pad", "right_finger_pad", "table_geom"][geom_id]

    def body_id2name(self, body_id: int) -> str:
        return ["bowl_body", "table_body"][body_id]

    def jnt_id2name(self, jnt_id: int) -> str:
        return ["gripper_right_hinge", "gripper_left_hinge", "shoulder_joint"][jnt_id]


class _FakeSim:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeInnerEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()


class _FakeEnv:
    def __init__(self) -> None:
        self._env = _FakeInnerEnv()


# ---------------------------------------------------------------------------
# Legacy bowl-smoke profile (unchanged — regression guard)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# F family — Effective Contact Friction
# ---------------------------------------------------------------------------


def test_effective_contact_friction_applies_to_object_and_finger_geoms():
    env = _FakeEnv()
    variation = EffectiveContactFrictionVariation(
        name="effective_contact_friction",
        target="bowl",  # required by the base dataclass; object_target drives behaviour
        min_value=0.4,
        max_value=0.4,
        object_target="bowl",
        finger_target="gripper_fingers",
    )

    variation.apply(env, 0.4)
    model = env._env.sim.model

    # Bowl geom (0) should be set.
    np.testing.assert_allclose(model.geom_friction[0], [0.4, 0.4, 0.4])
    # Both finger geoms (1, 2) should be set.
    np.testing.assert_allclose(model.geom_friction[1], [0.4, 0.4, 0.4])
    np.testing.assert_allclose(model.geom_friction[2], [0.4, 0.4, 0.4])
    # Table geom (3) untouched.
    np.testing.assert_allclose(model.geom_friction[3], [1.0, 1.0, 1.0])


def test_effective_contact_friction_tolerates_missing_finger_geoms():
    """Finger geoms may not exist in all envs — the primitive must not raise."""
    env = _FakeEnv()
    variation = EffectiveContactFrictionVariation(
        name="effective_contact_friction",
        target="bowl",
        min_value=0.5,
        max_value=0.5,
        object_target="bowl",
        finger_target="no_such_target",
    )
    variation.apply(env, 0.5)  # must not raise

    np.testing.assert_allclose(env._env.sim.model.geom_friction[0], [0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# M family — Object Mass
# ---------------------------------------------------------------------------


def test_object_weight_applies_to_body_mass():
    env = _FakeEnv()
    variation = ObjectWeightVariation(
        name="object_mass", target="bowl", min_value=0.25, max_value=0.25
    )
    variation.apply(env, 0.25)

    model = env._env.sim.model
    assert np.isclose(float(model.body_mass[0]), 0.25)
    # Table mass untouched.
    assert np.isclose(float(model.body_mass[1]), 2.0)


# ---------------------------------------------------------------------------
# C family — Object CoM / Inertia
# ---------------------------------------------------------------------------


def test_object_com_offset_is_additive_on_chosen_axis_and_preserves_mass():
    env = _FakeEnv()
    model = env._env.sim.model
    original_mass = model.body_mass.copy()

    # X-axis offset
    vx = ObjectCoMOffsetVariation(
        name="object_com_offset_x", target="bowl", min_value=0.02, max_value=0.02, axis=0
    )
    vx.apply(env, 0.02)
    # Y-axis offset
    vy = ObjectCoMOffsetVariation(
        name="object_com_offset_y", target="bowl", min_value=-0.01, max_value=-0.01, axis=1
    )
    vy.apply(env, -0.01)

    np.testing.assert_allclose(model.body_ipos[0], [0.02, -0.01, 0.0])
    # Table body untouched.
    np.testing.assert_allclose(model.body_ipos[1], [0.0, 0.0, 0.0])
    # Mass conservation (plan §7.3 guarantee #1).
    np.testing.assert_array_equal(model.body_mass, original_mass)


def test_object_inertia_diagonal_scales_and_preserves_mass():
    env = _FakeEnv()
    model = env._env.sim.model
    original_mass = model.body_mass.copy()

    variation = ObjectInertiaDiagonalVariation(
        name="object_inertia_diagonal", target="bowl", min_value=1.5, max_value=1.5
    )
    variation.apply(env, 1.5)

    np.testing.assert_allclose(model.body_inertia[0], [0.15, 0.15, 0.15])
    # Table body untouched.
    np.testing.assert_allclose(model.body_inertia[1], [1.0, 1.0, 1.0])
    # Mass conservation (plan §7.3 guarantee #1).
    np.testing.assert_array_equal(model.body_mass, original_mass)


# ---------------------------------------------------------------------------
# P family — Robot Passive Dynamics
# ---------------------------------------------------------------------------


def test_joint_damping_applies_only_to_matching_dofs():
    env = _FakeEnv()
    variation = JointDampingVariation(
        name="joint_damping", target="gripper", min_value=0.5, max_value=0.5
    )
    variation.apply(env, 0.5)

    model = env._env.sim.model
    # gripper_right_hinge and gripper_left_hinge match; shoulder_joint does not.
    np.testing.assert_allclose(model.dof_damping, [0.5, 0.5, 0.10])


def test_joint_frictionloss_applies_only_to_matching_dofs():
    env = _FakeEnv()
    variation = JointFrictionlossVariation(
        name="joint_frictionloss", target="gripper", min_value=0.02, max_value=0.02
    )
    variation.apply(env, 0.02)

    np.testing.assert_allclose(
        env._env.sim.model.dof_frictionloss, [0.02, 0.02, 0.01]
    )


def test_joint_armature_applies_only_to_matching_dofs():
    env = _FakeEnv()
    variation = JointArmatureVariation(
        name="joint_armature", target="gripper", min_value=0.003, max_value=0.003
    )
    variation.apply(env, 0.003)

    np.testing.assert_allclose(
        env._env.sim.model.dof_armature, [0.003, 0.003, 0.0]
    )


def test_joint_variation_raises_when_no_target_matches():
    env = _FakeEnv()
    variation = JointDampingVariation(
        name="joint_damping", target="no_such_joint", min_value=0.5, max_value=0.5
    )
    with pytest.raises(VariationApplicationError):
        variation.apply(env, 0.5)


# ---------------------------------------------------------------------------
# Family-aware profile builder
# ---------------------------------------------------------------------------


def test_build_family_variation_profile_F_samples_effective_friction():
    profile = build_family_variation_profile(
        family="F",
        profile_name="F_smoke",
        target="bowl",
        ranges={"effective_contact_friction": (0.3, 0.3)},
    )
    assert [v.name for v in profile.variables] == ["effective_contact_friction"]

    env = _FakeEnv()
    sampled = profile.sample_all(np.random.default_rng(0))
    profile.apply_all(env, sampled)

    np.testing.assert_allclose(env._env.sim.model.geom_friction[0], [0.3, 0.3, 0.3])
    np.testing.assert_allclose(env._env.sim.model.geom_friction[1], [0.3, 0.3, 0.3])


def test_build_family_variation_profile_M_samples_object_mass():
    profile = build_family_variation_profile(
        family="M", profile_name="M_smoke", target="bowl",
        ranges={"object_mass": (0.7, 0.7)},
    )
    assert [v.name for v in profile.variables] == ["object_mass"]

    env = _FakeEnv()
    profile.apply_all(env, profile.sample_all(np.random.default_rng(0)))

    assert np.isclose(float(env._env.sim.model.body_mass[0]), 0.7)


def test_build_family_variation_profile_C_samples_com_and_inertia_and_preserves_mass():
    profile = build_family_variation_profile(
        family="C", profile_name="C_smoke", target="bowl",
        ranges={
            "object_com_offset_x": (0.01, 0.01),
            "object_com_offset_y": (0.02, 0.02),
            "object_inertia_diagonal": (2.0, 2.0),
        },
    )
    names = [v.name for v in profile.variables]
    assert set(names) == {"object_com_offset_x", "object_com_offset_y", "object_inertia_diagonal"}

    env = _FakeEnv()
    original_mass = env._env.sim.model.body_mass.copy()
    profile.apply_all(env, profile.sample_all(np.random.default_rng(0)))

    np.testing.assert_allclose(env._env.sim.model.body_ipos[0], [0.01, 0.02, 0.0])
    np.testing.assert_allclose(env._env.sim.model.body_inertia[0], [0.20, 0.20, 0.20])
    # Mass conservation across the full C-family apply (plan §7.3 guarantee #1).
    np.testing.assert_array_equal(env._env.sim.model.body_mass, original_mass)


def test_build_family_variation_profile_P_samples_passive_dynamics():
    profile = build_family_variation_profile(
        family="P", profile_name="P_smoke", target="gripper",
        ranges={
            "joint_damping": (0.4, 0.4),
            "joint_frictionloss": (0.03, 0.03),
            "joint_armature": (0.005, 0.005),
        },
    )
    names = [v.name for v in profile.variables]
    assert set(names) == {"joint_damping", "joint_frictionloss", "joint_armature"}

    env = _FakeEnv()
    profile.apply_all(env, profile.sample_all(np.random.default_rng(0)))

    model = env._env.sim.model
    np.testing.assert_allclose(model.dof_damping, [0.4, 0.4, 0.10])
    np.testing.assert_allclose(model.dof_frictionloss, [0.03, 0.03, 0.01])
    np.testing.assert_allclose(model.dof_armature, [0.005, 0.005, 0.0])


def test_build_family_variation_profile_filters_by_supplied_ranges():
    """Only variables present in *ranges* should be enabled."""
    profile = build_family_variation_profile(
        family="C", profile_name="C_partial", target="bowl",
        ranges={"object_com_offset_x": (0.01, 0.01)},
    )
    assert [v.name for v in profile.variables] == ["object_com_offset_x"]


def test_build_family_variation_profile_rejects_unknown_family():
    with pytest.raises(ValueError):
        build_family_variation_profile(
            family="Z", profile_name="bad", target="bowl", ranges={}
        )
