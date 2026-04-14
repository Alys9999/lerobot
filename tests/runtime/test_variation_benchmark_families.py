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

"""Tests for the benchmark-family variation primitives (C and P families)
and the ``build_family_variation_profile`` helper."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.runtime.variation import (
    FAMILY_VARIABLE_REGISTRY,
    EffectiveContactFrictionVariation,
    ObjectCoMOffsetVariation,
    ObjectInertiaDiagonalVariation,
    JointArmatureVariation,
    JointDampingVariation,
    JointFrictionlossVariation,
    build_family_variation_profile,
)


# ---------------------------------------------------------------------------
# Fake MuJoCo environment stubs
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self):
        self.ngeom = 2
        self.nbody = 2
        self.njnt = 3
        self.geom_friction = np.ones((2, 3), dtype=np.float64)
        self.body_mass = np.array([0.5, 1.0], dtype=np.float64)
        self.body_ipos = np.zeros((2, 3), dtype=np.float64)
        self.body_inertia = np.ones((2, 3), dtype=np.float64)
        self.dof_damping = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.dof_frictionloss = np.zeros(3, dtype=np.float64)
        self.dof_armature = np.zeros(3, dtype=np.float64)
        self.jnt_dofadr = np.array([0, 1, 2], dtype=np.int32)
        self.jnt_type = np.array([3, 3, 3], dtype=np.int32)  # all hinge

    def geom_id2name(self, gid: int) -> str:
        return ["bowl_geom", "table_geom"][gid]

    def body_id2name(self, bid: int) -> str:
        return ["bowl_body", "table_body"][bid]

    def jnt_id2name(self, jid: int) -> str:
        return ["finger_l_joint", "finger_r_joint", "wrist_joint"][jid]


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


# ---------------------------------------------------------------------------
# F-family: Effective Contact Friction (single axis)
# ---------------------------------------------------------------------------


class TestEffectiveContactFrictionVariation:
    def test_applies_to_both_object_and_finger_geoms(self):
        """A single sampled value should be applied to both object and finger geoms."""
        env = _FakeEnv()
        # Add a finger geom to the model so both paths are exercised
        model = env._env.sim.model
        model.ngeom = 3
        model.geom_friction = np.ones((3, 3), dtype=np.float64)
        orig_id2name = model.geom_id2name
        model.geom_id2name = lambda gid: ["bowl_geom", "table_geom", "finger_geom"][gid]

        var = EffectiveContactFrictionVariation(
            name="effective_contact_friction",
            target="bowl",
            object_target="bowl",
            finger_target="finger",
            min_value=0.5,
            max_value=0.5,
        )
        var.apply(env, 0.5)

        # Bowl geom should be set to 0.5
        np.testing.assert_allclose(model.geom_friction[0, :], 0.5)
        # Finger geom should also be set to 0.5
        np.testing.assert_allclose(model.geom_friction[2, :], 0.5)
        # Table should be unchanged
        np.testing.assert_allclose(model.geom_friction[1, :], 1.0)

    def test_single_value_in_sample(self):
        """F-family should produce exactly one sampled value, not two."""
        profile = build_family_variation_profile(
            family="F",
            profile_name="test",
            target="bowl",
            ranges={"effective_contact_friction": (0.3, 0.7)},
        )
        rng = np.random.default_rng(42)
        sampled = profile.sample_all(rng)
        assert len(sampled) == 1
        assert "effective_contact_friction" in sampled
        # Must NOT have separate object/finger keys
        assert "effective_contact_friction_object" not in sampled
        assert "effective_contact_friction_finger" not in sampled


# ---------------------------------------------------------------------------
# C-family: CoM offset
# ---------------------------------------------------------------------------


class TestObjectCoMOffsetVariation:
    def test_apply_offsets_body_ipos(self):
        env = _FakeEnv()
        var = ObjectCoMOffsetVariation(name="com_x", target="bowl", min_value=-0.01, max_value=0.01, axis=0)
        var.apply(env, 0.005)
        np.testing.assert_allclose(env._env.sim.model.body_ipos[0, 0], 0.005)
        # Other axes unchanged
        assert env._env.sim.model.body_ipos[0, 1] == 0.0

    def test_apply_y_axis(self):
        env = _FakeEnv()
        var = ObjectCoMOffsetVariation(name="com_y", target="bowl", min_value=-0.02, max_value=0.02, axis=1)
        var.apply(env, -0.01)
        np.testing.assert_allclose(env._env.sim.model.body_ipos[0, 1], -0.01)


# ---------------------------------------------------------------------------
# C-family: Inertia diagonal scale
# ---------------------------------------------------------------------------


class TestObjectInertiaDiagonalVariation:
    def test_apply_scales_inertia(self):
        env = _FakeEnv()
        var = ObjectInertiaDiagonalVariation(name="inertia", target="bowl", min_value=0.5, max_value=2.0)
        var.apply(env, 2.0)
        np.testing.assert_allclose(env._env.sim.model.body_inertia[0], [2.0, 2.0, 2.0])
        # Table body should remain unchanged
        np.testing.assert_allclose(env._env.sim.model.body_inertia[1], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# P-family: Joint damping / frictionloss / armature
# ---------------------------------------------------------------------------


class TestJointDampingVariation:
    def test_apply_sets_dof_damping_for_matching_joints(self):
        env = _FakeEnv()
        var = JointDampingVariation(name="damping", target="finger", min_value=0.0, max_value=1.0)
        var.apply(env, 0.5)
        # finger_l_joint (dof 0) and finger_r_joint (dof 1) should be set
        assert env._env.sim.model.dof_damping[0] == 0.5
        assert env._env.sim.model.dof_damping[1] == 0.5
        # wrist_joint (dof 2) should be unchanged
        assert env._env.sim.model.dof_damping[2] == 0.1


class TestJointFrictionlossVariation:
    def test_apply_sets_dof_frictionloss(self):
        env = _FakeEnv()
        var = JointFrictionlossVariation(name="floss", target="finger", min_value=0.0, max_value=0.5)
        var.apply(env, 0.3)
        assert env._env.sim.model.dof_frictionloss[0] == 0.3
        assert env._env.sim.model.dof_frictionloss[1] == 0.3
        assert env._env.sim.model.dof_frictionloss[2] == 0.0


class TestJointArmatureVariation:
    def test_apply_sets_dof_armature(self):
        env = _FakeEnv()
        var = JointArmatureVariation(name="arm", target="wrist", min_value=0.0, max_value=0.5)
        var.apply(env, 0.2)
        assert env._env.sim.model.dof_armature[2] == 0.2
        # Finger joints unchanged
        assert env._env.sim.model.dof_armature[0] == 0.0


# ---------------------------------------------------------------------------
# build_family_variation_profile
# ---------------------------------------------------------------------------


class TestBuildFamilyVariationProfile:
    def test_build_f_family(self):
        profile = build_family_variation_profile(
            family="F",
            profile_name="test_f",
            target="bowl",
            ranges={
                "effective_contact_friction": (0.3, 1.5),
            },
        )
        assert profile.profile_name == "test_f"
        assert len(profile.variables) == 1
        assert profile.variables[0].name == "effective_contact_friction"

    def test_build_m_family(self):
        profile = build_family_variation_profile(
            family="M",
            profile_name="test_m",
            target="bowl",
            ranges={"object_mass": (0.05, 0.40)},
        )
        assert len(profile.variables) == 1

    def test_build_c_family(self):
        profile = build_family_variation_profile(
            family="C",
            profile_name="test_c",
            target="bowl",
            ranges={
                "object_com_offset_x": (-0.01, 0.01),
                "object_com_offset_y": (-0.01, 0.01),
                "object_inertia_diagonal": (0.5, 2.0),
            },
        )
        assert len(profile.variables) == 3

    def test_build_p_family(self):
        profile = build_family_variation_profile(
            family="P",
            profile_name="test_p",
            target="gripper_fingers",
            ranges={
                "joint_damping": (0.01, 0.5),
                "joint_frictionloss": (0.0, 0.15),
                "joint_armature": (0.0, 0.05),
            },
        )
        assert len(profile.variables) == 3

    def test_sample_and_apply_c_family(self):
        env = _FakeEnv()
        profile = build_family_variation_profile(
            family="C",
            profile_name="test",
            target="bowl",
            ranges={
                "object_com_offset_x": (0.01, 0.01),
                "object_inertia_diagonal": (2.0, 2.0),
            },
        )
        sampled = profile.sample_all(np.random.default_rng(0))
        profile.apply_all(env, sampled)
        np.testing.assert_allclose(env._env.sim.model.body_ipos[0, 0], 0.01)
        np.testing.assert_allclose(env._env.sim.model.body_inertia[0], [2.0, 2.0, 2.0])

    def test_sample_and_apply_p_family(self):
        env = _FakeEnv()
        profile = build_family_variation_profile(
            family="P",
            profile_name="test",
            target="finger",
            ranges={
                "joint_damping": (0.5, 0.5),
                "joint_frictionloss": (0.3, 0.3),
            },
        )
        sampled = profile.sample_all(np.random.default_rng(0))
        profile.apply_all(env, sampled)
        assert env._env.sim.model.dof_damping[0] == 0.5
        assert env._env.sim.model.dof_frictionloss[0] == 0.3

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark family"):
            build_family_variation_profile("X", "test", "bowl", {})

    def test_partial_ranges_selects_subset(self):
        profile = build_family_variation_profile(
            family="C",
            profile_name="partial",
            target="bowl",
            ranges={"object_inertia_diagonal": (1.0, 2.0)},
        )
        # Only one variable enabled (out of 3 registered for C)
        assert len(profile.variables) == 1

    def test_registry_has_all_families(self):
        assert set(FAMILY_VARIABLE_REGISTRY.keys()) == {"F", "M", "C", "P"}
