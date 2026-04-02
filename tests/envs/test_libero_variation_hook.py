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

import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import yaml

libero_spec = find_spec("libero.libero")
if libero_spec is None or libero_spec.origin is None:
    pytest.skip("libero not installed", allow_module_level=True)

_libero_package_root = Path(libero_spec.origin).resolve().parent
_libero_config_root = Path(tempfile.gettempdir()) / "lerobot_libero_test_config"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(_libero_config_root))
_libero_config_root.mkdir(parents=True, exist_ok=True)
_libero_config_path = _libero_config_root / "config.yaml"
if not _libero_config_path.exists():
    _libero_config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(_libero_package_root),
                "bddl_files": str(_libero_package_root / "bddl_files"),
                "init_states": str(_libero_package_root / "init_files"),
                "datasets": str((_libero_package_root / "../datasets").resolve()),
                "assets": str(_libero_package_root / "assets"),
            }
        ),
        encoding="utf-8",
    )

from lerobot.envs.libero import LiberoEnv
from lerobot.runtime.variation import (
    VariationConfig,
    VariationVariableConfig,
    VariationVariablesConfig,
    build_variation_profile,
)


def _make_raw_obs() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((16, 16, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float64),
        "robot0_joint_pos": np.zeros(7, dtype=np.float64),
        "robot0_joint_vel": np.zeros(7, dtype=np.float64),
    }


class _FakeTask:
    name = "pick_up_the_bowl"
    language = "pick up the bowl"
    problem_folder = "fake_problem"
    bddl_file = "fake_problem.bddl"
    init_states_file = "fake_problem.pt"


class _FakeSuite:
    def __init__(self):
        self.tasks = [_FakeTask()]

    def get_task(self, task_id: int) -> _FakeTask:
        return self.tasks[task_id]


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

    def forward(self) -> None:
        return None


class _FakeController:
    def __init__(self):
        self.ee_ori_mat = np.eye(3, dtype=np.float64)
        self.use_delta = True


class _FakeRobot:
    def __init__(self):
        self.controller = _FakeController()


class _FakeOffscreenEnv:
    def __init__(self):
        self.sim = _FakeSim()
        self.robots = [_FakeRobot()]
        self.seed_value = None

    def seed(self, seed: int | None) -> None:
        self.seed_value = seed

    def reset(self) -> dict[str, np.ndarray]:
        return _make_raw_obs()

    def set_init_state(self, _state: object) -> dict[str, np.ndarray]:
        return _make_raw_obs()

    def step(self, _action: object):
        return _make_raw_obs(), 0.0, False, {}

    def check_success(self) -> bool:
        return False

    def close(self) -> None:
        return None


def _make_fake_env(self, task_suite, task_id):
    task = task_suite.get_task(task_id)
    self.task = task.name
    self.task_description = task.language
    return _FakeOffscreenEnv()


def test_libero_env_variation_hook_applies_before_episode(monkeypatch):
    monkeypatch.setattr(LiberoEnv, "_make_envs_task", _make_fake_env)

    env = LiberoEnv(
        task_suite=_FakeSuite(),
        task_id=0,
        task_suite_name="libero_object",
        obs_type="pixels_agent_pos",
        init_states=False,
        num_steps_wait=0,
        autoreset_on_done=False,
    )
    profile = build_variation_profile(
        VariationConfig(
            variables=VariationVariablesConfig(
                object_friction=VariationVariableConfig(enabled=True, target="bowl", range=(0.2, 0.2)),
                finger_friction=VariationVariableConfig(enabled=True, target="gripper_fingers", range=(0.6, 0.6)),
                object_weight=VariationVariableConfig(enabled=True, target="bowl", range=(0.3, 0.3)),
            )
        )
    )
    env.set_variation_profile(profile, seed=123)

    _observation, info = env.reset(seed=123)

    assert info["variation"] == {
        "object_friction": 0.2,
        "finger_friction": 0.6,
        "object_weight": 0.3,
    }
    assert env.last_variation_sample == info["variation"]
    np.testing.assert_allclose(env._env.sim.model.geom_friction[0], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(env._env.sim.model.geom_friction[1], [0.6, 0.6, 0.6])
    assert np.isclose(float(env._env.sim.model.body_mass[0]), 0.3)

    _observation, _reward, terminated, _truncated, step_info = env.step(np.zeros(7, dtype=np.float32))

    assert terminated is False
    assert step_info["variation"] == info["variation"]
