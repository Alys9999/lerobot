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

from pathlib import Path

import numpy as np

from lerobot.envs.configs import LiberoEnv
from lerobot.scripts import lerobot_openpi_bowl_smoke as smoke


def _make_robot_state_batch() -> dict[str, dict[str, np.ndarray]]:
    return {
        "eef": {
            "pos": np.zeros((1, 3), dtype=np.float32),
            "quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "mat": np.repeat(np.eye(3, dtype=np.float32)[None, ...], 1, axis=0),
        },
        "gripper": {
            "qpos": np.zeros((1, 2), dtype=np.float32),
            "qvel": np.zeros((1, 2), dtype=np.float32),
        },
        "joints": {
            "pos": np.zeros((1, 7), dtype=np.float32),
            "vel": np.zeros((1, 7), dtype=np.float32),
        },
    }


def _make_observation_batch() -> dict[str, object]:
    return {
        "pixels": {
            "image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "image2": np.ones((1, 8, 8, 3), dtype=np.uint8),
        },
        "robot_state": _make_robot_state_batch(),
    }


class _FakePolicyClient:
    def __init__(self, _cfg):
        self.server_metadata = {"server": "fake"}

    def infer(self, _observation):
        return {"actions": np.full((10, 7), 0.25, dtype=np.float32)}

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeSingleEnv:
    metadata = {"render_fps": 30}

    def __init__(self, task_id: int = 0):
        self.task = f"pick_up_the_bowl_{task_id}"
        self.task_description = f"pick up the bowl task {task_id}"
        self.task_id = task_id
        self.last_variation_sample = {}
        self._profile = None
        self._rng = np.random.default_rng(0)
        self._step = 0

    def set_variation_profile(self, profile, seed=None):
        self._profile = profile
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        self._step = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._profile is not None:
            self.last_variation_sample = self._profile.sample_all(self._rng)
        return _make_observation_batch(), {"variation": dict(self.last_variation_sample)}

    def render(self):
        return np.full((8, 8, 3), 127, dtype=np.uint8)

    def step(self, _action):
        self._step += 1
        done = self._step >= 2
        info = {
            "is_success": np.array([done]),
            "variation": np.array([dict(self.last_variation_sample)], dtype=object),
        }
        if done:
            info["final_info"] = {"is_success": np.array([True])}
        return _make_observation_batch(), np.array([1.0], dtype=np.float32), np.array([done]), np.array([False]), info


class _FakeVecEnv:
    def __init__(self, task_id: int = 0):
        self.envs = [_FakeSingleEnv(task_id=task_id)]
        self.num_envs = 1

    def reset(self, seed=None):
        return self.envs[0].reset(seed=seed)

    def step(self, action):
        return self.envs[0].step(action)

    def call(self, attr_name):
        return [getattr(self.envs[0], attr_name)]

    def close(self) -> None:
        return None


def test_run_bowl_smoke_writes_summary_and_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: _FakePolicyClient(
            (_cfg, action_dim, action_horizon)
        ),
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("libero_object", env_cfg.task_ids[0], _FakeVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        env=LiberoEnv(task="libero_object", task_ids=[3], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(n_episodes=1, max_steps=5, output_dir=str(tmp_path)),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert summary["num_tasks"] == 1
    assert len(summary["tasks"]) == 1
    assert Path(tmp_path / "summary.json").exists()
    assert Path(summary["episodes"][0]["trace_path"]).exists()
    assert Path(summary["episodes"][0]["video_path"]).exists()


def test_run_bowl_smoke_supports_multiple_tasks_with_variation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: _FakePolicyClient(
            (_cfg, action_dim, action_horizon)
        ),
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("libero_spatial", env_cfg.task_ids[0], _FakeVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        env=LiberoEnv(task="libero_spatial", task_ids=[0, 1], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["num_tasks"] == 2
    assert len(summary["tasks"]) == 2
    assert len(summary["episodes"]) == 2
    assert summary["tasks"][0]["task_id"] == 0
    assert summary["tasks"][1]["task_id"] == 1
    assert summary["episodes"][0]["variation"]
    assert summary["episodes"][1]["variation"]
    assert Path(summary["episodes"][0]["trace_path"]).exists()
    assert Path(summary["episodes"][1]["trace_path"]).exists()
