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
from lerobot.runtime.contracts import ActionCommand, PolicyResponse
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
        return {"actions": np.full((2, 7), 0.25, dtype=np.float32)}

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeSingleEnv:
    metadata = {"render_fps": 30}

    def __init__(self):
        self.task = "pick_up_the_bowl"
        self.task_description = "pick up the bowl"
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
    def __init__(self):
        self.envs = [_FakeSingleEnv()]
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
    monkeypatch.setattr(smoke, "WebsocketOpenPIJaxClient", _FakePolicyClient)
    monkeypatch.setattr(smoke, "make_single_task_vec_env", lambda env_cfg: ("libero_object", 3, _FakeVecEnv()))

    cfg = smoke.OpenPIBowlSmokeConfig(
        env=LiberoEnv(task="libero_object", task_ids=[3], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(n_episodes=1, max_steps=5, output_dir=str(tmp_path)),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert Path(tmp_path / "summary.json").exists()
    assert Path(summary["episodes"][0]["trace_path"]).exists()
