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

from lerobot.runtime.bowl_attempt_analysis import BowlAttemptAnalysisConfig, BowlAttemptAnalyzer


def _make_observation(*, eef_pos: tuple[float, float, float]) -> dict[str, object]:
    return {
        "robot_state": {
            "eef": {
                "pos": np.asarray(eef_pos, dtype=np.float32),
            }
        }
    }


class _FakeSimModel:
    nbody = 2

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return ["bowl_body", "table_body"][body_id]


class _FakeSimData:
    def __init__(self, env: "_FakeEnv"):
        self._env = env

    @property
    def body_xpos(self) -> np.ndarray:
        table_position = np.zeros(3, dtype=np.float32)
        return np.asarray([self._env.bowl_position, table_position], dtype=np.float32)


class _FakeSim:
    def __init__(self, env: "_FakeEnv"):
        self.model = _FakeSimModel()
        self.data = _FakeSimData(env)


class _FakeEnv:
    def __init__(self):
        self.bowl_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.sim = _FakeSim(self)


def test_bowl_attempt_analyzer_tracks_fail_then_recover_sequence():
    config = BowlAttemptAnalysisConfig(
        settle_grace_steps=1,
        far_grace_steps=1,
    )
    analyzer = BowlAttemptAnalyzer(config)
    env = _FakeEnv()

    analyzer.reset(env, _make_observation(eef_pos=(0.25, 0.0, 0.0)))

    env.bowl_position = np.array([0.03, 0.0, 0.02], dtype=np.float32)
    analyzer.update(
        env,
        _make_observation(eef_pos=(0.015, 0.0, 0.022)),
        step_id=1,
        timestamp=0.1,
        success=False,
    )

    env.bowl_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    analyzer.update(
        env,
        _make_observation(eef_pos=(0.22, 0.0, 0.0)),
        step_id=2,
        timestamp=0.2,
        success=False,
    )

    env.bowl_position = np.array([0.04, 0.0, 0.02], dtype=np.float32)
    analyzer.update(
        env,
        _make_observation(eef_pos=(0.012, 0.0, 0.02)),
        step_id=3,
        timestamp=0.3,
        success=False,
    )

    env.bowl_position = np.array([0.08, 0.0, 0.03], dtype=np.float32)
    analyzer.update(
        env,
        _make_observation(eef_pos=(0.01, 0.0, 0.03)),
        step_id=4,
        timestamp=0.4,
        success=True,
    )

    metrics, metadata = analyzer.finalize(step_id=4, timestamp=0.4, success=True)

    assert metrics["attempt_count"] == 2.0
    assert metrics["failed_attempt_count"] == 1.0
    assert metrics["successful_attempt_count"] == 1.0
    assert np.isclose(metrics["first_attempt_to_success_s"], 0.3)
    assert np.isclose(metrics["first_failure_to_success_s"], 0.2)
    assert np.isclose(metrics["mean_attempt_duration_s"], 0.1)
    assert np.isclose(metrics["max_attempt_duration_s"], 0.1)
    assert metadata["enabled"] is True
    assert metadata["signal_available"] is True
    assert metadata["recovered_after_failure"] is True
    assert metadata["object_body_name"] == "bowl_body"
    assert [event["outcome"] for event in metadata["events"]] == ["failed", "succeeded"]
