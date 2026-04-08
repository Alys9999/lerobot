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

import json
from pathlib import Path

import numpy as np

from lerobot.adapters.openpi_jax import OpenPIJaxLiberoSpec
from lerobot.envs.configs import AlohaEnv, LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.runtime.contracts import ObservationPacket, PolicyRequest, RobotSpec, RuntimeSpec, TaskSpec
from lerobot.adapters.openpi_jax.output_codec_libero import OpenPIJaxLiberoOutputCodec
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


def _make_observation_batch_with_eef(*, eef_pos: tuple[float, float, float]) -> dict[str, object]:
    observation = _make_observation_batch()
    observation["robot_state"]["eef"]["pos"] = np.asarray([eef_pos], dtype=np.float32)
    return observation


def _make_aloha_observation_batch() -> dict[str, object]:
    return {
        "pixels": {
            "top": np.full((1, 8, 8, 3), 64, dtype=np.uint8),
        },
        "agent_pos": np.zeros((1, 14), dtype=np.float32),
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


class _FakeAttemptSimModel:
    nbody = 2

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return ["bowl_body", "table_body"][body_id]


class _FakeAttemptSimData:
    def __init__(self, env: "_FakeAttemptSingleEnv"):
        self._env = env

    @property
    def body_xpos(self) -> np.ndarray:
        bowl_position = self._env.current_state["bowl_position"]
        table_position = np.zeros(3, dtype=np.float32)
        return np.asarray([bowl_position, table_position], dtype=np.float32)


class _FakeAttemptSim:
    def __init__(self, env: "_FakeAttemptSingleEnv"):
        self.model = _FakeAttemptSimModel()
        self.data = _FakeAttemptSimData(env)


class _FakeAttemptSingleEnv:
    metadata = {"render_fps": 30}

    def __init__(self, task_id: int = 0):
        self.task = f"pick_up_the_bowl_{task_id}"
        self.task_description = f"pick up the bowl task {task_id}"
        self.task_id = task_id
        self.last_variation_sample = {}
        self._profile = None
        self._rng = np.random.default_rng(0)
        self._step = 0
        self._states = [
            {
                "eef_pos": (0.25, 0.0, 0.0),
                "bowl_position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "done": False,
                "success": False,
            },
            {
                "eef_pos": (0.015, 0.0, 0.022),
                "bowl_position": np.array([0.03, 0.0, 0.02], dtype=np.float32),
                "done": False,
                "success": False,
            },
            {
                "eef_pos": (0.22, 0.0, 0.0),
                "bowl_position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "done": False,
                "success": False,
            },
            {
                "eef_pos": (0.012, 0.0, 0.02),
                "bowl_position": np.array([0.04, 0.0, 0.02], dtype=np.float32),
                "done": False,
                "success": False,
            },
            {
                "eef_pos": (0.01, 0.0, 0.03),
                "bowl_position": np.array([0.08, 0.0, 0.03], dtype=np.float32),
                "done": True,
                "success": True,
            },
        ]
        self.sim = _FakeAttemptSim(self)

    @property
    def current_state(self) -> dict[str, object]:
        return self._states[self._step]

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
        return _make_observation_batch_with_eef(eef_pos=self.current_state["eef_pos"]), {
            "variation": dict(self.last_variation_sample)
        }

    def render(self):
        return np.full((8, 8, 3), 90, dtype=np.uint8)

    def step(self, _action):
        self._step = min(self._step + 1, len(self._states) - 1)
        done = bool(self.current_state["done"])
        info = {
            "is_success": np.array([self.current_state["success"]]),
            "variation": np.array([dict(self.last_variation_sample)], dtype=object),
        }
        if done:
            info["final_info"] = {"is_success": np.array([self.current_state["success"]])}
        return (
            _make_observation_batch_with_eef(eef_pos=self.current_state["eef_pos"]),
            np.array([1.0], dtype=np.float32),
            np.array([done]),
            np.array([False]),
            info,
        )


class _FakeAttemptVecEnv:
    def __init__(self, task_id: int = 0):
        self.envs = [_FakeAttemptSingleEnv(task_id=task_id)]
        self.num_envs = 1

    def reset(self, seed=None):
        return self.envs[0].reset(seed=seed)

    def step(self, action):
        return self.envs[0].step(action)

    def call(self, attr_name):
        return [getattr(self.envs[0], attr_name)]

    def close(self) -> None:
        return None


class _FakeAlohaSingleEnv:
    metadata = {"render_fps": 50}

    def __init__(self):
        self.task = "AlohaTransferCube-v0"
        self.last_variation_sample = {}
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return _make_aloha_observation_batch(), {}

    def render(self):
        return np.full((8, 8, 3), 200, dtype=np.uint8)

    def step(self, _action):
        self._step += 1
        done = self._step >= 2
        info = {"is_success": np.array([done])}
        if done:
            info["final_info"] = {"is_success": np.array([True])}
        return (
            _make_aloha_observation_batch(),
            np.array([1.0], dtype=np.float32),
            np.array([done]),
            np.array([False]),
            info,
        )


class _FakeAlohaVecEnv:
    def __init__(self):
        self.envs = [_FakeAlohaSingleEnv()]
        self.num_envs = 1

    def reset(self, seed=None):
        return self.envs[0].reset(seed=seed)

    def step(self, action):
        return self.envs[0].step(action)

    def call(self, attr_name):
        return [getattr(self.envs[0], attr_name)]

    def close(self) -> None:
        return None


class _RecordingPolicyClient:
    def __init__(self, _cfg):
        self.server_metadata = {"server": "recording"}
        self.requests: list[dict[str, object]] = []

    def infer(self, observation):
        self.requests.append(observation)
        actions = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
        return {"actions": actions}

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _RecordingAlohaPolicyClient:
    def __init__(self, _cfg):
        self.server_metadata = {"server": "recording_aloha"}
        self.requests: list[dict[str, object]] = []

    def infer(self, observation):
        self.requests.append(observation)
        actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
        return {"actions": actions}

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _RecordingDroidPolicyClient:
    def __init__(self, _cfg):
        self.server_metadata = {"server": "recording_droid"}
        self.requests: list[dict[str, object]] = []

    def infer(self, observation):
        self.requests.append(observation)
        actions = np.arange(15 * 8, dtype=np.float32).reshape(15, 8)
        return {"actions": actions}

    def reset(self) -> None:
        return None

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


def test_libero_env_gym_kwargs_include_camera_mapping():
    cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        camera_name_mapping={
            "agentview_image": "base_0_rgb",
            "robot0_eye_in_hand_image": "left_wrist_0_rgb",
        },
    )

    assert cfg.gym_kwargs["camera_name_mapping"] == {
        "agentview_image": "base_0_rgb",
        "robot0_eye_in_hand_image": "left_wrist_0_rgb",
    }


def test_ensure_smoke_runtime_ready_sets_headless_mujoco_defaults_for_aloha(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    smoke.ensure_smoke_runtime_ready(AlohaEnv(task="AlohaTransferCube-v0"))

    assert smoke.os.environ["MUJOCO_GL"] == "egl"
    assert smoke.os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_processed_observation_and_codec_honor_configurable_contract():
    spec = OpenPIJaxLiberoSpec(
        model_id="custom_pi05_contract",
        robot_id="widowx_250",
        embodiment_id="widowx",
        backend_id="mujoco",
        packet_image_keys={
            "base": "observation.images.image",
            "left": "observation.images.image2",
            "right": "observation.images.image2",
        },
        remote_image_keys={
            "base": "observation/images/base_0_rgb",
            "left": "observation/images/left_wrist_0_rgb",
            "right": "observation/images/right_wrist_0_rgb",
        },
        state_packet_key="libero_state_32d",
        state_remote_key="observation/state_32d",
        state_dim=32,
        action_dim=7,
        action_horizon=10,
        server_action_dim=32,
        output_action_indices=[0, 1, 2, 3, 4, 5, 6],
    )
    observation_cfg = smoke.SmokeObservationAdapterConfig(
        image_flip=False,
        state_components=["eef_pos", "eef_quat", "gripper_qpos", "joints_pos", "joints_vel", "eef_mat"],
    )
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0], autoreset_on_done=False)

    preprocessed = preprocess_observation(_make_observation_batch())
    processed = smoke.make_smoke_env_preprocessor(
        env_cfg,
        spec=spec,
        observation_cfg=observation_cfg,
    )(preprocessed)
    packet = smoke.processed_observation_to_packet(
        processed,
        suite_name="libero_spatial",
        task_id=0,
        episode_index=0,
        step_id=0,
        task_text="pick up the bowl",
        spec=spec,
    )

    assert packet.robot_id == "widowx_250"
    assert packet.embodiment_id == "widowx"
    assert packet.backend_id == "mujoco"
    assert sorted(packet.images) == ["base", "left", "right"]
    assert packet.robot_state["libero_state_32d"].shape == (32,)

    codec = OpenPIJaxLiberoOutputCodec(spec)
    req = PolicyRequest(
        observation=ObservationPacket(
            timestamp=0.0,
            episode_id="ep0",
            step_id=0,
            images=packet.images,
            robot_state=packet.robot_state,
            task_text="pick up the bowl",
            task_id="0",
            robot_id=packet.robot_id,
            embodiment_id=packet.embodiment_id,
            backend_id=packet.backend_id,
        ),
        robot_spec=RobotSpec(robot_id="widowx_250", action_dim=7, camera_keys=("base", "left", "right")),
        task_spec=TaskSpec(task_suite="libero_spatial", task_id="0", prompt="pick up the bowl"),
        runtime_spec=RuntimeSpec(control_dt=1 / 30, max_steps=10),
    )
    raw_actions = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
    decoded = codec.decode({"actions": raw_actions}, req)

    assert decoded.values.shape == (10, 7)
    np.testing.assert_array_equal(decoded.values, raw_actions[:, :7])


def test_run_bowl_smoke_uses_configurable_policy_contract(tmp_path, monkeypatch):
    client = _RecordingPolicyClient(None)
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: client,
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("libero_spatial", env_cfg.task_ids[0], _FakeVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        policy_spec=OpenPIJaxLiberoSpec(
            model_id="custom_pi05_contract",
            robot_id="widowx_250",
            embodiment_id="widowx",
            backend_id="mujoco",
            packet_image_keys={
                "base": "observation.images.image",
                "left": "observation.images.image2",
                "right": "observation.images.image2",
            },
            remote_image_keys={
                "base": "observation/images/base_0_rgb",
                "left": "observation/images/left_wrist_0_rgb",
                "right": "observation/images/right_wrist_0_rgb",
            },
            state_packet_key="libero_state_32d",
            state_remote_key="observation/state_32d",
            state_dim=32,
            action_dim=7,
            action_horizon=10,
            server_action_dim=32,
            output_action_indices=[0, 1, 2, 3, 4, 5, 6],
        ),
        observation=smoke.SmokeObservationAdapterConfig(
            image_flip=False,
            state_components=["eef_pos", "eef_quat", "gripper_qpos", "joints_pos", "joints_vel", "eef_mat"],
        ),
        env=LiberoEnv(task="libero_spatial", task_ids=[0], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert client.requests
    first_request = client.requests[0]
    assert sorted(first_request) == [
        "observation/images/base_0_rgb",
        "observation/images/left_wrist_0_rgb",
        "observation/images/right_wrist_0_rgb",
        "observation/state_32d",
        "prompt",
    ]
    assert np.asarray(first_request["observation/state_32d"]).shape == (32,)


def test_run_bowl_smoke_supports_aloha_arm_contract(tmp_path, monkeypatch):
    client = _RecordingAlohaPolicyClient(None)
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: client,
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("aloha", 0, _FakeAlohaVecEnv()),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        policy_spec=OpenPIJaxLiberoSpec(
            model_id="openpi_jax_pi05_aloha",
            env_type="aloha",
            robot_id="aloha",
            embodiment_id="aloha",
            backend_id="gym_aloha",
            packet_image_keys={"cam_high": "observation.images.top"},
            remote_image_keys={"cam_high": "cam_high"},
            state_observation_key="observation.state",
            state_packet_key="state",
            state_remote_key="state",
            state_dim=14,
            remote_image_container_key="images",
            remote_image_layout="chw",
            prompt_required=False,
            action_space="env_native_14d",
            action_dim=14,
            action_horizon=50,
        ),
        env=AlohaEnv(task="AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert client.requests
    first_request = client.requests[0]
    assert sorted(first_request) == ["images", "prompt", "state"]
    assert sorted(first_request["images"]) == ["cam_high"]
    assert np.asarray(first_request["images"]["cam_high"]).shape == (3, 8, 8)
    assert np.asarray(first_request["state"]).shape == (14,)


def test_run_bowl_smoke_supports_aloha_policy_on_libero_bench(tmp_path, monkeypatch):
    client = _RecordingAlohaPolicyClient(None)
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: client,
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("libero_spatial", env_cfg.task_ids[0], _FakeVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        policy_spec=OpenPIJaxLiberoSpec(
            model_id="openpi_jax_pi05_aloha",
            env_type="aloha",
            robot_id="aloha",
            embodiment_id="aloha",
            backend_id="gym_aloha",
            packet_image_keys={"cam_high": "observation.images.image"},
            remote_image_keys={"cam_high": "cam_high"},
            state_observation_key="observation.state",
            state_packet_key="state",
            state_remote_key="state",
            state_dim=14,
            remote_image_container_key="images",
            remote_image_layout="chw",
            prompt_required=False,
            action_space="env_native_7d",
            action_dim=7,
            action_horizon=50,
            server_action_dim=14,
            output_action_indices=[0, 1, 2, 3, 4, 5, 6],
        ),
        observation=smoke.SmokeObservationAdapterConfig(
            image_flip=True,
            state_components=["eef_pos", "joints_pos", "gripper_qpos", "gripper_qvel"],
        ),
        env=LiberoEnv(task="libero_spatial", task_ids=[0], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert client.requests
    first_request = client.requests[0]
    assert sorted(first_request) == ["images", "prompt", "state"]
    assert sorted(first_request["images"]) == ["cam_high"]
    assert np.asarray(first_request["images"]["cam_high"]).shape == (3, 8, 8)
    assert np.asarray(first_request["state"]).shape == (14,)


def test_run_bowl_smoke_supports_droid_policy_contract_on_libero_bench(tmp_path, monkeypatch):
    client = _RecordingDroidPolicyClient(None)
    monkeypatch.setattr(
        smoke,
        "make_openpi_jax_client",
        lambda _cfg, *, action_dim, action_horizon: client,
    )
    monkeypatch.setattr(
        smoke,
        "make_single_task_vec_env",
        lambda env_cfg: ("libero_spatial", env_cfg.task_ids[0], _FakeVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        policy_spec=OpenPIJaxLiberoSpec(
            model_id="openpi_jax_pi05_droid",
            env_type="droid",
            robot_id="franka_panda",
            embodiment_id="droid",
            backend_id="droid",
            packet_image_keys={
                "base": "observation.images.image",
                "left_wrist": "observation.images.image2",
            },
            remote_image_keys={
                "base": "observation/exterior_image_1_left",
                "left_wrist": "observation/wrist_image_left",
            },
            state_observation_key="observation.state",
            state_packet_key="droid_state_8d",
            state_remote_key=None,
            state_remote_keys={
                "observation/joint_position": [0, 1, 2, 3, 4, 5, 6],
                "observation/gripper_position": [7],
            },
            state_dim=8,
            action_space="env_native_7d",
            action_dim=7,
            action_horizon=15,
            server_action_dim=8,
            output_action_indices=[0, 1, 2, 3, 4, 5, 6],
        ),
        observation=smoke.SmokeObservationAdapterConfig(
            image_flip=True,
            state_components=["joints_pos", "gripper_position"],
        ),
        env=LiberoEnv(task="libero_spatial", task_ids=[0], autoreset_on_done=False),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    assert summary["success_rate"] == 1.0
    assert client.requests
    first_request = client.requests[0]
    assert sorted(first_request) == [
        "observation/exterior_image_1_left",
        "observation/gripper_position",
        "observation/joint_position",
        "observation/wrist_image_left",
        "prompt",
    ]
    assert np.asarray(first_request["observation/joint_position"]).shape == (7,)
    assert np.asarray(first_request["observation/gripper_position"]).shape == (1,)


def test_run_bowl_smoke_records_attempt_analysis_metrics_and_trace(tmp_path, monkeypatch):
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
        lambda env_cfg: ("libero_spatial", env_cfg.task_ids[0], _FakeAttemptVecEnv(task_id=env_cfg.task_ids[0])),
    )

    cfg = smoke.OpenPIBowlSmokeConfig(
        env=LiberoEnv(task="libero_spatial", task_ids=[0], fps=10, autoreset_on_done=False),
        attempt_analysis=smoke.BowlAttemptAnalysisConfig(
            settle_grace_steps=1,
            far_grace_steps=1,
        ),
        runtime=smoke.SmokeRuntimeConfig(
            n_episodes=1,
            max_steps=6,
            output_dir=str(tmp_path),
            write_video=False,
        ),
    )

    summary = smoke.run_bowl_smoke(cfg)

    episode_metrics = summary["episodes"][0]["metrics"]
    assert episode_metrics["attempt_count"] == 2.0
    assert episode_metrics["failed_attempt_count"] == 1.0
    assert episode_metrics["successful_attempt_count"] == 1.0
    assert np.isclose(episode_metrics["first_attempt_to_success_s"], 0.3)
    assert np.isclose(episode_metrics["first_failure_to_success_s"], 0.2)

    trace_path = Path(summary["episodes"][0]["trace_path"])
    trace_summary = json.loads(trace_path.read_text(encoding="utf-8"))
    attempt_analysis = trace_summary["metadata"]["attempt_analysis"]
    assert attempt_analysis["enabled"] is True
    assert attempt_analysis["signal_available"] is True
    assert attempt_analysis["recovered_after_failure"] is True
    assert [event["outcome"] for event in attempt_analysis["events"]] == ["failed", "succeeded"]
