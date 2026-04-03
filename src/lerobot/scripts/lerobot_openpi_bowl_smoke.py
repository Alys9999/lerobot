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

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.adapters.openpi_jax import (
    OpenPIJaxAdapter,
    OpenPIJaxClientConfig,
    OpenPIJaxLiberoSpec,
    make_openpi_jax_client,
)
from lerobot.configs import parser
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.envs.libero_bootstrap import ensure_libero_runtime_ready
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, env_to_policy_features, preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.runtime.contracts import (
    ActionCommand,
    EpisodeTrace,
    ObservationPacket,
    PolicyRequest,
    RobotSpec,
    RuntimeSpec,
    TaskSpec,
)
from lerobot.runtime.trace import write_episode_trace
from lerobot.runtime.variation import VariationConfig, build_variation_profile
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging
from lerobot.utils.constants import ACTION

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BowlTaskResolverConfig:
    task_suite: str = "libero_object"
    name_contains: str = "bowl"
    max_matches: int | None = 1


@dataclass(slots=True)
class SmokeRuntimeConfig:
    n_episodes: int = 3
    max_steps: int = 300
    seed: int = 123
    write_trace: bool = True
    write_video: bool = True
    video_fps: int | None = None
    output_dir: str = "outputs/openpi_bowl_smoke"
    fail_fast: bool = True


@dataclass(slots=True)
class SmokeObservationAdapterConfig:
    image_flip: bool = True
    state_components: list[str] = field(
        default_factory=lambda: ["eef_pos", "eef_axis_angle", "gripper_qpos"]
    )


@dataclass(slots=True)
class OpenPIBowlSmokeConfig:
    policy: OpenPIJaxClientConfig = field(default_factory=OpenPIJaxClientConfig)
    policy_spec: OpenPIJaxLiberoSpec = field(default_factory=OpenPIJaxLiberoSpec)
    env: EnvConfig = field(
        default_factory=lambda: LiberoEnv(
            task="libero_object",
            task_ids=None,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            autoreset_on_done=False,
        )
    )
    observation: SmokeObservationAdapterConfig = field(default_factory=SmokeObservationAdapterConfig)
    bowl_task_resolver: BowlTaskResolverConfig = field(default_factory=BowlTaskResolverConfig)
    variation: VariationConfig = field(default_factory=VariationConfig)
    runtime: SmokeRuntimeConfig = field(default_factory=SmokeRuntimeConfig)


@dataclass(slots=True)
class SmokeEpisodeArtifacts:
    trace: EpisodeTrace
    video_frames: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ResolvedSmokeTask:
    env_cfg: EnvConfig
    suite_name: str
    task_id: int
    task_name: str | None = None
    task_prompt: str | None = None


def _rewrite_legacy_config_flag() -> None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith("--config="):
            sys.argv[index] = "--config_path=" + arg[len("--config=") :]


def _to_numpy_unbatched(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _to_uint8_hwc_image(value: Any) -> np.ndarray:
    array = _to_numpy_unbatched(value)
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after unbatching, got {array.shape}.")
    if array.shape[0] in (1, 3, 4) and array.shape[0] < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f":
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _to_scalar_or_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_scalar_or_dict(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] == 1:
            return _to_scalar_or_dict(value[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_scalar_or_dict(value[0])
    return value


def _extract_success(info: dict[str, Any]) -> bool:
    normalized = _to_scalar_or_dict(info)
    if not isinstance(normalized, dict):
        return False
    final_info = normalized.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return bool(final_info["is_success"])
    if "is_success" in normalized:
        return bool(normalized["is_success"])
    return False


def _normalize_task_ids(task_ids: list[int] | None) -> list[int]:
    if task_ids is None:
        return []

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_task_id in task_ids:
        task_id = int(raw_task_id)
        if task_id in seen:
            continue
        normalized.append(task_id)
        seen.add(task_id)
    return normalized


def resolve_smoke_tasks(env_cfg: EnvConfig, resolver_cfg: BowlTaskResolverConfig) -> list[ResolvedSmokeTask]:
    if env_cfg.type != "libero":
        return [
            ResolvedSmokeTask(
                env_cfg=env_cfg,
                suite_name=env_cfg.type,
                task_id=0,
                task_name=env_cfg.task,
                task_prompt=env_cfg.task,
            )
        ]

    normalized_task_ids = _normalize_task_ids(env_cfg.task_ids)
    if normalized_task_ids:
        return [
            ResolvedSmokeTask(
                env_cfg=replace(env_cfg, task=env_cfg.task, task_ids=[task_id]),
                suite_name=env_cfg.task,
                task_id=task_id,
            )
            for task_id in normalized_task_ids
        ]

    try:
        ensure_libero_runtime_ready()
        from libero.libero import benchmark
    except ImportError as exc:
        raise ImportError(
            "LIBERO is required to resolve bowl tasks dynamically. "
            "Install the `lerobot[libero]` extra or pass env.task_ids explicitly."
        ) from exc

    suite_name = resolver_cfg.task_suite or env_cfg.task
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite '{suite_name}'.")

    suite = benchmark_dict[suite_name]()
    needle = resolver_cfg.name_contains.lower().strip()
    matched_tasks: list[ResolvedSmokeTask] = []
    for task_id, task in enumerate(suite.tasks):
        task_name = getattr(task, "name", "")
        task_prompt = getattr(task, "language", "")
        if needle in task_name.lower() or needle in task_prompt.lower():
            matched_tasks.append(
                ResolvedSmokeTask(
                    env_cfg=replace(env_cfg, task=suite_name, task_ids=[task_id]),
                    suite_name=suite_name,
                    task_id=task_id,
                    task_name=task_name,
                    task_prompt=task_prompt,
                )
            )
            if resolver_cfg.max_matches is not None and len(matched_tasks) >= resolver_cfg.max_matches:
                break

    if not matched_tasks:
        raise ValueError(f"Could not resolve a bowl task in suite '{suite_name}' with token '{needle}'.")
    return matched_tasks


def ensure_smoke_runtime_ready(env_cfg: EnvConfig) -> None:
    if env_cfg.type == "libero":
        ensure_libero_runtime_ready()
        return

    if env_cfg.type == "aloha" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def make_single_task_vec_env(env_cfg: LiberoEnv):
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    if len(envs) != 1:
        raise ValueError(f"Expected exactly one suite, got suites={list(envs.keys())}.")
    suite_name = next(iter(envs))
    task_map = envs[suite_name]
    if len(task_map) != 1:
        raise ValueError(f"Expected exactly one task, got task_ids={list(task_map.keys())}.")
    task_id = next(iter(task_map))
    return suite_name, task_id, task_map[task_id]


def make_smoke_env_preprocessor(
    env_cfg: EnvConfig,
    *,
    spec: OpenPIJaxLiberoSpec,
    observation_cfg: SmokeObservationAdapterConfig,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    if env_cfg.type == "libero":
        libero_step = LiberoProcessorStep(
            image_flip=observation_cfg.image_flip,
            state_components=observation_cfg.state_components,
            state_output_key=spec.state_observation_key,
        )
        if libero_step.state_dim != spec.state_dim:
            raise ValueError(
                "LIBERO smoke observation config does not match policy_spec.state_dim. "
                f"Got state_components={observation_cfg.state_components} -> {libero_step.state_dim} dims, "
                f"but policy_spec.state_dim={spec.state_dim}."
            )
        return PolicyProcessorPipeline(steps=[libero_step])

    return PolicyProcessorPipeline(steps=[])


def validate_smoke_contract(
    env_cfg: EnvConfig,
    *,
    spec: OpenPIJaxLiberoSpec,
) -> None:
    policy_features = env_to_policy_features(env_cfg)
    action_feature = policy_features.get(ACTION)
    if action_feature is None:
        raise ValueError(f"Env type '{env_cfg.type}' does not expose an action feature.")
    env_action_dim = int(action_feature.shape[0])
    if env_action_dim != spec.action_dim:
        raise ValueError(
            "policy_spec.action_dim does not match the environment action space. "
            f"Got env action dim {env_action_dim} and policy_spec.action_dim={spec.action_dim}."
        )

    missing_packet_sources = [
        source_key for source_key in spec.packet_image_keys.values() if source_key not in policy_features
    ]
    if missing_packet_sources:
        raise ValueError(
            "policy_spec.packet_image_keys references observation keys that the environment does not expose. "
            f"Missing keys: {missing_packet_sources}."
        )

    if env_cfg.type != "libero":
        state_feature = policy_features.get(spec.state_observation_key)
        if state_feature is None:
            raise ValueError(
                f"Env type '{env_cfg.type}' does not expose state key {spec.state_observation_key!r}."
            )
        env_state_dim = int(state_feature.shape[0])
        if env_state_dim != spec.state_dim:
            raise ValueError(
                "policy_spec.state_dim does not match the environment state dimension. "
                f"Got env state dim {env_state_dim} and policy_spec.state_dim={spec.state_dim}."
            )


def capture_smoke_frame(vec_env: Any) -> np.ndarray:
    if hasattr(vec_env, "envs") and len(vec_env.envs) == 1 and hasattr(vec_env.envs[0], "render"):
        frame = vec_env.envs[0].render()
    elif hasattr(vec_env, "render"):
        frame = vec_env.render()
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
            frame = frame[0]
    else:
        raise ValueError("Smoke env does not expose a render method that can be used for video capture.")

    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3:
        raise ValueError(f"Expected rendered frame with shape (H, W, C), got {frame.shape}.")
    return frame


def processed_observation_to_packet(
    processed_observation: dict[str, Any],
    *,
    suite_name: str,
    task_id: int,
    episode_index: int,
    step_id: int,
    task_text: str,
    prev_action: np.ndarray | None = None,
    spec: OpenPIJaxLiberoSpec | None = None,
) -> ObservationPacket:
    spec = spec or OpenPIJaxLiberoSpec()
    state = np.asarray(
        _to_numpy_unbatched(processed_observation[spec.state_observation_key]),
        dtype=np.float32,
    ).reshape(-1)
    images = {
        alias: _to_uint8_hwc_image(processed_observation[source_key])
        for alias, source_key in spec.packet_image_keys.items()
    }
    return ObservationPacket(
        timestamp=time.time(),
        episode_id=f"{suite_name}:{task_id}:episode_{episode_index}",
        step_id=step_id,
        images=images,
        robot_state={spec.state_packet_key: state},
        task_text=task_text,
        task_id=str(task_id),
        robot_id=spec.robot_id,
        embodiment_id=spec.embodiment_id,
        backend_id=spec.backend_id,
        prev_action=prev_action,
    )


def run_smoke_episode(
    vec_env: Any,
    adapter: Any,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    runtime_cfg: SmokeRuntimeConfig,
    *,
    suite_name: str,
    task_id: int,
    episode_index: int,
    task_name: str | None = None,
    task_prompt: str | None = None,
    control_dt: float = 1 / 30,
    spec: OpenPIJaxLiberoSpec | None = None,
    record_video: bool = False,
) -> SmokeEpisodeArtifacts:
    spec = spec or OpenPIJaxLiberoSpec()
    adapter.reset()
    seed = runtime_cfg.seed + episode_index
    observation, _info = vec_env.reset(seed=seed)
    video_frames = [capture_smoke_frame(vec_env)] if record_video else []
    current_task_name = task_name or getattr(vec_env.envs[0], "task", "") or ""
    current_task_prompt = task_prompt or getattr(vec_env.envs[0], "task_description", "") or current_task_name
    variation = dict(getattr(vec_env.envs[0], "last_variation_sample", {}))

    trace = EpisodeTrace(
        metadata={
            "suite": suite_name,
            "task_id": task_id,
            "task_name": current_task_name,
            "task_prompt": current_task_prompt,
            "variation": variation,
            "action_chunk_shapes": [],
        }
    )

    robot_spec = RobotSpec(
        robot_id=spec.robot_id,
        action_dim=spec.action_dim,
        camera_keys=tuple(spec.required_image_keys),
    )
    task_spec = TaskSpec(task_suite=suite_name, task_id=str(task_id), prompt=current_task_prompt)
    runtime_spec = RuntimeSpec(control_dt=control_dt, max_steps=runtime_cfg.max_steps)

    pending_actions: deque[np.ndarray] = deque()
    prev_action: np.ndarray | None = None
    latencies_ms: list[float] = []
    policy_calls = 0
    episode_return = 0.0

    for step_id in range(runtime_cfg.max_steps):
        policy_observation = preprocess_observation(observation)
        policy_observation = add_envs_task(vec_env, policy_observation)
        policy_observation = env_preprocessor(policy_observation)
        packet = processed_observation_to_packet(
            policy_observation,
            suite_name=suite_name,
            task_id=task_id,
            episode_index=episode_index,
            step_id=step_id,
            task_text=current_task_prompt,
            prev_action=prev_action,
            spec=spec,
        )
        trace.observations.append(packet)

        if not pending_actions:
            request = PolicyRequest(
                observation=packet,
                robot_spec=robot_spec,
                task_spec=task_spec,
                runtime_spec=runtime_spec,
            )
            response = adapter.infer(request)
            chunk = np.asarray(response.action.values, dtype=np.float32)
            for chunk_action in chunk:
                pending_actions.append(np.asarray(chunk_action, dtype=np.float32))
            latencies_ms.append(float(response.latency_ms))
            trace.actions.append(response.action)
            trace.metadata["action_chunk_shapes"].append(list(chunk.shape))
            policy_calls += 1

        action = pending_actions.popleft()
        prev_action = action.copy()
        observation, reward, terminated, truncated, info = vec_env.step(action.reshape(1, -1))
        if record_video:
            video_frames.append(capture_smoke_frame(vec_env))
        reward_value = float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
        done = bool(np.asarray(terminated | truncated).reshape(-1)[0])

        episode_return += reward_value
        trace.rewards.append(reward_value)
        trace.dones.append(done)
        trace.infos.append(_to_scalar_or_dict(info))

        if done:
            trace.success = _extract_success(info)
            break
    else:
        trace.success = False

    trace.metrics = {
        "steps": float(len(trace.rewards)),
        "episode_return": episode_return,
        "policy_calls": float(policy_calls),
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "max_latency_ms": float(np.max(latencies_ms)) if latencies_ms else 0.0,
    }
    return SmokeEpisodeArtifacts(trace=trace, video_frames=video_frames)


def run_bowl_smoke(cfg: OpenPIBowlSmokeConfig) -> dict[str, Any]:
    set_seed(cfg.runtime.seed)
    ensure_smoke_runtime_ready(cfg.env)
    validate_smoke_contract(cfg.env, spec=cfg.policy_spec)
    resolved_tasks = resolve_smoke_tasks(cfg.env, cfg.bowl_task_resolver)
    variation_profile = build_variation_profile(cfg.variation)
    output_dir = Path(cfg.runtime.output_dir)
    traces_dir = output_dir / "traces"
    videos_dir = output_dir / "videos"
    spec = cfg.policy_spec
    adapter = OpenPIJaxAdapter(
        make_openpi_jax_client(cfg.policy, action_dim=spec.action_dim, action_horizon=spec.action_horizon),
        spec=spec,
    )

    episode_summaries: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    global_episode_index = 0
    try:
        for resolved_task in resolved_tasks:
            suite_name, task_id, vec_env = make_single_task_vec_env(resolved_task.env_cfg)
            resolved_task_name = resolved_task.task_name or getattr(vec_env.envs[0], "task", None)
            resolved_task_prompt = resolved_task.task_prompt or getattr(vec_env.envs[0], "task_description", None)
            env_preprocessor = make_smoke_env_preprocessor(
                resolved_task.env_cfg,
                spec=spec,
                observation_cfg=cfg.observation,
            )
            task_episode_summaries: list[dict[str, Any]] = []

            try:
                for task_episode_index in range(cfg.runtime.n_episodes):
                    if variation_profile is not None and hasattr(vec_env.envs[0], "set_variation_profile"):
                        vec_env.envs[0].set_variation_profile(
                            variation_profile,
                            seed=cfg.variation.seed + global_episode_index,
                        )

                    try:
                        artifacts = run_smoke_episode(
                            vec_env,
                            adapter,
                            env_preprocessor,
                            cfg.runtime,
                            suite_name=suite_name,
                            task_id=task_id,
                            episode_index=global_episode_index,
                            task_name=resolved_task_name,
                            task_prompt=resolved_task_prompt,
                            control_dt=1.0 / float(resolved_task.env_cfg.fps),
                            spec=adapter.spec,
                            record_video=cfg.runtime.write_video,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if cfg.runtime.fail_fast:
                            raise
                        artifacts = SmokeEpisodeArtifacts(
                            trace=EpisodeTrace(
                                success=False,
                                metadata={
                                    "suite": suite_name,
                                    "task_id": task_id,
                                    "task_name": resolved_task_name,
                                    "task_prompt": resolved_task_prompt,
                                    "exception": repr(exc),
                                },
                            )
                        )

                    trace = artifacts.trace
                    trace_path = None
                    if cfg.runtime.write_trace:
                        trace_path = write_episode_trace(trace, traces_dir, global_episode_index)

                    video_path = None
                    if cfg.runtime.write_video and artifacts.video_frames:
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        video_path = videos_dir / f"episode_{global_episode_index:03d}.mp4"
                        write_video(
                            str(video_path),
                            artifacts.video_frames,
                            fps=cfg.runtime.video_fps or int(resolved_task.env_cfg.fps),
                        )

                    episode_summary = {
                        "episode_index": global_episode_index,
                        "task_episode_index": task_episode_index,
                        "suite": suite_name,
                        "task_id": task_id,
                        "task_name": resolved_task_name,
                        "task_prompt": resolved_task_prompt,
                        "success": trace.success,
                        "metrics": dict(trace.metrics),
                        "trace_path": str(trace_path) if trace_path is not None else None,
                        "video_path": str(video_path) if video_path is not None else None,
                        "variation": trace.metadata.get("variation", {}),
                        "exception": trace.metadata.get("exception"),
                    }
                    task_episode_summaries.append(episode_summary)
                    episode_summaries.append(episode_summary)
                    global_episode_index += 1
            finally:
                vec_env.close()

            task_success_rate = sum(1 for item in task_episode_summaries if item["success"]) / max(
                len(task_episode_summaries), 1
            )
            task_summaries.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": resolved_task_name,
                    "task_prompt": resolved_task_prompt,
                    "episodes": task_episode_summaries,
                    "success_rate": task_success_rate,
                }
            )

        success_rate = sum(1 for item in episode_summaries if item["success"]) / max(len(episode_summaries), 1)
        summary_task = task_summaries[0] if len(task_summaries) == 1 else None
        summary = {
            "suite": summary_task["suite"] if summary_task is not None else None,
            "task_id": summary_task["task_id"] if summary_task is not None else None,
            "task_name": summary_task["task_name"] if summary_task is not None else None,
            "task_prompt": summary_task["task_prompt"] if summary_task is not None else None,
            "num_tasks": len(task_summaries),
            "episodes_per_task": cfg.runtime.n_episodes,
            "tasks": task_summaries,
            "episodes": episode_summaries,
            "success_rate": success_rate,
            "output_dir": str(output_dir),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        LOGGER.info("OpenPI smoke complete. success_rate=%.2f", success_rate)
        return summary
    finally:
        adapter.close()


@parser.wrap()
def smoke_main(cfg: OpenPIBowlSmokeConfig) -> dict[str, Any]:
    return run_bowl_smoke(cfg)


def main() -> None:
    _rewrite_legacy_config_flag()
    init_logging()
    register_third_party_plugins()
    smoke_main()


if __name__ == "__main__":
    main()
