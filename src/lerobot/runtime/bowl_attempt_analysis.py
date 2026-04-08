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

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class BowlAttemptAnalysisConfig:
    enabled: bool = True
    object_target: str = "bowl"
    start_near_distance_m: float = 0.08
    start_height_delta_m: float = 0.015
    start_displacement_m: float = 0.02
    settle_height_delta_m: float = 0.008
    settle_displacement_m: float = 0.01
    fail_far_distance_m: float = 0.12
    settle_grace_steps: int = 2
    far_grace_steps: int = 2
    detector_version: str = "bowl_attempt_v1"


@dataclass(slots=True)
class BowlAttemptEvent:
    attempt_index: int
    start_step: int
    start_timestamp: float
    start_reason: str
    end_step: int | None = None
    end_timestamp: float | None = None
    end_reason: str | None = None
    outcome: str = "unfinished"
    min_eef_object_dist_m: float | None = None
    max_object_height_delta_m: float | None = None
    max_object_displacement_m: float | None = None


@dataclass(slots=True)
class BowlAttemptSignal:
    object_body_name: str | None
    object_position: np.ndarray | None
    eef_position: np.ndarray | None
    eef_object_dist_m: float | None
    object_height_delta_m: float | None
    object_displacement_m: float | None


def _matches_target(name: str | None, target: str) -> bool:
    if not name:
        return False
    normalized_name = name.lower()
    normalized_target = target.lower()
    return normalized_target in normalized_name


def _find_inner_sim(env: Any) -> Any | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None and hasattr(sim, "model"):
            return sim
        current = getattr(current, "env", None) or getattr(current, "_env", None)
    return None


def _iter_named_bodies(model: Any):
    count = getattr(model, "nbody", None)
    lookup = getattr(model, "body_id2name", None)
    if count is None or not callable(lookup):
        return
    for body_id in range(int(count)):
        yield body_id, lookup(body_id)


def _get_body_position(sim: Any, body_id: int) -> np.ndarray | None:
    data = getattr(sim, "data", None)
    if data is not None:
        if hasattr(data, "body_xpos"):
            return np.asarray(data.body_xpos[body_id], dtype=np.float32).reshape(3)
        if hasattr(data, "xpos"):
            return np.asarray(data.xpos[body_id], dtype=np.float32).reshape(3)

    model = getattr(sim, "model", None)
    if model is not None and hasattr(model, "body_pos"):
        return np.asarray(model.body_pos[body_id], dtype=np.float32).reshape(3)
    return None


def _extract_eef_position(observation: dict[str, Any]) -> np.ndarray | None:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        return None
    eef = robot_state.get("eef")
    if not isinstance(eef, dict) or "pos" not in eef:
        return None
    return np.asarray(eef["pos"], dtype=np.float32).reshape(3)


class BowlAttemptAnalyzer:
    def __init__(self, config: BowlAttemptAnalysisConfig | None = None):
        self.config = config or BowlAttemptAnalysisConfig()
        self._object_body_id: int | None = None
        self._object_body_name: str | None = None
        self._initial_object_position: np.ndarray | None = None
        self._events: list[BowlAttemptEvent] = []
        self._active_event: BowlAttemptEvent | None = None
        self._settle_counter = 0
        self._far_counter = 0
        self._disabled_reason: str | None = None
        self._signal_available = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self, env: Any, observation: dict[str, Any]) -> None:
        self._events = []
        self._active_event = None
        self._settle_counter = 0
        self._far_counter = 0
        self._disabled_reason = None
        self._signal_available = False
        self._object_body_id = None
        self._object_body_name = None
        self._initial_object_position = None

        if not self.config.enabled:
            self._disabled_reason = "disabled"
            return

        sim = _find_inner_sim(env)
        if sim is None:
            self._disabled_reason = "sim_unavailable"
            return

        model = getattr(sim, "model", None)
        if model is None:
            self._disabled_reason = "model_unavailable"
            return

        for body_id, body_name in _iter_named_bodies(model):
            if _matches_target(body_name, self.config.object_target):
                self._object_body_id = int(body_id)
                self._object_body_name = str(body_name)
                break

        if self._object_body_id is None:
            self._disabled_reason = f"object_body_not_found:{self.config.object_target}"
            return

        signal = self._compute_signal(env, observation)
        if signal.object_position is None:
            self._disabled_reason = "object_position_unavailable"
            return

        self._initial_object_position = signal.object_position.copy()
        self._signal_available = True

    def _compute_signal(self, env: Any, observation: dict[str, Any]) -> BowlAttemptSignal:
        sim = _find_inner_sim(env)
        object_position = None
        if sim is not None and self._object_body_id is not None:
            object_position = _get_body_position(sim, self._object_body_id)
        eef_position = _extract_eef_position(observation)

        eef_object_dist_m = None
        if object_position is not None and eef_position is not None:
            eef_object_dist_m = float(np.linalg.norm(eef_position - object_position))

        object_height_delta_m = None
        object_displacement_m = None
        if object_position is not None and self._initial_object_position is not None:
            object_height_delta_m = float(object_position[2] - self._initial_object_position[2])
            object_displacement_m = float(np.linalg.norm(object_position - self._initial_object_position))

        return BowlAttemptSignal(
            object_body_name=self._object_body_name,
            object_position=object_position,
            eef_position=eef_position,
            eef_object_dist_m=eef_object_dist_m,
            object_height_delta_m=object_height_delta_m,
            object_displacement_m=object_displacement_m,
        )

    def _is_start_signal(self, signal: BowlAttemptSignal) -> tuple[bool, str | None]:
        if signal.eef_object_dist_m is None:
            return False, None
        if signal.eef_object_dist_m > self.config.start_near_distance_m:
            return False, None
        if signal.object_height_delta_m is not None and signal.object_height_delta_m >= self.config.start_height_delta_m:
            return True, "lifted_near_object"
        if signal.object_displacement_m is not None and signal.object_displacement_m >= self.config.start_displacement_m:
            return True, "moved_near_object"
        return False, None

    def _update_active_event(self, signal: BowlAttemptSignal) -> None:
        if self._active_event is None:
            return
        if signal.eef_object_dist_m is not None:
            current = self._active_event.min_eef_object_dist_m
            self._active_event.min_eef_object_dist_m = (
                signal.eef_object_dist_m if current is None else min(current, signal.eef_object_dist_m)
            )
        if signal.object_height_delta_m is not None:
            current = self._active_event.max_object_height_delta_m
            self._active_event.max_object_height_delta_m = (
                signal.object_height_delta_m if current is None else max(current, signal.object_height_delta_m)
            )
        if signal.object_displacement_m is not None:
            current = self._active_event.max_object_displacement_m
            self._active_event.max_object_displacement_m = (
                signal.object_displacement_m if current is None else max(current, signal.object_displacement_m)
            )

    def _start_attempt(self, step_id: int, timestamp: float, reason: str, signal: BowlAttemptSignal) -> None:
        event = BowlAttemptEvent(
            attempt_index=len(self._events) + 1,
            start_step=step_id,
            start_timestamp=timestamp,
            start_reason=reason,
        )
        self._events.append(event)
        self._active_event = event
        self._settle_counter = 0
        self._far_counter = 0
        self._update_active_event(signal)

    def _close_attempt(self, step_id: int, timestamp: float, *, outcome: str, reason: str) -> None:
        if self._active_event is None:
            return
        self._active_event.end_step = step_id
        self._active_event.end_timestamp = timestamp
        self._active_event.outcome = outcome
        self._active_event.end_reason = reason
        self._active_event = None
        self._settle_counter = 0
        self._far_counter = 0

    def update(self, env: Any, observation: dict[str, Any], *, step_id: int, timestamp: float, success: bool) -> None:
        if not self._signal_available:
            return

        signal = self._compute_signal(env, observation)
        should_start, start_reason = self._is_start_signal(signal)

        if self._active_event is None and should_start and start_reason is not None:
            self._start_attempt(step_id, timestamp, start_reason, signal)

        if self._active_event is None and success:
            self._start_attempt(step_id, timestamp, "implicit_success", signal)

        self._update_active_event(signal)

        if self._active_event is None:
            return

        if success:
            self._close_attempt(step_id, timestamp, outcome="succeeded", reason="env_success")
            return

        settled = (
            signal.object_height_delta_m is not None
            and signal.object_displacement_m is not None
            and signal.object_height_delta_m <= self.config.settle_height_delta_m
            and signal.object_displacement_m <= self.config.settle_displacement_m
        )
        far = signal.eef_object_dist_m is not None and signal.eef_object_dist_m >= self.config.fail_far_distance_m

        self._settle_counter = self._settle_counter + 1 if settled else 0
        self._far_counter = self._far_counter + 1 if far else 0

        if self._settle_counter >= self.config.settle_grace_steps:
            self._close_attempt(step_id, timestamp, outcome="failed", reason="object_recovered_to_rest")
        elif self._far_counter >= self.config.far_grace_steps:
            self._close_attempt(step_id, timestamp, outcome="failed", reason="eef_left_object")

    def finalize(self, *, step_id: int, timestamp: float, success: bool) -> tuple[dict[str, float], dict[str, Any]]:
        if self._active_event is not None:
            self._close_attempt(
                step_id,
                timestamp,
                outcome="succeeded" if success else "failed",
                reason="episode_end_success" if success else "episode_end_failure",
            )

        attempt_count = len(self._events)
        failed_count = sum(event.outcome == "failed" for event in self._events)
        succeeded_count = sum(event.outcome == "succeeded" for event in self._events)

        success_event = next((event for event in self._events if event.outcome == "succeeded"), None)
        first_failed_event = next((event for event in self._events if event.outcome == "failed"), None)

        metrics: dict[str, float] = {
            "attempt_count": float(attempt_count),
            "failed_attempt_count": float(failed_count),
            "successful_attempt_count": float(succeeded_count),
        }

        if success_event is not None:
            metrics["first_attempt_to_success_s"] = float(success_event.end_timestamp - self._events[0].start_timestamp)
            if first_failed_event is not None and first_failed_event.end_timestamp is not None:
                metrics["first_failure_to_success_s"] = float(
                    success_event.end_timestamp - first_failed_event.end_timestamp
                )

        durations = [
            float(event.end_timestamp - event.start_timestamp)
            for event in self._events
            if event.end_timestamp is not None
        ]
        if durations:
            metrics["mean_attempt_duration_s"] = float(np.mean(durations))
            metrics["max_attempt_duration_s"] = float(np.max(durations))

        metadata: dict[str, Any] = {
            "detector_version": self.config.detector_version,
            "detector_config": asdict(self.config),
            "enabled": bool(self.config.enabled),
            "signal_available": bool(self._signal_available),
            "disabled_reason": self._disabled_reason,
            "object_body_name": self._object_body_name,
            "recovered_after_failure": bool(success_event is not None and first_failed_event is not None),
            "events": [asdict(event) for event in self._events],
        }
        if self._initial_object_position is not None:
            metadata["initial_object_position"] = self._initial_object_position.tolist()

        return metrics, metadata
