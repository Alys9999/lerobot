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

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class VariationApplicationError(RuntimeError):
    pass


class VariationVariable(Protocol):
    name: str

    def sample(self, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def apply(self, env: Any, value: float) -> None:
        raise NotImplementedError

    def validate(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class VariationVariableConfig:
    enabled: bool = False
    target: str = ""
    range: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class VariationVariablesConfig:
    object_friction: VariationVariableConfig = field(
        default_factory=lambda: VariationVariableConfig(enabled=True, target="bowl", range=(0.15, 1.20))
    )
    finger_friction: VariationVariableConfig = field(
        default_factory=lambda: VariationVariableConfig(
            enabled=True, target="gripper_fingers", range=(0.30, 1.50)
        )
    )
    object_weight: VariationVariableConfig = field(
        default_factory=lambda: VariationVariableConfig(enabled=True, target="bowl", range=(0.05, 0.40))
    )


@dataclass(slots=True)
class VariationConfig:
    enabled: bool = True
    seed: int = 123
    profile_name: str = "openpi_bowl_smoke"
    variables: VariationVariablesConfig = field(default_factory=VariationVariablesConfig)


@dataclass(slots=True)
class ScalarRangeVariation:
    name: str
    target: str
    min_value: float
    max_value: float

    def validate(self) -> None:
        if not self.target:
            raise ValueError(f"Variation '{self.name}' requires a non-empty target.")
        if self.min_value > self.max_value:
            raise ValueError(
                f"Variation '{self.name}' has invalid range [{self.min_value}, {self.max_value}]."
            )

    def sample(self, rng: np.random.Generator) -> float:
        self.validate()
        if self.min_value == self.max_value:
            return float(self.min_value)
        return float(rng.uniform(self.min_value, self.max_value))


@dataclass(slots=True)
class ObjectFrictionVariation(ScalarRangeVariation):
    def apply(self, env: Any, value: float) -> None:
        _set_matching_geom_friction(env, self.target, value)


@dataclass(slots=True)
class FingerFrictionVariation(ScalarRangeVariation):
    def apply(self, env: Any, value: float) -> None:
        _set_matching_geom_friction(env, self.target, value)


@dataclass(slots=True)
class ObjectWeightVariation(ScalarRangeVariation):
    def apply(self, env: Any, value: float) -> None:
        _set_matching_body_mass(env, self.target, value)


@dataclass(slots=True)
class VariationProfile:
    profile_name: str
    variables: list[VariationVariable]

    def sample_all(self, rng: np.random.Generator) -> dict[str, float]:
        return {variable.name: variable.sample(rng) for variable in self.variables}

    def apply_all(self, env: Any, sampled: dict[str, float]) -> None:
        for variable in self.variables:
            if variable.name in sampled:
                variable.apply(env, float(sampled[variable.name]))


_TARGET_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "bowl": (("bowl",),),
    "gripper_fingers": (("finger",), ("gripper", "finger")),
}


def build_variation_profile(config: VariationConfig) -> VariationProfile | None:
    if not config.enabled:
        return None

    variables: list[VariationVariable] = []
    object_friction = config.variables.object_friction
    if object_friction.enabled:
        variables.append(
            ObjectFrictionVariation(
                name="object_friction",
                target=object_friction.target,
                min_value=object_friction.range[0],
                max_value=object_friction.range[1],
            )
        )

    finger_friction = config.variables.finger_friction
    if finger_friction.enabled:
        variables.append(
            FingerFrictionVariation(
                name="finger_friction",
                target=finger_friction.target,
                min_value=finger_friction.range[0],
                max_value=finger_friction.range[1],
            )
        )

    object_weight = config.variables.object_weight
    if object_weight.enabled:
        variables.append(
            ObjectWeightVariation(
                name="object_weight",
                target=object_weight.target,
                min_value=object_weight.range[0],
                max_value=object_weight.range[1],
            )
        )

    return VariationProfile(profile_name=config.profile_name, variables=variables)


def _get_sim_model(env: Any) -> Any:
    inner_env = getattr(env, "_env", env)
    if hasattr(inner_env, "sim") and hasattr(inner_env.sim, "model"):
        return inner_env.sim.model
    if hasattr(inner_env, "env") and hasattr(inner_env.env, "sim") and hasattr(inner_env.env.sim, "model"):
        return inner_env.env.sim.model
    raise VariationApplicationError("Could not access MuJoCo simulation model from the environment.")


def _forward_simulation(env: Any) -> None:
    inner_env = getattr(env, "_env", env)
    sim = getattr(inner_env, "sim", None)
    if sim is None and hasattr(inner_env, "env"):
        sim = getattr(inner_env.env, "sim", None)
    if sim is not None and hasattr(sim, "forward"):
        sim.forward()


def _iter_named_ids(model: Any, entity_type: str):
    count = getattr(model, f"n{entity_type}", None)
    if count is None:
        raise VariationApplicationError(f"Model does not expose n{entity_type}.")

    lookup = getattr(model, f"{entity_type}_id2name", None)
    if not callable(lookup):
        raise VariationApplicationError(f"Model does not expose {entity_type}_id2name.")

    for entity_id in range(int(count)):
        yield entity_id, lookup(entity_id)


def _matches_target(name: str | None, target: str) -> bool:
    if not name:
        return False
    normalized_name = name.lower()
    normalized_target = target.lower()
    alias_groups = _TARGET_ALIASES.get(normalized_target)
    if alias_groups is not None:
        return any(all(token in normalized_name for token in tokens) for tokens in alias_groups)
    return normalized_target in normalized_name


def _set_matching_geom_friction(env: Any, target: str, value: float) -> None:
    model = _get_sim_model(env)
    matched = 0
    for geom_id, geom_name in _iter_named_ids(model, "geom"):
        if not _matches_target(geom_name, target):
            continue
        model.geom_friction[geom_id, :] = float(value)
        matched += 1

    if matched == 0:
        raise VariationApplicationError(f"No matching geoms found for target '{target}'.")
    _forward_simulation(env)


def _set_matching_body_mass(env: Any, target: str, value: float) -> None:
    model = _get_sim_model(env)
    matched = 0
    for body_id, body_name in _iter_named_ids(model, "body"):
        if not _matches_target(body_name, target):
            continue
        model.body_mass[body_id] = float(value)
        matched += 1

    if matched == 0:
        raise VariationApplicationError(f"No matching bodies found for target '{target}'.")
    _forward_simulation(env)
