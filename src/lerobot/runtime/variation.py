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
class EffectiveContactFrictionVariation(ScalarRangeVariation):
    """Single-axis friction variation that applies to BOTH the object and
    the gripper finger geoms simultaneously.

    The benchmark exposes a single semantic axis ``effective_contact_friction``
    rather than separate object/finger axes (per the plan's core design
    principle: the final operative friction is the composite contact friction).
    Internally we set the same sampled value on both the object target geoms
    and the finger geoms so MuJoCo's contact resolution sees a uniform change.
    """

    object_target: str = "bowl"
    finger_target: str = "gripper_fingers"

    def apply(self, env: Any, value: float) -> None:
        # Apply the same friction to both object and finger geoms.
        _set_matching_geom_friction(env, self.object_target, value)
        try:
            _set_matching_geom_friction(env, self.finger_target, value)
        except VariationApplicationError:
            # Finger geoms may not exist in all envs — that's okay.
            pass


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


# ---------------------------------------------------------------------------
# Benchmark family: C — Object CoM offset
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObjectCoMOffsetVariation(ScalarRangeVariation):
    """Offsets the center-of-mass of matching bodies along a single axis.

    ``target`` selects bodies by substring/alias.
    ``axis`` is 0 (x), 1 (y), or 2 (z) in body-local frame.
    The sampled value is an **additive** offset in metres.
    """

    axis: int = 0

    def apply(self, env: Any, value: float) -> None:
        _offset_matching_body_ipos(env, self.target, self.axis, value)


@dataclass(slots=True)
class ObjectInertiaDiagonalVariation(ScalarRangeVariation):
    """Scales the diagonal inertia of matching bodies.

    The sampled value is a **multiplicative** scale factor applied to all
    three diagonal inertia entries of each matched body.
    """

    def apply(self, env: Any, value: float) -> None:
        _scale_matching_body_inertia(env, self.target, value)


# ---------------------------------------------------------------------------
# Benchmark family: P — Robot Passive Dynamics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JointDampingVariation(ScalarRangeVariation):
    """Overrides ``dof_damping`` for joints matching *target*."""

    def apply(self, env: Any, value: float) -> None:
        _set_matching_dof_field(env, self.target, "dof_damping", value)


@dataclass(slots=True)
class JointFrictionlossVariation(ScalarRangeVariation):
    """Overrides ``dof_frictionloss`` for joints matching *target*."""

    def apply(self, env: Any, value: float) -> None:
        _set_matching_dof_field(env, self.target, "dof_frictionloss", value)


@dataclass(slots=True)
class JointArmatureVariation(ScalarRangeVariation):
    """Overrides ``dof_armature`` for joints matching *target*."""

    def apply(self, env: Any, value: float) -> None:
        _set_matching_dof_field(env, self.target, "dof_armature", value)


# ---------------------------------------------------------------------------
# Target aliases
# ---------------------------------------------------------------------------


_TARGET_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "bowl": (("bowl",),),
    "gripper_fingers": (("finger",), ("gripper", "finger")),
    "gripper": (("gripper",),),
    "finger": (("finger",),),
    "wrist": (("wrist",),),
    # Per-task object targets used by the benchmark catalog
    "alphabet_soup": (("alphabet", "soup"), ("soup",)),
    "frying_pan": (("frying", "pan"), ("frying_pan",), ("pan",)),
    "book": (("book",),),
    "cream_cheese": (("cream", "cheese"), ("cream_cheese",)),
    "ketchup": (("ketchup",),),
}


# ---------------------------------------------------------------------------
# Benchmark-family-aware variation builder
# ---------------------------------------------------------------------------

# Mapping from benchmark family shorthand to factory functions.
# Each entry is (VariationClass, extra_kwargs).
FAMILY_VARIABLE_REGISTRY: dict[str, list[tuple[type, dict[str, Any]]]] = {
    "F": [
        (EffectiveContactFrictionVariation, {"name": "effective_contact_friction"}),
    ],
    "M": [
        (ObjectWeightVariation, {"name": "object_mass"}),
    ],
    "C": [
        (ObjectCoMOffsetVariation, {"name": "object_com_offset_x", "axis": 0}),
        (ObjectCoMOffsetVariation, {"name": "object_com_offset_y", "axis": 1}),
        (ObjectInertiaDiagonalVariation, {"name": "object_inertia_diagonal"}),
    ],
    "P": [
        (JointDampingVariation, {"name": "joint_damping"}),
        (JointFrictionlossVariation, {"name": "joint_frictionloss"}),
        (JointArmatureVariation, {"name": "joint_armature"}),
    ],
}


def build_variation_profile(config: VariationConfig) -> VariationProfile | None:
    """Build a VariationProfile from legacy ``VariationConfig``."""
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


def build_family_variation_profile(
    family: str,
    profile_name: str,
    target: str,
    ranges: dict[str, tuple[float, float]],
) -> VariationProfile:
    """Build a :class:`VariationProfile` for a benchmark *family*.

    Parameters
    ----------
    family:
        One of ``"F"``, ``"M"``, ``"C"``, ``"P"``.
    profile_name:
        Human-readable profile name for tracing.
    target:
        Default target (body/geom/joint substring) for all variables.
        Individual variables can be overridden via *ranges* keys that end
        with ``":target=<name>"``.
    ranges:
        Mapping of ``variable_name -> (min, max)``.  Only variables whose
        name appears in this mapping will be enabled.  If empty, all
        registered variables for the family are enabled with defaults
        ``(0.0, 0.0)`` (i.e. nominal / no perturbation).
    """
    registry = FAMILY_VARIABLE_REGISTRY.get(family)
    if registry is None:
        raise ValueError(f"Unknown benchmark family '{family}'. Expected one of {list(FAMILY_VARIABLE_REGISTRY)}.")

    variables: list[VariationVariable] = []
    for cls, defaults in registry:
        var_name = defaults["name"]
        if ranges and var_name not in ranges:
            continue
        lo, hi = ranges.get(var_name, (0.0, 0.0))
        kwargs: dict[str, Any] = {**defaults, "target": target, "min_value": lo, "max_value": hi}
        # EffectiveContactFrictionVariation needs object_target and finger_target
        # instead of the generic target.
        if cls is EffectiveContactFrictionVariation:
            kwargs["object_target"] = target
            kwargs["finger_target"] = "gripper_fingers"
        variables.append(cls(**kwargs))

    return VariationProfile(profile_name=profile_name, variables=variables)


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


def _offset_matching_body_ipos(env: Any, target: str, axis: int, offset: float) -> None:
    """Add *offset* to ``body_ipos[body_id, axis]`` for matching bodies."""
    model = _get_sim_model(env)
    matched = 0
    for body_id, body_name in _iter_named_ids(model, "body"):
        if not _matches_target(body_name, target):
            continue
        model.body_ipos[body_id, axis] += float(offset)
        matched += 1

    if matched == 0:
        raise VariationApplicationError(f"No matching bodies found for target '{target}'.")
    _forward_simulation(env)


def _scale_matching_body_inertia(env: Any, target: str, scale: float) -> None:
    """Multiply all three diagonal inertia entries by *scale* for matching bodies."""
    model = _get_sim_model(env)
    matched = 0
    for body_id, body_name in _iter_named_ids(model, "body"):
        if not _matches_target(body_name, target):
            continue
        model.body_inertia[body_id, :] *= float(scale)
        matched += 1

    if matched == 0:
        raise VariationApplicationError(f"No matching bodies found for target '{target}'.")
    _forward_simulation(env)


def _set_matching_dof_field(env: Any, target: str, field_name: str, value: float) -> None:
    """Set ``model.<field_name>[dof_id]`` for DOFs whose parent joint matches *target*."""
    model = _get_sim_model(env)
    arr = getattr(model, field_name, None)
    if arr is None:
        raise VariationApplicationError(f"Model does not expose '{field_name}'.")

    matched = 0
    # MuJoCo: iterate joints, map each to its DOF range.
    njnt = getattr(model, "njnt", 0)
    jnt_id2name = getattr(model, "jnt_id2name", None)
    jnt_dofadr = getattr(model, "jnt_dofadr", None)
    jnt_type = getattr(model, "jnt_type", None)

    if not callable(jnt_id2name):
        raise VariationApplicationError("Model does not expose jnt_id2name.")

    for jnt_id in range(int(njnt)):
        jnt_name = jnt_id2name(jnt_id)
        if not _matches_target(jnt_name, target):
            continue
        dof_start = int(jnt_dofadr[jnt_id]) if jnt_dofadr is not None else jnt_id
        # For hinge (type 3) or slide (type 2) joints: 1 DOF.
        # For free (type 0) or ball (type 1) joints: 6 or 3 DOFs.
        if jnt_type is not None:
            jt = int(jnt_type[jnt_id])
            ndof = {0: 6, 1: 3, 2: 1, 3: 1}.get(jt, 1)
        else:
            ndof = 1
        for d in range(ndof):
            arr[dof_start + d] = float(value)
        matched += 1

    if matched == 0:
        raise VariationApplicationError(f"No matching joints found for target '{target}' (field={field_name}).")
    _forward_simulation(env)
