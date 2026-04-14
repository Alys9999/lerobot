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

"""Hidden-Physics Diagnostic Benchmark v1 — task catalog.

The catalog defines every concrete benchmark task as a
:class:`BenchmarkTaskDefinition`.  The built-in ``V1_CATALOG`` contains the
full 24-task set (20 diagnostic + 4 challenge).

Catalog loading and filtering are the only two responsibilities of this
module — it does not touch environments, policies, or results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .hidden_physics_config import (
    BENCHMARK_FAMILIES,
    BENCHMARK_TEMPLATES,
    CHALLENGE_FAMILY,
    DIAGNOSTIC_FAMILIES,
    HiddenPhysicsBenchmarkConfig,
)


# ---------------------------------------------------------------------------
# Variation level definitions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariationLevelDefinition:
    """Numeric range for a single variation level (e.g. iid_high)."""

    level_name: str
    ranges: dict[str, tuple[float, float]]


# ---------------------------------------------------------------------------
# Variation profile definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariationProfileDefinition:
    """Maps a benchmark family to concrete sampling ranges per level."""

    profile_name: str
    family: str
    target: str
    nominal: dict[str, tuple[float, float]] = field(default_factory=dict)
    iid_low: dict[str, tuple[float, float]] = field(default_factory=dict)
    iid_high: dict[str, tuple[float, float]] = field(default_factory=dict)
    ood_low: dict[str, tuple[float, float]] = field(default_factory=dict)
    ood_high: dict[str, tuple[float, float]] = field(default_factory=dict)

    def ranges_for_level(self, level: str) -> dict[str, tuple[float, float]]:
        mapping = {
            "nominal": self.nominal,
            "iid_low": self.iid_low,
            "iid_high": self.iid_high,
            "ood_low": self.ood_low,
            "ood_high": self.ood_high,
        }
        if level not in mapping:
            raise ValueError(f"Unknown level '{level}'. Expected one of {list(mapping)}.")
        return mapping[level]


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkTaskDefinition:
    """A single benchmark task card."""

    benchmark_task_id: str
    family: str
    template: str
    suite_name: str
    base_task_id: int
    prompt: str
    variation_profile: str
    variation_target: str = "bowl"
    success_contract: str = "libero_default_success"
    iid_level: str = "iid_high"
    ood_level: str = "ood_high"
    seed_group: str = "sg1"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in variation profiles for each diagnostic family
# ---------------------------------------------------------------------------

# F — Effective Contact Friction (single axis — see design discussion §4.3, §7.1)
# A single sampled value is applied to both object and finger geoms;
# the benchmark exposes ONE axis called ``effective_contact_friction``.
_PROFILE_F = VariationProfileDefinition(
    profile_name="friction_diagnostic_v1",
    family="F",
    target="bowl",
    nominal={"effective_contact_friction": (1.0, 1.0)},
    iid_low={"effective_contact_friction": (0.6, 1.0)},
    iid_high={"effective_contact_friction": (0.3, 1.5)},
    ood_low={"effective_contact_friction": (0.1, 0.3)},
    ood_high={"effective_contact_friction": (0.05, 0.15)},
)

# M — Object Mass
_PROFILE_M = VariationProfileDefinition(
    profile_name="mass_diagnostic_v1",
    family="M",
    target="bowl",
    nominal={"object_mass": (0.15, 0.15)},
    iid_low={"object_mass": (0.10, 0.20)},
    iid_high={"object_mass": (0.05, 0.40)},
    ood_low={"object_mass": (0.40, 0.80)},
    ood_high={"object_mass": (0.80, 1.50)},
)

# C — Object CoM / Inertia
_PROFILE_C = VariationProfileDefinition(
    profile_name="com_inertia_diagnostic_v1",
    family="C",
    target="bowl",
    nominal={
        "object_com_offset_x": (0.0, 0.0),
        "object_com_offset_y": (0.0, 0.0),
        "object_inertia_diagonal": (1.0, 1.0),
    },
    iid_low={
        "object_com_offset_x": (-0.005, 0.005),
        "object_com_offset_y": (-0.005, 0.005),
        "object_inertia_diagonal": (0.8, 1.2),
    },
    iid_high={
        "object_com_offset_x": (-0.015, 0.015),
        "object_com_offset_y": (-0.015, 0.015),
        "object_inertia_diagonal": (0.5, 2.0),
    },
    ood_low={
        "object_com_offset_x": (-0.03, 0.03),
        "object_com_offset_y": (-0.03, 0.03),
        "object_inertia_diagonal": (0.2, 3.0),
    },
    ood_high={
        "object_com_offset_x": (-0.05, 0.05),
        "object_com_offset_y": (-0.05, 0.05),
        "object_inertia_diagonal": (0.1, 5.0),
    },
)

# P — Robot Passive Dynamics
_PROFILE_P = VariationProfileDefinition(
    profile_name="passive_dynamics_diagnostic_v1",
    family="P",
    target="gripper_fingers",
    nominal={
        "joint_damping": (0.1, 0.1),
        "joint_frictionloss": (0.0, 0.0),
        "joint_armature": (0.0, 0.0),
    },
    iid_low={
        "joint_damping": (0.05, 0.20),
        "joint_frictionloss": (0.0, 0.05),
        "joint_armature": (0.0, 0.01),
    },
    iid_high={
        "joint_damping": (0.01, 0.50),
        "joint_frictionloss": (0.0, 0.15),
        "joint_armature": (0.0, 0.05),
    },
    ood_low={
        "joint_damping": (0.5, 2.0),
        "joint_frictionloss": (0.15, 0.50),
        "joint_armature": (0.05, 0.20),
    },
    ood_high={
        "joint_damping": (2.0, 5.0),
        "joint_frictionloss": (0.50, 1.00),
        "joint_armature": (0.20, 0.50),
    },
)

BUILTIN_PROFILES: dict[str, VariationProfileDefinition] = {
    _PROFILE_F.profile_name: _PROFILE_F,
    _PROFILE_M.profile_name: _PROFILE_M,
    _PROFILE_C.profile_name: _PROFILE_C,
    _PROFILE_P.profile_name: _PROFILE_P,
}

# ---------------------------------------------------------------------------
# Built-in v1 task catalog — 20 diagnostic + 4 challenge = 24 tasks
# ---------------------------------------------------------------------------

# Shared LIBERO backbone mappings per template
# template -> (suite_name, base_task_id, default_prompt, object_variation_target)
_TEMPLATE_BACKBONES: dict[str, tuple[str, int, str, str]] = {
    "T1": ("libero_spatial", 0, "pick up the black bowl on the table center and place it on the plate", "bowl"),
    "T2": ("libero_object", 0, "pick up the alphabet soup and place it in the basket", "alphabet_soup"),
    "T3": ("libero_90", 3, "put the frying pan on the cabinet shelf", "frying_pan"),
    "T4": ("libero_90", 20, "put the book in the back compartment of the caddy", "book"),
    "T5": ("libero_90", 10, "put the bowl in the bottom drawer", "bowl"),
}

# Family -> variation profile name
_FAMILY_PROFILE: dict[str, str] = {
    "F": "friction_diagnostic_v1",
    "M": "mass_diagnostic_v1",
    "C": "com_inertia_diagnostic_v1",
    "P": "passive_dynamics_diagnostic_v1",
}


def _build_diagnostic_tasks() -> list[BenchmarkTaskDefinition]:
    """Generate the 20 diagnostic tasks (4 families x 5 templates)."""
    tasks: list[BenchmarkTaskDefinition] = []
    for family in DIAGNOSTIC_FAMILIES:
        for template in BENCHMARK_TEMPLATES:
            suite_name, base_task_id, prompt, obj_target = _TEMPLATE_BACKBONES[template]
            task_id = f"{family}-{template}"
            # P-family targets robot joints, not the object.
            var_target = "gripper_fingers" if family == "P" else obj_target
            tasks.append(
                BenchmarkTaskDefinition(
                    benchmark_task_id=task_id,
                    family=family,
                    template=template,
                    suite_name=suite_name,
                    base_task_id=base_task_id,
                    prompt=prompt,
                    variation_profile=_FAMILY_PROFILE[family],
                    variation_target=var_target,
                    tags=["diagnostic"],
                )
            )
    return tasks


def _build_challenge_tasks() -> list[BenchmarkTaskDefinition]:
    """Generate the 4 challenge (R-suite) tasks.

    R-T1 — Slippery Object in Box Transfer
        LIBERO libero_object task 5: involves picking a slippery object.
        Friction diagnostic profile adds slip challenge.
    R-T2 — Corner Extraction Then Place
        LIBERO libero_spatial task 5: object between obstacles.
        Mass diagnostic profile adds weight challenge.
    R-T3 — Blocked-Object Rescue
        LIBERO libero_object task 7: pick object with obstacles nearby.
        CoM/inertia diagnostic profile complicates the rescue.
    R-T4 — Reorient-Then-Grasp
        LIBERO libero_spatial task 8: object in awkward orientation area.
        Passive dynamics profile makes reorientation harder.
    """
    return [
        BenchmarkTaskDefinition(
            benchmark_task_id="R-T1",
            family="R",
            template="T1",
            suite_name="libero_object",
            base_task_id=5,
            prompt="pick up the cream cheese and place it in the basket",
            variation_profile="friction_diagnostic_v1",
            variation_target="cream_cheese",
            tags=["challenge", "reasoning"],
        ),
        BenchmarkTaskDefinition(
            benchmark_task_id="R-T2",
            family="R",
            template="T2",
            suite_name="libero_spatial",
            base_task_id=5,
            prompt="pick up the black bowl between the plate and the ramekin and place it on the plate",
            variation_profile="mass_diagnostic_v1",
            variation_target="bowl",
            tags=["challenge", "extraction"],
        ),
        BenchmarkTaskDefinition(
            benchmark_task_id="R-T3",
            family="R",
            template="T3",
            suite_name="libero_object",
            base_task_id=7,
            prompt="pick up the ketchup and place it in the basket",
            variation_profile="com_inertia_diagnostic_v1",
            variation_target="ketchup",
            tags=["challenge", "obstacle_clearing"],
        ),
        BenchmarkTaskDefinition(
            benchmark_task_id="R-T4",
            family="R",
            template="T4",
            suite_name="libero_spatial",
            base_task_id=8,
            prompt="pick up the black bowl next to the ramekin and place it on the stove",
            variation_profile="passive_dynamics_diagnostic_v1",
            variation_target="gripper_fingers",
            tags=["challenge", "reorientation"],
        ),
    ]


V1_CATALOG: list[BenchmarkTaskDefinition] = _build_diagnostic_tasks() + _build_challenge_tasks()


# ---------------------------------------------------------------------------
# Catalog loading & filtering
# ---------------------------------------------------------------------------


def load_catalog(path: str | Path | None = None) -> list[BenchmarkTaskDefinition]:
    """Load a task catalog.

    If *path* is ``None`` or empty, returns the built-in ``V1_CATALOG``.
    Otherwise, supports three formats:

    - **JSON file** (``.json``): a list of task definition dicts.
    - **YAML index** (``index.yaml``): contains a ``task_files`` list
      pointing to individual per-task YAML files relative to the index.
    - **Single YAML file** (``.yaml``/``.yml``): a list of task dicts.
    """
    if not path:
        return list(V1_CATALOG)

    catalog_path = Path(path)
    suffix = catalog_path.suffix.lower()

    if suffix == ".json":
        with catalog_path.open("r", encoding="utf-8") as fh:
            raw: list[dict[str, Any]] = json.load(fh)
        return [BenchmarkTaskDefinition(**entry) for entry in raw]

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML catalogs.") from exc

        with catalog_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        # Index file format: { catalog_version, task_files: [...] }
        if isinstance(data, dict) and "task_files" in data:
            tasks: list[BenchmarkTaskDefinition] = []
            base_dir = catalog_path.parent
            for rel_path in data["task_files"]:
                task_path = base_dir / rel_path
                with task_path.open("r", encoding="utf-8") as tfh:
                    task_data = yaml.safe_load(tfh)
                tasks.append(BenchmarkTaskDefinition(**task_data))
            return tasks

        # Flat list format
        if isinstance(data, list):
            return [BenchmarkTaskDefinition(**entry) for entry in data]

        raise ValueError(f"Unrecognised YAML catalog format at {catalog_path}.")

    raise ValueError(f"Unsupported catalog file extension '{suffix}'. Use .json or .yaml.")


def filter_catalog(
    catalog: Sequence[BenchmarkTaskDefinition],
    *,
    families: Sequence[str] | None = None,
    templates: Sequence[str] | None = None,
    iid_ood_levels: Sequence[str] | None = None,
    seed_groups: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
) -> list[BenchmarkTaskDefinition]:
    """Return only tasks matching all given filters.  Empty/None = no filter."""
    result: list[BenchmarkTaskDefinition] = []
    for task in catalog:
        if families and task.family not in families:
            continue
        if templates and task.template not in templates:
            continue
        if iid_ood_levels and task.iid_level not in iid_ood_levels and task.ood_level not in iid_ood_levels:
            continue
        if seed_groups and task.seed_group not in seed_groups:
            continue
        if task_ids and task.benchmark_task_id not in task_ids:
            continue
        result.append(task)
    return result


def filter_catalog_from_config(
    catalog: Sequence[BenchmarkTaskDefinition],
    config: HiddenPhysicsBenchmarkConfig,
) -> list[BenchmarkTaskDefinition]:
    """Convenience wrapper that reads filters from a benchmark config."""
    return filter_catalog(
        catalog,
        families=config.families or None,
        templates=config.templates or None,
        iid_ood_levels=config.iid_ood_levels or None,
        seed_groups=config.seed_groups or None,
        task_ids=config.task_ids or None,
    )


def save_catalog(catalog: Sequence[BenchmarkTaskDefinition], path: str | Path) -> Path:
    """Persist a catalog as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump([t.to_dict() for t in catalog], fh, indent=2)
    return out


def get_variation_profile_for_task(
    task: BenchmarkTaskDefinition,
) -> VariationProfileDefinition | None:
    """Look up the built-in variation profile for a task."""
    return BUILTIN_PROFILES.get(task.variation_profile)


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------


class CatalogValidationError(Exception):
    """Raised when a catalog fails integrity checks."""


def validate_catalog(
    catalog: Sequence[BenchmarkTaskDefinition],
    *,
    strict: bool = False,
) -> list[str]:
    """Validate a catalog for structural integrity.

    Checks performed:
      1. **Task ID uniqueness** — every ``benchmark_task_id`` must be unique.
      2. **Family validity** — every task's ``family`` must be in
         :data:`BENCHMARK_FAMILIES`.
      3. **Template validity** — every task's ``template`` must be in
         :data:`BENCHMARK_TEMPLATES`.
      4. **Variation profile reference integrity** — for diagnostic tasks the
         ``variation_profile`` must resolve to a known built-in profile.
      5. **Suite name non-empty** — ``suite_name`` must be a non-empty string.

    Returns a list of warning/error messages (empty = valid).  When
    *strict* is ``True`` a :class:`CatalogValidationError` is raised on the
    first error instead of collecting.
    """
    errors: list[str] = []

    def _err(msg: str) -> None:
        if strict:
            raise CatalogValidationError(msg)
        errors.append(msg)

    # 1. Task ID uniqueness
    seen_ids: dict[str, int] = {}
    for idx, task in enumerate(catalog):
        if task.benchmark_task_id in seen_ids:
            _err(
                f"Duplicate benchmark_task_id '{task.benchmark_task_id}' "
                f"at index {idx} (first seen at {seen_ids[task.benchmark_task_id]})."
            )
        else:
            seen_ids[task.benchmark_task_id] = idx

    for task in catalog:
        tid = task.benchmark_task_id

        # 2. Family validity
        if task.family not in BENCHMARK_FAMILIES:
            _err(f"Task '{tid}': unknown family '{task.family}'.")

        # 3. Template validity
        if task.template not in BENCHMARK_TEMPLATES:
            _err(f"Task '{tid}': unknown template '{task.template}'.")

        # 4. Variation profile reference integrity (diagnostic only)
        if task.family in DIAGNOSTIC_FAMILIES:
            if task.variation_profile not in BUILTIN_PROFILES:
                _err(
                    f"Task '{tid}': variation_profile '{task.variation_profile}' "
                    f"not found in BUILTIN_PROFILES."
                )

        # 5. Suite name non-empty
        if not task.suite_name:
            _err(f"Task '{tid}': suite_name is empty.")

    return errors
