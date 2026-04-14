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
import tempfile
from pathlib import Path

from lerobot.runtime.hidden_physics_catalog import (
    BUILTIN_PROFILES,
    V1_CATALOG,
    BenchmarkTaskDefinition,
    CatalogValidationError,
    filter_catalog,
    get_variation_profile_for_task,
    load_catalog,
    save_catalog,
    validate_catalog,
)
from lerobot.runtime.hidden_physics_config import (
    BENCHMARK_FAMILIES,
    BENCHMARK_TEMPLATES,
    CHALLENGE_FAMILY,
    DIAGNOSTIC_FAMILIES,
    HiddenPhysicsBenchmarkConfig,
)


class TestV1Catalog:
    """Tests for the built-in 24-task v1 catalog."""

    def test_catalog_has_24_tasks(self):
        assert len(V1_CATALOG) == 24

    def test_diagnostic_tasks_count(self):
        diag = [t for t in V1_CATALOG if t.family in DIAGNOSTIC_FAMILIES]
        assert len(diag) == 20

    def test_challenge_tasks_count(self):
        challenge = [t for t in V1_CATALOG if t.family == CHALLENGE_FAMILY]
        assert len(challenge) == 4

    def test_unique_task_ids(self):
        ids = [t.benchmark_task_id for t in V1_CATALOG]
        assert len(ids) == len(set(ids))

    def test_all_diagnostic_families_covered(self):
        families = {t.family for t in V1_CATALOG if t.family != CHALLENGE_FAMILY}
        assert families == set(DIAGNOSTIC_FAMILIES)

    def test_all_templates_covered_per_diagnostic_family(self):
        for fam in DIAGNOSTIC_FAMILIES:
            templates = {t.template for t in V1_CATALOG if t.family == fam}
            assert templates == set(BENCHMARK_TEMPLATES), f"Family {fam} missing templates"

    def test_each_task_has_valid_variation_profile(self):
        for task in V1_CATALOG:
            if task.family in DIAGNOSTIC_FAMILIES:
                profile = get_variation_profile_for_task(task)
                assert profile is not None, f"Task {task.benchmark_task_id} has no profile"
                assert profile.family == task.family

    def test_task_to_dict_roundtrip(self):
        task = V1_CATALOG[0]
        d = task.to_dict()
        rebuilt = BenchmarkTaskDefinition(**d)
        assert rebuilt.benchmark_task_id == task.benchmark_task_id
        assert rebuilt.family == task.family


class TestCatalogFilter:
    def test_filter_by_family(self):
        result = filter_catalog(V1_CATALOG, families=["F"])
        assert all(t.family == "F" for t in result)
        assert len(result) == 5

    def test_filter_by_template(self):
        result = filter_catalog(V1_CATALOG, templates=["T1"])
        assert all(t.template == "T1" for t in result)

    def test_filter_by_task_id(self):
        result = filter_catalog(V1_CATALOG, task_ids=["F-T1", "M-T2"])
        assert len(result) == 2

    def test_filter_combined(self):
        result = filter_catalog(V1_CATALOG, families=["F", "M"], templates=["T1", "T2"])
        assert len(result) == 4

    def test_filter_empty_means_all(self):
        result = filter_catalog(V1_CATALOG)
        assert len(result) == 24


class TestCatalogPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.json"
            save_catalog(V1_CATALOG, path)
            loaded = load_catalog(path)
            assert len(loaded) == len(V1_CATALOG)
            assert loaded[0].benchmark_task_id == V1_CATALOG[0].benchmark_task_id

    def test_load_default_returns_v1(self):
        catalog = load_catalog(None)
        assert len(catalog) == 24


class TestCatalogValidation:
    """Tests for validate_catalog()."""

    def test_v1_catalog_is_valid(self):
        errors = validate_catalog(V1_CATALOG)
        assert errors == [], f"V1 catalog has validation errors: {errors}"

    def test_duplicate_task_id_detected(self):
        dup = list(V1_CATALOG) + [V1_CATALOG[0]]
        errors = validate_catalog(dup)
        assert len(errors) >= 1
        assert "Duplicate" in errors[0]

    def test_unknown_family_detected(self):
        bad_task = BenchmarkTaskDefinition(
            benchmark_task_id="X-T1",
            family="X",
            template="T1",
            suite_name="libero_spatial",
            base_task_id=0,
            prompt="test",
            variation_profile="friction_diagnostic_v1",
        )
        errors = validate_catalog([bad_task])
        assert any("unknown family" in e for e in errors)

    def test_unknown_template_detected(self):
        bad_task = BenchmarkTaskDefinition(
            benchmark_task_id="F-T99",
            family="F",
            template="T99",
            suite_name="libero_spatial",
            base_task_id=0,
            prompt="test",
            variation_profile="friction_diagnostic_v1",
        )
        errors = validate_catalog([bad_task])
        assert any("unknown template" in e for e in errors)

    def test_missing_profile_detected(self):
        bad_task = BenchmarkTaskDefinition(
            benchmark_task_id="F-T1",
            family="F",
            template="T1",
            suite_name="libero_spatial",
            base_task_id=0,
            prompt="test",
            variation_profile="nonexistent_profile",
        )
        errors = validate_catalog([bad_task])
        assert any("not found in BUILTIN_PROFILES" in e for e in errors)

    def test_empty_suite_detected(self):
        bad_task = BenchmarkTaskDefinition(
            benchmark_task_id="F-T1",
            family="F",
            template="T1",
            suite_name="",
            base_task_id=0,
            prompt="test",
            variation_profile="friction_diagnostic_v1",
        )
        errors = validate_catalog([bad_task])
        assert any("suite_name is empty" in e for e in errors)

    def test_strict_mode_raises(self):
        dup = list(V1_CATALOG) + [V1_CATALOG[0]]
        import pytest
        with pytest.raises(CatalogValidationError):
            validate_catalog(dup, strict=True)


class TestYAMLCatalogLoading:
    """Tests for YAML-based catalog loading."""

    def test_load_yaml_index(self):
        index_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "catalog"
            / "index.yaml"
        )
        if not index_path.exists():
            import pytest
            pytest.skip(f"YAML index not found at {index_path}")

        catalog = load_catalog(index_path)
        assert len(catalog) == 24
        ids = {t.benchmark_task_id for t in catalog}
        assert "F-T1" in ids
        assert "R-T4" in ids

    def test_yaml_catalog_matches_builtin(self):
        index_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "catalog"
            / "index.yaml"
        )
        if not index_path.exists():
            import pytest
            pytest.skip(f"YAML index not found at {index_path}")

        yaml_catalog = load_catalog(index_path)
        builtin_catalog = load_catalog(None)
        assert len(yaml_catalog) == len(builtin_catalog)
        for yt, bt in zip(yaml_catalog, builtin_catalog):
            assert yt.benchmark_task_id == bt.benchmark_task_id
            assert yt.family == bt.family
            assert yt.variation_target == bt.variation_target


class TestBuiltinProfiles:
    def test_all_diagnostic_families_have_profiles(self):
        profile_families = {p.family for p in BUILTIN_PROFILES.values()}
        assert set(DIAGNOSTIC_FAMILIES).issubset(profile_families)

    def test_profiles_have_all_levels(self):
        for profile in BUILTIN_PROFILES.values():
            for level in ("nominal", "iid_low", "iid_high", "ood_low", "ood_high"):
                ranges = profile.ranges_for_level(level)
                assert isinstance(ranges, dict)
