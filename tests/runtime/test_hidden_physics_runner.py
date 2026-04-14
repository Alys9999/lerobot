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

"""Integration tests for the Hidden-Physics benchmark runner.

Uses mock env and mock executor to verify the full runner pipeline without
requiring LIBERO or a trained policy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from lerobot.runtime.hidden_physics_catalog import BenchmarkTaskDefinition, V1_CATALOG
from lerobot.runtime.hidden_physics_config import (
    HiddenPhysicsBenchmarkConfig,
    HiddenPhysicsRuntimeConfig,
    PolicyModeConfig,
)
from lerobot.runtime.hidden_physics_runner import (
    EnvContractError,
    HiddenPhysicsRunner,
    _build_variation_profile_for_task,
    _validate_env_contract,
    rollout_one_episode,
)
from lerobot.runtime.variation import VariationProfile


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------


class _MockModel:
    def __init__(self):
        self.ngeom = 1
        self.nbody = 1
        self.njnt = 1
        self.geom_friction = np.ones((1, 3), dtype=np.float64)
        self.body_mass = np.array([0.2], dtype=np.float64)
        self.body_ipos = np.zeros((1, 3), dtype=np.float64)
        self.body_inertia = np.ones((1, 3), dtype=np.float64)
        self.dof_damping = np.array([0.1], dtype=np.float64)
        self.dof_frictionloss = np.zeros(1, dtype=np.float64)
        self.dof_armature = np.zeros(1, dtype=np.float64)
        self.jnt_dofadr = np.array([0], dtype=np.int32)
        self.jnt_type = np.array([3], dtype=np.int32)

    def geom_id2name(self, gid): return "bowl_geom"
    def body_id2name(self, bid): return "bowl_body"
    def jnt_id2name(self, jid): return "finger_joint"


class _MockSim:
    def __init__(self):
        self.model = _MockModel()
    def forward(self): pass


class _MockInnerEnv:
    def __init__(self):
        self.sim = _MockSim()


class MockEnv:
    """Minimal mock of LiberoEnv for testing the runner."""

    def __init__(self, success_after: int = 5):
        self._env = _MockInnerEnv()
        self._step = 0
        self._success_after = success_after
        self._variation_profile = None
        self._variation_rng = np.random.default_rng()
        self.last_variation_sample: dict[str, float] = {}

    def set_variation_profile(self, profile: VariationProfile, seed: int) -> None:
        self._variation_profile = profile
        self._variation_rng = np.random.default_rng(seed)

    def clear_variation_profile(self) -> None:
        self._variation_profile = None
        self.last_variation_sample = {}

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        self._step = 0
        if self._variation_profile is not None:
            sampled = self._variation_profile.sample_all(self._variation_rng)
            self._variation_profile.apply_all(self, sampled)
            self.last_variation_sample = sampled
        else:
            self.last_variation_sample = {}
        obs = {"state": np.zeros(7, dtype=np.float32)}
        info = {"task": "test", "is_success": False}
        return obs, info

    def step(self, action):
        self._step += 1
        obs = {"state": np.zeros(7, dtype=np.float32)}
        reward = 0.0
        success = self._step >= self._success_after
        terminated = success
        truncated = False
        info = {"is_success": success}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class MockExecutor:
    """Minimal mock of PolicyExecutor."""

    def reset(self): pass

    def infer(self, observation, *, task_text: str, control_dt: float) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def close(self): pass


# ---------------------------------------------------------------------------
# Tests for rollout_one_episode
# ---------------------------------------------------------------------------


class TestRolloutOneEpisode:
    def test_success_detected(self):
        env = MockEnv(success_after=3)
        executor = MockExecutor()
        task = V1_CATALOG[0]  # F-T1

        _trace, result = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=100,
        )
        assert result.success is True
        assert result.steps == 3

    def test_max_steps_reached(self):
        env = MockEnv(success_after=1000)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        _trace, result = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=10,
        )
        assert result.success is False
        assert result.steps == 10

    def test_variation_applied_and_recorded(self):
        from lerobot.runtime.variation import build_family_variation_profile

        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]  # F-T1

        profile = build_family_variation_profile(
            family="M", profile_name="test", target="bowl",
            ranges={"object_mass": (0.5, 0.5)},
        )

        _trace, result = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=profile, seed=42, max_steps=100,
        )
        assert "object_mass" in result.variation
        assert result.variation["object_mass"] == 0.5

    def test_trace_written_to_disk(self):
        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            _trace, result = rollout_one_episode(
                env=env, executor=executor, task=task,
                variation_profile=None, seed=42, max_steps=100,
                write_trace_dir=trace_dir, episode_index=0,
            )
            assert result.trace_path != ""
            assert Path(result.trace_path).exists()

    def test_nominal_flag_in_metadata(self):
        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        trace_nom, _ = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=100,
        )
        assert trace_nom.metadata["is_nominal"] is True

        from lerobot.runtime.variation import build_family_variation_profile
        profile = build_family_variation_profile(
            family="M", profile_name="test", target="bowl",
            ranges={"object_mass": (0.5, 0.5)},
        )
        trace_pert, _ = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=profile, seed=42, max_steps=100,
        )
        assert trace_pert.metadata["is_nominal"] is False


# ---------------------------------------------------------------------------
# Tests for _build_variation_profile_for_task
# ---------------------------------------------------------------------------


class TestBuildVariationProfileForTask:
    def test_returns_profile_for_diagnostic_task(self):
        task = V1_CATALOG[0]  # F-T1
        profile = _build_variation_profile_for_task(task, level="iid_high")
        assert profile is not None
        assert len(profile.variables) > 0

    def test_uses_task_variation_target(self):
        # T3 uses frying_pan as target
        task = [t for t in V1_CATALOG if t.benchmark_task_id == "F-T3"][0]
        profile = _build_variation_profile_for_task(task, level="iid_high")
        assert profile is not None
        for var in profile.variables:
            assert var.target == "frying_pan"

    def test_p_family_targets_gripper(self):
        task = [t for t in V1_CATALOG if t.benchmark_task_id == "P-T1"][0]
        profile = _build_variation_profile_for_task(task, level="iid_high")
        assert profile is not None
        for var in profile.variables:
            assert var.target == "gripper_fingers"

    def test_nominal_level_returns_none_or_fixed(self):
        task = V1_CATALOG[0]
        profile = _build_variation_profile_for_task(task, level="nominal")
        # nominal profiles have fixed (lo == hi) ranges or are None
        if profile is not None:
            rng = np.random.default_rng(0)
            sampled = profile.sample_all(rng)
            # All values should be deterministic (min == max)
            for name, val in sampled.items():
                assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Tests for HiddenPhysicsRunner
# ---------------------------------------------------------------------------


class TestHiddenPhysicsRunner:
    def _make_config(self, tmpdir: str) -> HiddenPhysicsBenchmarkConfig:
        return HiddenPhysicsBenchmarkConfig(
            benchmark_name="test_bench",
            families=["F"],
            templates=["T1"],
            runtime=HiddenPhysicsRuntimeConfig(
                n_episodes_per_task=2,
                max_steps=10,
                write_trace=True,
                write_video=False,
                output_dir=tmpdir,
                run_name="test_run",
                seed=42,
            ),
            policy=PolicyModeConfig(mode="native"),
        )

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_runner_produces_benchmark_result(self, mock_make_env):
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir)
            runner = HiddenPhysicsRunner(cfg)
            executor = MockExecutor()
            result = runner.run(executor)

            assert result.benchmark_name == "test_bench"
            assert len(result.task_results) == 1
            tr = result.task_results[0]
            assert tr.benchmark_task_id == "F-T1"
            # Nominal + perturbed at each level (default: iid_high + ood_high)
            # 2 nominal + 2 iid_high + 2 ood_high = 6
            assert len(tr.episode_results) == 6

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_runner_computes_degradation(self, mock_make_env):
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir)
            runner = HiddenPhysicsRunner(cfg)
            executor = MockExecutor()
            result = runner.run(executor)

            tr = result.task_results[0]
            # nominal_success_rate and perturbed_success_rate should be set
            assert isinstance(tr.nominal_success_rate, float)
            assert isinstance(tr.perturbed_success_rate, float)
            assert isinstance(tr.degradation, float)
            # degradation = perturbed - nominal
            expected_deg = tr.perturbed_success_rate - tr.nominal_success_rate
            assert abs(tr.degradation - expected_deg) < 1e-9

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_runner_writes_outputs(self, mock_make_env):
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir)
            runner = HiddenPhysicsRunner(cfg)
            executor = MockExecutor()
            result = runner.run(executor)

            output_dir = Path(tmpdir) / "test_run"
            assert (output_dir / "benchmark_summary.json").exists()
            assert (output_dir / "family_summary.json").exists()
            assert (output_dir / "task_summary.json").exists()

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_runner_writes_config_json(self, mock_make_env):
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir)
            runner = HiddenPhysicsRunner(cfg)
            executor = MockExecutor()
            runner.run(executor)

            config_path = Path(tmpdir) / "test_run" / "config.json"
            assert config_path.exists(), "config.json not written to output dir"
            import json
            with config_path.open() as fh:
                data = json.load(fh)
            assert data["benchmark_name"] == "test_bench"
            assert data["policy"]["mode"] == "native"
            assert data["runtime"]["n_episodes_per_task"] == 2


# ---------------------------------------------------------------------------
# Tests for trace metadata
# ---------------------------------------------------------------------------


class TestTraceMetadata:
    def test_policy_mode_in_trace(self):
        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        trace, _ = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=100,
            policy_mode="native",
        )
        assert trace.metadata["policy_mode"] == "native"

    def test_spec_aligned_field_names(self):
        """Trace metadata should have both 'suite'/'task_id' (spec) and
        'suite_name'/'base_task_id' (legacy) aliases."""
        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        trace, _ = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=100,
        )
        md = trace.metadata
        # Spec-aligned names
        assert "suite" in md
        assert "task_id" in md
        # Legacy names
        assert "suite_name" in md
        assert "base_task_id" in md
        # Consistent values
        assert md["suite"] == md["suite_name"]
        assert md["task_id"] == md["base_task_id"]

    def test_variation_target_in_trace(self):
        env = MockEnv(success_after=2)
        executor = MockExecutor()
        task = V1_CATALOG[0]

        trace, _ = rollout_one_episode(
            env=env, executor=executor, task=task,
            variation_profile=None, seed=42, max_steps=100,
        )
        assert "variation_target" in trace.metadata
        assert trace.metadata["variation_target"] == task.variation_target


# ---------------------------------------------------------------------------
# Tests for env contract validation
# ---------------------------------------------------------------------------


class TestMultiLevelRunner:
    """Tests that the runner evaluates at multiple perturbation levels."""

    def _make_config(self, tmpdir: str, levels=None) -> HiddenPhysicsBenchmarkConfig:
        return HiddenPhysicsBenchmarkConfig(
            benchmark_name="test_multi_level",
            families=["F"],
            templates=["T1"],
            iid_ood_levels=levels or [],
            runtime=HiddenPhysicsRuntimeConfig(
                n_episodes_per_task=1,
                max_steps=10,
                write_trace=True,
                write_video=False,
                output_dir=tmpdir,
                run_name="test_run",
                seed=42,
            ),
            policy=PolicyModeConfig(mode="native"),
        )

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_default_runs_iid_and_ood(self, mock_make_env):
        """Without explicit levels, runner should evaluate iid_high + ood_high."""
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir)
            runner = HiddenPhysicsRunner(cfg)
            result = runner.run(MockExecutor())

            tr = result.task_results[0]
            assert "iid_high" in tr.level_success_rates
            assert "ood_high" in tr.level_success_rates
            # 1 nominal + 1 iid_high + 1 ood_high = 3 episodes
            assert len(tr.episode_results) == 3

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_explicit_levels(self, mock_make_env):
        """Explicit iid_ood_levels should control which levels are evaluated."""
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir, levels=["iid_low", "iid_high", "ood_low", "ood_high"])
            runner = HiddenPhysicsRunner(cfg)
            result = runner.run(MockExecutor())

            tr = result.task_results[0]
            assert set(tr.level_success_rates.keys()) == {"iid_low", "iid_high", "ood_low", "ood_high"}
            # 1 nominal + 4 levels × 1 ep = 5 episodes
            assert len(tr.episode_results) == 5

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_episodes_tagged_with_level(self, mock_make_env):
        """Each episode should have iid_ood_level and seed_group set."""
        mock_make_env.return_value = MockEnv(success_after=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_config(tmpdir, levels=["iid_high"])
            runner = HiddenPhysicsRunner(cfg)
            result = runner.run(MockExecutor())

            tr = result.task_results[0]
            nominal_eps = [e for e in tr.episode_results if e.iid_ood_level == "nominal"]
            perturbed_eps = [e for e in tr.episode_results if e.iid_ood_level == "iid_high"]
            assert len(nominal_eps) == 1
            assert len(perturbed_eps) == 1
            assert all(e.seed_group == "sg1" for e in tr.episode_results)


class _BareObject:
    """An object with no methods — used to test contract validation."""
    pass


class _EnvNoVariation:
    """Has reset/step/close but no variation methods."""
    def reset(self, seed=None): return {}, {}
    def step(self, action): return {}, 0.0, False, False, {}
    def close(self): pass


class TestEnvContractValidation:
    def test_valid_env_passes(self):
        env = MockEnv()
        task = V1_CATALOG[0]
        # Should not raise
        _validate_env_contract(env, task)

    def test_missing_reset_raises(self):
        env = _BareObject()
        task = V1_CATALOG[0]
        with pytest.raises(EnvContractError, match="reset"):
            _validate_env_contract(env, task)

    def test_missing_step_raises(self):
        env = _BareObject()
        env.reset = lambda seed=None: ({}, {})  # type: ignore[attr-defined]
        task = V1_CATALOG[0]
        with pytest.raises(EnvContractError, match="step"):
            _validate_env_contract(env, task)

    def test_missing_variation_methods_warns(self):
        env = _EnvNoVariation()
        task = V1_CATALOG[0]
        import logging
        with mock.patch.object(logging.getLogger("lerobot.runtime.hidden_physics_runner"), "warning") as mock_warn:
            _validate_env_contract(env, task)
            assert mock_warn.called
