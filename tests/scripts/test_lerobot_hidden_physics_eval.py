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

"""Integration tests for the Hidden-Physics benchmark CLI scripts.

These tests verify the CLI argument parsing, config loading from YAML,
and the report regeneration script — all without requiring a real LIBERO
env or a trained policy (mocked out).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Reusable mocks (same pattern as runner tests)
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

    def geom_id2name(self, gid):
        return "bowl_geom"

    def body_id2name(self, bid):
        return "bowl_body"

    def jnt_id2name(self, jid):
        return "finger_joint"


class _MockSim:
    def __init__(self):
        self.model = _MockModel()

    def forward(self):
        pass


class _MockInnerEnv:
    def __init__(self):
        self.sim = _MockSim()


class MockEnv:
    def __init__(self, success_after=3):
        self._env = _MockInnerEnv()
        self._step = 0
        self._success_after = success_after
        self._variation_profile = None
        self._variation_rng = np.random.default_rng()
        self.last_variation_sample = {}

    def set_variation_profile(self, profile, seed):
        self._variation_profile = profile
        self._variation_rng = np.random.default_rng(seed)

    def clear_variation_profile(self):
        self._variation_profile = None
        self.last_variation_sample = {}

    def reset(self, seed=None):
        self._step = 0
        if self._variation_profile is not None:
            sampled = self._variation_profile.sample_all(self._variation_rng)
            self._variation_profile.apply_all(self, sampled)
            self.last_variation_sample = sampled
        else:
            self.last_variation_sample = {}
        return {"state": np.zeros(7, dtype=np.float32)}, {"is_success": False}

    def step(self, action):
        self._step += 1
        success = self._step >= self._success_after
        return (
            {"state": np.zeros(7, dtype=np.float32)},
            0.0,
            success,
            False,
            {"is_success": success},
        )

    def close(self):
        pass


class MockExecutor:
    def reset(self):
        pass

    def infer(self, observation, *, task_text, control_dt):
        return np.zeros(7, dtype=np.float32)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests for the eval CLI
# ---------------------------------------------------------------------------


class TestEvalCLIParsing:
    """Tests for argument parsing in lerobot_hidden_physics_eval."""

    def test_parse_defaults(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _parse_args

        args = _parse_args([])
        assert args.config_path == ""
        assert args.families is None
        assert args.policy_mode is None

    def test_parse_families_filter(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _parse_args

        args = _parse_args(["--families", "F", "M"])
        assert args.families == ["F", "M"]

    def test_parse_runtime_overrides(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _parse_args

        args = _parse_args(["--n_episodes_per_task", "3", "--max_steps", "50", "--seed", "99"])
        assert args.n_episodes_per_task == 3
        assert args.max_steps == 50
        assert args.seed == 99


class TestEvalCLIYAMLLoading:
    """Tests that YAML config files are loaded correctly."""

    def test_load_smoke_config(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _load_config_from_yaml

        smoke_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "hidden_physics_v1_smoke.yaml"
        )
        if not smoke_path.exists():
            pytest.skip(f"Smoke config not found at {smoke_path}")

        cfg = _load_config_from_yaml(str(smoke_path))
        assert cfg.benchmark_name == "hidden_physics_v1_smoke"
        assert cfg.families == ["F", "M", "C", "P"]
        assert cfg.templates == ["T1"]
        assert cfg.runtime.n_episodes_per_task == 2
        assert cfg.runtime.fail_fast is True
        assert cfg.policy.mode == "native"

    def test_load_full_config(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _load_config_from_yaml

        full_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "hidden_physics_v1_full.yaml"
        )
        if not full_path.exists():
            pytest.skip(f"Full config not found at {full_path}")

        cfg = _load_config_from_yaml(str(full_path))
        assert cfg.benchmark_name == "hidden_physics_v1"
        assert cfg.families == []  # empty = all
        assert cfg.runtime.n_episodes_per_task == 10

    def test_cli_overrides_yaml(self):
        from lerobot.scripts.lerobot_hidden_physics_eval import _load_config, _parse_args

        smoke_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "hidden_physics_v1_smoke.yaml"
        )
        if not smoke_path.exists():
            pytest.skip(f"Smoke config not found at {smoke_path}")

        args = _parse_args([
            "--config_path", str(smoke_path),
            "--families", "F",
            "--n_episodes_per_task", "5",
        ])
        cfg = _load_config(args)
        assert cfg.families == ["F"]  # overridden
        assert cfg.runtime.n_episodes_per_task == 5  # overridden
        assert cfg.runtime.max_steps == 200  # from YAML


class TestEvalCLIEndToEnd:
    """End-to-end test with mocked env and executor."""

    @mock.patch("lerobot.runtime.hidden_physics_runner._make_single_env")
    def test_main_runs_smoke(self, mock_make_env, capsys):
        mock_make_env.return_value = MockEnv(success_after=3)

        smoke_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "benchmark"
            / "hidden_physics"
            / "hidden_physics_v1_smoke.yaml"
        )
        if not smoke_path.exists():
            pytest.skip(f"Smoke config not found at {smoke_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            from lerobot.scripts.lerobot_hidden_physics_eval import _load_config, _parse_args
            from lerobot.runtime.hidden_physics_analysis import (
                format_text_summary,
                write_analysis_report,
                write_markdown_report,
            )
            from lerobot.runtime.hidden_physics_runner import HiddenPhysicsRunner

            args = _parse_args([
                "--config_path", str(smoke_path),
                "--output_dir", tmpdir,
                "--run_name", "cli_test",
                "--n_episodes_per_task", "1",
            ])
            cfg = _load_config(args)

            executor = MockExecutor()
            runner = HiddenPhysicsRunner(cfg)
            result = runner.run(executor)

            # The runner writes to its own output dir
            import time
            output_dir = Path(tmpdir) / "cli_test"

            # Write analysis outputs (same as main() does)
            write_analysis_report(result, output_dir)
            write_markdown_report(result, output_dir)

            assert (output_dir / "benchmark_summary.json").exists()
            assert (output_dir / "config.json").exists()
            assert (output_dir / "analysis_report.json").exists()
            assert (output_dir / "report.md").exists()


# ---------------------------------------------------------------------------
# Tests for the report CLI
# ---------------------------------------------------------------------------


class TestReportCLI:
    """Tests for lerobot_hidden_physics_report."""

    def test_missing_summary_exits_with_error(self):
        """Report CLI should exit(1) when benchmark_summary.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from lerobot.scripts.lerobot_hidden_physics_report import main

            with pytest.raises(SystemExit) as exc_info:
                main(["--result_dir", tmpdir])
            assert exc_info.value.code == 1

    def test_report_regeneration(self):
        """Write a mock benchmark_summary.json, then run the report CLI."""
        from lerobot.runtime.hidden_physics_results import (
            BenchmarkResult,
            EpisodeResult,
            TaskResult,
            aggregate_benchmark_result,
            aggregate_task_result,
            write_benchmark_result,
        )

        # Create a minimal result
        episodes = [
            EpisodeResult(
                benchmark_task_id="F-T1", family="F", template="T1",
                episode_index=i, success=(i % 2 == 0), steps=5,
            )
            for i in range(4)
        ]
        tr = aggregate_task_result("F-T1", "F", "T1", "libero_spatial", 0, episodes)
        br = aggregate_benchmark_result("test", [tr], "test_policy", "native")

        with tempfile.TemporaryDirectory() as tmpdir:
            write_benchmark_result(br, tmpdir)

            from lerobot.scripts.lerobot_hidden_physics_report import main

            main(["--result_dir", tmpdir])

            assert (Path(tmpdir) / "analysis_report.json").exists()
            assert (Path(tmpdir) / "report.md").exists()
