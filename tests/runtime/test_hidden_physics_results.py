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

from lerobot.runtime.hidden_physics_results import (
    BenchmarkResult,
    EpisodeResult,
    FamilyResult,
    TaskResult,
    aggregate_benchmark_result,
    aggregate_family_result,
    aggregate_task_result,
    write_benchmark_result,
)


def _make_episodes(task_id: str, family: str, template: str, successes: list[bool]) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            benchmark_task_id=task_id,
            family=family,
            template=template,
            episode_index=i,
            success=s,
            steps=100,
        )
        for i, s in enumerate(successes)
    ]


class TestAggregation:
    def test_aggregate_task_result(self):
        episodes = _make_episodes("F-T1", "F", "T1", [True, True, False, True])
        tr = aggregate_task_result("F-T1", "F", "T1", "libero_spatial", 0, episodes)
        assert tr.success_rate == 0.75
        assert tr.mean_steps == 100.0
        assert len(tr.episode_results) == 4

    def test_aggregate_task_result_empty(self):
        tr = aggregate_task_result("F-T1", "F", "T1", "libero_spatial", 0, [])
        assert tr.success_rate == 0.0

    def test_aggregate_family_result(self):
        tr1 = TaskResult(benchmark_task_id="F-T1", family="F", template="T1",
                         suite_name="s", base_task_id=0, success_rate=0.8)
        tr2 = TaskResult(benchmark_task_id="F-T2", family="F", template="T2",
                         suite_name="s", base_task_id=0, success_rate=0.6)
        fr = aggregate_family_result("F", [tr1, tr2], "Effective Contact Friction")
        assert fr.mean_success_rate == 0.7
        assert fr.template_breakdown == {"T1": 0.8, "T2": 0.6}

    def test_aggregate_benchmark_result(self):
        tr1 = TaskResult(benchmark_task_id="F-T1", family="F", template="T1",
                         suite_name="s", base_task_id=0, success_rate=0.8)
        tr2 = TaskResult(benchmark_task_id="M-T1", family="M", template="T1",
                         suite_name="s", base_task_id=0, success_rate=0.6)
        br = aggregate_benchmark_result("test", [tr1, tr2])
        assert br.overall_success_rate == 0.7
        assert len(br.family_results) == 2

    def test_to_dict_serialisable(self):
        episodes = _make_episodes("F-T1", "F", "T1", [True, False])
        tr = aggregate_task_result("F-T1", "F", "T1", "libero_spatial", 0, episodes)
        br = aggregate_benchmark_result("test", [tr])
        d = br.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)


class TestPersistence:
    def test_write_benchmark_result(self):
        episodes = _make_episodes("F-T1", "F", "T1", [True, False])
        tr = aggregate_task_result("F-T1", "F", "T1", "libero_spatial", 0, episodes)
        br = aggregate_benchmark_result("test", [tr])

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_benchmark_result(br, tmpdir)
            assert "benchmark_summary" in paths
            assert paths["benchmark_summary"].exists()
            assert paths["family_summary"].exists()
            assert paths["task_summary"].exists()

            with paths["benchmark_summary"].open() as fh:
                data = json.load(fh)
            assert data["benchmark_name"] == "test"
