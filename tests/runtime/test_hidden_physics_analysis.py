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

from lerobot.runtime.hidden_physics_analysis import (
    compare_templates,
    diagnose_families,
    format_text_summary,
    generate_full_analysis,
    report_challenge_suite,
    write_analysis_report,
)
from lerobot.runtime.hidden_physics_results import (
    BenchmarkResult,
    FamilyResult,
    TaskResult,
    aggregate_benchmark_result,
)


def _make_benchmark_result() -> BenchmarkResult:
    """Build a small synthetic benchmark result for testing."""
    tasks = [
        TaskResult(benchmark_task_id="F-T1", family="F", template="T1",
                   suite_name="s", base_task_id=0, success_rate=0.8, degradation=-0.1),
        TaskResult(benchmark_task_id="F-T2", family="F", template="T2",
                   suite_name="s", base_task_id=0, success_rate=0.6, degradation=-0.3),
        TaskResult(benchmark_task_id="M-T1", family="M", template="T1",
                   suite_name="s", base_task_id=0, success_rate=0.9, degradation=-0.05),
        TaskResult(benchmark_task_id="R-T1", family="R", template="T1",
                   suite_name="s", base_task_id=0, success_rate=0.4, degradation=0.0),
    ]
    return aggregate_benchmark_result("test_bench", tasks, "test_policy", "native")


class TestDiagnosis:
    def test_diagnose_families_excludes_challenge(self):
        result = _make_benchmark_result()
        diag = diagnose_families(result)
        families = {d["family"] for d in diag}
        assert "R" not in families
        assert "F" in families
        assert "M" in families

    def test_diagnose_families_sorted_by_degradation(self):
        result = _make_benchmark_result()
        diag = diagnose_families(result)
        # F has worse degradation than M
        assert diag[0]["family"] == "M"  # -0.05 first because of abs value? No, sorted descending
        # F mean_deg = (-0.1 + -0.3)/2 = -0.2, M mean_deg = -0.05
        # Sorted by degradation descending (worst first): M has -0.05, F has -0.2
        # Actually -0.05 > -0.2, so M comes first (sorted reverse)
        assert diag[0]["mean_degradation"] >= diag[1]["mean_degradation"]


class TestTemplateComparison:
    def test_compare_templates_excludes_challenge(self):
        result = _make_benchmark_result()
        comparison = compare_templates(result)
        for entry in comparison:
            assert "R" not in entry["family_success_rates"]

    def test_compare_templates_has_correct_templates(self):
        result = _make_benchmark_result()
        comparison = compare_templates(result)
        templates = {e["template"] for e in comparison}
        assert "T1" in templates
        assert "T2" in templates


class TestChallengeSuite:
    def test_report_challenge_suite(self):
        result = _make_benchmark_result()
        report = report_challenge_suite(result)
        assert report["family"] == "R"
        assert report["n_tasks"] == 1
        assert report["tasks"][0]["benchmark_task_id"] == "R-T1"


class TestFullAnalysis:
    def test_generate_full_analysis_is_serialisable(self):
        result = _make_benchmark_result()
        analysis = generate_full_analysis(result)
        json.dumps(analysis)  # must not raise

    def test_write_analysis_report(self):
        result = _make_benchmark_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_analysis_report(result, tmpdir)
            assert path.exists()
            with path.open() as fh:
                data = json.load(fh)
            assert "family_diagnosis" in data
            assert "template_comparison" in data
            assert "challenge_suite" in data


class TestTextSummary:
    def test_format_text_summary(self):
        result = _make_benchmark_result()
        text = format_text_summary(result)
        assert "test_bench" in text
        assert "Diagnostic Families" in text
