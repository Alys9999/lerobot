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

"""Result schema and aggregation for the Hidden-Physics Diagnostic Benchmark.

Defines the result hierarchy:

    EpisodeResult -> TaskResult -> FamilyResult -> BenchmarkResult

All objects are JSON-serialisable via :meth:`to_dict`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EpisodeResult:
    benchmark_task_id: str
    family: str
    template: str
    episode_index: int
    success: bool
    steps: int = 0
    elapsed_s: float = 0.0
    iid_ood_level: str = ""
    seed_group: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    variation: dict[str, float] = field(default_factory=dict)
    trace_path: str = ""
    video_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaskResult:
    benchmark_task_id: str
    family: str
    template: str
    suite_name: str
    base_task_id: int
    success_rate: float = 0.0
    nominal_success_rate: float = 0.0
    perturbed_success_rate: float = 0.0
    degradation: float = 0.0
    mean_steps: float = 0.0
    mean_latency: float = 0.0
    seed_level_variance: float = 0.0
    level_success_rates: dict[str, float] = field(default_factory=dict)
    episode_results: list[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["episode_results"] = [e.to_dict() for e in self.episode_results]
        return d


# ---------------------------------------------------------------------------
# Family
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FamilyResult:
    family: str
    family_full_name: str = ""
    task_ids: list[str] = field(default_factory=list)
    mean_success_rate: float = 0.0
    mean_degradation: float = 0.0
    template_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Overall benchmark
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkResult:
    benchmark_name: str
    policy_name: str = ""
    policy_mode: str = ""
    task_results: list[TaskResult] = field(default_factory=list)
    family_results: list[FamilyResult] = field(default_factory=list)
    overall_success_rate: float = 0.0
    overall_degradation: float = 0.0
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "benchmark_name": self.benchmark_name,
            "policy_name": self.policy_name,
            "policy_mode": self.policy_mode,
            "overall_success_rate": self.overall_success_rate,
            "overall_degradation": self.overall_degradation,
            "report_paths": dict(self.report_paths),
            "task_results": [t.to_dict() for t in self.task_results],
            "family_results": [f.to_dict() for f in self.family_results],
        }
        return d


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_task_result(
    benchmark_task_id: str,
    family: str,
    template: str,
    suite_name: str,
    base_task_id: int,
    episodes: Sequence[EpisodeResult],
) -> TaskResult:
    """Aggregate a list of episode results into a single task result."""
    if not episodes:
        return TaskResult(
            benchmark_task_id=benchmark_task_id,
            family=family,
            template=template,
            suite_name=suite_name,
            base_task_id=base_task_id,
        )

    successes = [e.success for e in episodes]
    steps = [e.steps for e in episodes]
    latencies = [e.elapsed_s for e in episodes if e.elapsed_s > 0]
    success_rate = sum(successes) / len(successes)
    mean_steps = sum(steps) / len(steps) if steps else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Seed-level variance: variance of per-seed success (0 or 1) across
    # all episodes.  A low value means the result is stable across seeds.
    seed_var = 0.0
    if len(successes) > 1:
        s_arr = [float(s) for s in successes]
        mean_s = sum(s_arr) / len(s_arr)
        seed_var = sum((x - mean_s) ** 2 for x in s_arr) / len(s_arr)

    return TaskResult(
        benchmark_task_id=benchmark_task_id,
        family=family,
        template=template,
        suite_name=suite_name,
        base_task_id=base_task_id,
        success_rate=success_rate,
        perturbed_success_rate=success_rate,
        mean_steps=mean_steps,
        mean_latency=mean_latency,
        seed_level_variance=seed_var,
        episode_results=list(episodes),
    )


def aggregate_family_result(
    family: str,
    task_results: Sequence[TaskResult],
    family_full_name: str = "",
) -> FamilyResult:
    """Aggregate task results for a single family."""
    if not task_results:
        return FamilyResult(family=family, family_full_name=family_full_name)

    task_ids = [t.benchmark_task_id for t in task_results]
    rates = [t.success_rate for t in task_results]
    degradations = [t.degradation for t in task_results]
    mean_rate = sum(rates) / len(rates)
    mean_deg = sum(degradations) / len(degradations)
    template_breakdown = {t.template: t.success_rate for t in task_results}

    return FamilyResult(
        family=family,
        family_full_name=family_full_name,
        task_ids=task_ids,
        mean_success_rate=mean_rate,
        mean_degradation=mean_deg,
        template_breakdown=template_breakdown,
    )


def aggregate_benchmark_result(
    benchmark_name: str,
    task_results: Sequence[TaskResult],
    policy_name: str = "",
    policy_mode: str = "",
) -> BenchmarkResult:
    """Build the top-level benchmark result from all task results."""
    from .hidden_physics_config import FAMILY_FULL_NAMES

    # Group by family
    family_map: dict[str, list[TaskResult]] = {}
    for tr in task_results:
        family_map.setdefault(tr.family, []).append(tr)

    family_results = [
        aggregate_family_result(
            family=fam,
            task_results=trs,
            family_full_name=FAMILY_FULL_NAMES.get(fam, fam),
        )
        for fam, trs in sorted(family_map.items())
    ]

    all_rates = [t.success_rate for t in task_results]
    overall_rate = sum(all_rates) / len(all_rates) if all_rates else 0.0
    all_deg = [t.degradation for t in task_results]
    overall_deg = sum(all_deg) / len(all_deg) if all_deg else 0.0

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        policy_name=policy_name,
        policy_mode=policy_mode,
        task_results=list(task_results),
        family_results=family_results,
        overall_success_rate=overall_rate,
        overall_degradation=overall_deg,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_benchmark_result(result: BenchmarkResult, output_dir: str | Path) -> dict[str, Path]:
    """Write benchmark, family, and task summaries to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Overall
    bpath = out / "benchmark_summary.json"
    with bpath.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    paths["benchmark_summary"] = bpath

    # Family summaries
    fpath = out / "family_summary.json"
    with fpath.open("w", encoding="utf-8") as fh:
        json.dump([f.to_dict() for f in result.family_results], fh, indent=2)
    paths["family_summary"] = fpath

    # Task summaries
    tpath = out / "task_summary.json"
    with tpath.open("w", encoding="utf-8") as fh:
        json.dump([t.to_dict() for t in result.task_results], fh, indent=2)
    paths["task_summary"] = tpath

    return paths
