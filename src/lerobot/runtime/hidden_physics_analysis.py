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

"""Analysis and reporting for the Hidden-Physics Diagnostic Benchmark.

Responsibilities:
  - Family-level diagnosis (which families cause the most degradation)
  - Template-level comparison (which operation phases are most sensitive)
  - Challenge suite (R-suite) separate reporting

This module reads :class:`BenchmarkResult` objects and produces human-readable
reports.  It never runs rollouts or modifies trace data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hidden_physics_config import CHALLENGE_FAMILY, DIAGNOSTIC_FAMILIES, FAMILY_FULL_NAMES, TEMPLATE_FULL_NAMES
from .hidden_physics_results import BenchmarkResult, FamilyResult, TaskResult


# ---------------------------------------------------------------------------
# Family-level diagnosis
# ---------------------------------------------------------------------------


def diagnose_families(result: BenchmarkResult) -> list[dict[str, Any]]:
    """Return a list of per-family diagnosis dicts sorted by degradation (worst first)."""
    diag: list[dict[str, Any]] = []
    for fr in result.family_results:
        if fr.family == CHALLENGE_FAMILY:
            continue  # R-suite is reported separately
        diag.append({
            "family": fr.family,
            "family_full_name": fr.family_full_name or FAMILY_FULL_NAMES.get(fr.family, fr.family),
            "mean_success_rate": round(fr.mean_success_rate, 4),
            "mean_degradation": round(fr.mean_degradation, 4),
            "template_breakdown": {k: round(v, 4) for k, v in fr.template_breakdown.items()},
            "n_tasks": len(fr.task_ids),
        })
    diag.sort(key=lambda d: d["mean_degradation"], reverse=True)
    return diag


# ---------------------------------------------------------------------------
# Template-level comparison
# ---------------------------------------------------------------------------


def compare_templates(result: BenchmarkResult) -> list[dict[str, Any]]:
    """Compare the same template across different families.

    Returns a list of per-template dicts, each containing per-family success
    rates.  This answers: "which operation phase is most sensitive to which
    physics family?"
    """
    # Collect task results grouped by template
    template_map: dict[str, dict[str, float]] = {}
    for tr in result.task_results:
        if tr.family == CHALLENGE_FAMILY:
            continue
        template_map.setdefault(tr.template, {})[tr.family] = tr.success_rate

    comparison: list[dict[str, Any]] = []
    for template, family_rates in sorted(template_map.items()):
        rates = list(family_rates.values())
        comparison.append({
            "template": template,
            "template_full_name": TEMPLATE_FULL_NAMES.get(template, template),
            "family_success_rates": {k: round(v, 4) for k, v in family_rates.items()},
            "mean_success_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "worst_family": min(family_rates, key=family_rates.get) if family_rates else "",
        })
    return comparison


# ---------------------------------------------------------------------------
# Challenge suite reporting
# ---------------------------------------------------------------------------


def report_challenge_suite(result: BenchmarkResult) -> dict[str, Any]:
    """Generate a separate report for R-suite (challenge) tasks.

    R-suite is intentionally NOT mixed with the diagnostic F/M/C/P analysis.
    It only reports complex reasoning / recovery / multi-strategy ability.
    """
    challenge_tasks = [tr for tr in result.task_results if tr.family == CHALLENGE_FAMILY]
    if not challenge_tasks:
        return {"family": CHALLENGE_FAMILY, "n_tasks": 0, "tasks": []}

    rates = [t.success_rate for t in challenge_tasks]
    return {
        "family": CHALLENGE_FAMILY,
        "family_full_name": FAMILY_FULL_NAMES.get(CHALLENGE_FAMILY, "Challenge"),
        "n_tasks": len(challenge_tasks),
        "mean_success_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "tasks": [
            {
                "benchmark_task_id": t.benchmark_task_id,
                "template": t.template,
                "success_rate": round(t.success_rate, 4),
                "mean_steps": round(t.mean_steps, 2),
            }
            for t in challenge_tasks
        ],
    }


# ---------------------------------------------------------------------------
# Full analysis report
# ---------------------------------------------------------------------------


def generate_full_analysis(result: BenchmarkResult) -> dict[str, Any]:
    """Produce the complete analysis report as a JSON-serialisable dict."""
    return {
        "benchmark_name": result.benchmark_name,
        "policy_name": result.policy_name,
        "policy_mode": result.policy_mode,
        "overall_success_rate": round(result.overall_success_rate, 4),
        "overall_degradation": round(result.overall_degradation, 4),
        "family_diagnosis": diagnose_families(result),
        "template_comparison": compare_templates(result),
        "challenge_suite": report_challenge_suite(result),
    }


def write_analysis_report(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Write the full analysis report to ``<output_dir>/analysis_report.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "analysis_report.json"
    report = generate_full_analysis(result)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report_path


# ---------------------------------------------------------------------------
# Text summary (for CLI / logging)
# ---------------------------------------------------------------------------


def write_markdown_report(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Write a human-readable ``report.md`` to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.md"

    lines: list[str] = []
    lines.append(f"# {result.benchmark_name}")
    lines.append("")
    lines.append(f"**Policy:** {result.policy_name} ({result.policy_mode})")
    lines.append(f"**Overall success rate:** {result.overall_success_rate:.2%}")
    lines.append(f"**Overall degradation:** {result.overall_degradation:+.2%}")
    lines.append("")

    # Diagnostic families table
    diag = diagnose_families(result)
    if diag:
        lines.append("## Diagnostic Families")
        lines.append("")
        lines.append("| Family | Full Name | Success Rate | Degradation | Tasks |")
        lines.append("|--------|-----------|-------------|-------------|-------|")
        for d in diag:
            lines.append(
                f"| {d['family']} | {d['family_full_name']} "
                f"| {d['mean_success_rate']:.2%} "
                f"| {d['mean_degradation']:+.2%} "
                f"| {d['n_tasks']} |"
            )
        lines.append("")

    # Template comparison table
    comparison = compare_templates(result)
    if comparison:
        lines.append("## Template Comparison")
        lines.append("")
        # Collect all families that appear
        all_fams = sorted({f for c in comparison for f in c["family_success_rates"]})
        header = "| Template | Full Name | " + " | ".join(all_fams) + " | Mean | Worst |"
        sep = "|----------|-----------|" + "|".join(["------"] * len(all_fams)) + "|------|-------|"
        lines.append(header)
        lines.append(sep)
        for c in comparison:
            fam_cols = " | ".join(
                f"{c['family_success_rates'].get(f, 0.0):.2%}" for f in all_fams
            )
            lines.append(
                f"| {c['template']} | {c['template_full_name']} "
                f"| {fam_cols} "
                f"| {c['mean_success_rate']:.2%} "
                f"| {c['worst_family']} |"
            )
        lines.append("")

    # Challenge suite
    challenge = report_challenge_suite(result)
    if challenge.get("n_tasks", 0) > 0:
        lines.append("## Challenge Suite (R)")
        lines.append("")
        lines.append(f"**Mean success rate:** {challenge['mean_success_rate']:.2%}")
        lines.append("")
        lines.append("| Task | Template | Success Rate | Mean Steps |")
        lines.append("|------|----------|-------------|------------|")
        for t in challenge.get("tasks", []):
            lines.append(
                f"| {t['benchmark_task_id']} | {t['template']} "
                f"| {t['success_rate']:.2%} "
                f"| {t['mean_steps']:.0f} |"
            )
        lines.append("")

    # Per-task details
    lines.append("## Per-Task Results")
    lines.append("")
    lines.append("| Task ID | Family | Template | Nominal | Perturbed | Degradation |")
    lines.append("|---------|--------|----------|---------|-----------|-------------|")
    for tr in result.task_results:
        lines.append(
            f"| {tr.benchmark_task_id} | {tr.family} | {tr.template} "
            f"| {tr.nominal_success_rate:.2%} "
            f"| {tr.perturbed_success_rate:.2%} "
            f"| {tr.degradation:+.2%} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def format_text_summary(result: BenchmarkResult) -> str:
    """Return a human-readable multi-line summary of the benchmark run."""
    lines: list[str] = []
    lines.append(f"=== {result.benchmark_name} ===")
    lines.append(f"Policy: {result.policy_name} ({result.policy_mode})")
    lines.append(f"Overall success rate: {result.overall_success_rate:.2%}")
    lines.append("")

    # Diagnostic families
    lines.append("--- Diagnostic Families ---")
    for fr in result.family_results:
        if fr.family == CHALLENGE_FAMILY:
            continue
        full_name = fr.family_full_name or FAMILY_FULL_NAMES.get(fr.family, fr.family)
        lines.append(f"  {fr.family} ({full_name}): success={fr.mean_success_rate:.2%}, degradation={fr.mean_degradation:+.2%}")
        for tmpl, rate in sorted(fr.template_breakdown.items()):
            tmpl_name = TEMPLATE_FULL_NAMES.get(tmpl, tmpl)
            lines.append(f"    {tmpl} ({tmpl_name}): {rate:.2%}")

    # Challenge suite
    challenge = [fr for fr in result.family_results if fr.family == CHALLENGE_FAMILY]
    if challenge:
        lines.append("")
        lines.append("--- Challenge Suite (R) ---")
        for fr in challenge:
            lines.append(f"  R ({fr.family_full_name}): success={fr.mean_success_rate:.2%}")

    return "\n".join(lines)
