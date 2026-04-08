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

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from lerobot.configs import parser
from lerobot.runtime.trace import read_episode_trace_summary
from lerobot.utils.utils import init_logging

LOGGER = logging.getLogger(__name__)

OVERALL_COLUMNS = [
    "arm",
    "episodes",
    "success_rate",
    "mean_attempt_count",
    "median_attempt_count",
    "episodes_with_failure",
    "recovered_after_failure_rate",
    "mean_first_failure_to_success_s",
    "p50_first_failure_to_success_s",
    "avg_steps",
    "avg_latency_ms",
    "num_exceptions",
]

PER_TASK_COLUMNS = [
    "arm",
    "suite",
    "task_id",
    "task_name",
    "episodes",
    "success_rate",
    "mean_attempt_count",
    "episodes_with_failure",
    "recovered_after_failure_rate",
    "mean_first_failure_to_success_s",
    "avg_steps",
    "avg_latency_ms",
    "num_exceptions",
]


@dataclass(slots=True)
class BowlRunComparisonConfig:
    results_root: str
    arms: list[str] = field(default_factory=lambda: ["aloha", "droid", "base"])
    summary_filename: str = "summary.json"
    output_dir: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_trace_path(summary_path: Path, trace_path: str | None) -> Path | None:
    if not trace_path:
        return None

    candidate = Path(trace_path)
    candidates = [candidate, summary_path.parent / candidate]
    for item in candidates:
        if item.exists():
            return item.resolve()
    return candidates[0]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _safe_float(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _render_markdown_table(title: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---:" if column not in {"arm", "suite", "task_name"} else "---" for column in columns) + " |"
    body = [
        "| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    if not body:
        body = ["| " + " | ".join("-" for _ in columns) + " |"]
    return "\n".join([f"## {title}", header, separator, *body])


def _read_episode_records(arm: str, summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    episodes = summary.get("episodes", [])
    records: list[dict[str, Any]] = []
    for episode in episodes:
        episode_metrics = dict(episode.get("metrics", {}))
        trace_path = _resolve_trace_path(summary_path, episode.get("trace_path"))
        attempt_analysis: dict[str, Any] = {}
        trace_summary: dict[str, Any] = {}
        if trace_path is not None and trace_path.exists():
            trace_summary = read_episode_trace_summary(trace_path)
            episode_metrics.update(
                {
                    key: value
                    for key, value in trace_summary.get("metrics", {}).items()
                    if key not in episode_metrics
                }
            )
            attempt_analysis = (
                trace_summary.get("metadata", {}).get("attempt_analysis", {})
                if isinstance(trace_summary.get("metadata"), dict)
                else {}
            )

        failed_attempt_count = _safe_float(episode_metrics.get("failed_attempt_count")) or 0.0
        recovered_after_failure = attempt_analysis.get("recovered_after_failure")
        if recovered_after_failure is None and failed_attempt_count > 0.0:
            recovered_after_failure = episode_metrics.get("first_failure_to_success_s") is not None

        records.append(
            {
                "arm": arm,
                "suite": episode.get("suite") or summary.get("suite"),
                "task_id": episode.get("task_id") if episode.get("task_id") is not None else summary.get("task_id"),
                "task_name": episode.get("task_name") or summary.get("task_name"),
                "episode_index": episode.get("episode_index"),
                "success": bool(episode.get("success", False)),
                "steps": _safe_float(episode_metrics.get("steps")),
                "avg_latency_ms": _safe_float(episode_metrics.get("avg_latency_ms")),
                "attempt_count": _safe_float(episode_metrics.get("attempt_count")),
                "failed_attempt_count": _safe_float(episode_metrics.get("failed_attempt_count")),
                "successful_attempt_count": _safe_float(episode_metrics.get("successful_attempt_count")),
                "first_attempt_to_success_s": _safe_float(episode_metrics.get("first_attempt_to_success_s")),
                "first_failure_to_success_s": _safe_float(episode_metrics.get("first_failure_to_success_s")),
                "recovered_after_failure": (
                    bool(recovered_after_failure) if recovered_after_failure is not None else None
                ),
                "exception": episode.get("exception"),
                "trace_path": str(trace_path) if trace_path is not None else None,
            }
        )
    return records


def _aggregate_records(records: list[dict[str, Any]], base_row: dict[str, Any]) -> dict[str, Any]:
    success_values = [1.0 if record["success"] else 0.0 for record in records]
    failure_records = [
        record for record in records if (_safe_float(record.get("failed_attempt_count")) or 0.0) > 0.0
    ]
    recovery_values = [
        1.0 if record.get("recovered_after_failure") else 0.0
        for record in failure_records
        if record.get("recovered_after_failure") is not None
    ]
    aggregated = dict(base_row)
    aggregated.update(
        {
            "episodes": len(records),
            "success_rate": _mean(success_values),
            "mean_attempt_count": _mean(_collect_numeric(records, "attempt_count")),
            "median_attempt_count": _median(_collect_numeric(records, "attempt_count")),
            "episodes_with_failure": len(failure_records),
            "recovered_after_failure_rate": _mean(recovery_values),
            "mean_first_failure_to_success_s": _mean(_collect_numeric(records, "first_failure_to_success_s")),
            "p50_first_failure_to_success_s": _median(
                _collect_numeric(records, "first_failure_to_success_s")
            ),
            "avg_steps": _mean(_collect_numeric(records, "steps")),
            "avg_latency_ms": _mean(_collect_numeric(records, "avg_latency_ms")),
            "num_exceptions": sum(1 for record in records if record.get("exception")),
        }
    )
    return aggregated


def _build_overall_rows(records: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_records = [record for record in records if record["arm"] == arm]
        rows.append(_aggregate_records(arm_records, {"arm": arm}))
    return rows


def _build_per_task_rows(records: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        suite = "" if record.get("suite") is None else str(record["suite"])
        task_id = "" if record.get("task_id") is None else str(record["task_id"])
        task_name = "" if record.get("task_name") is None else str(record["task_name"])
        key = (record["arm"], suite, task_id, task_name)
        grouped.setdefault(key, []).append(record)

    arm_order = {arm: index for index, arm in enumerate(arms)}
    sorted_keys = sorted(
        grouped,
        key=lambda item: (
            item[1],
            int(item[2]) if item[2].isdigit() else item[2],
            arm_order.get(item[0], len(arms)),
            item[3],
        ),
    )

    rows: list[dict[str, Any]] = []
    for arm, suite, task_id, task_name in sorted_keys:
        rows.append(
            _aggregate_records(
                grouped[(arm, suite, task_id, task_name)],
                {
                    "arm": arm,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                },
            )
        )
    return rows


def compare_bowl_runs(cfg: BowlRunComparisonConfig) -> dict[str, Any]:
    results_root = Path(cfg.results_root)
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else results_root / "comparison"

    records: list[dict[str, Any]] = []
    summary_paths: dict[str, str] = {}
    for arm in cfg.arms:
        summary_path = results_root / arm / cfg.summary_filename
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for arm '{arm}': {summary_path}")
        summary_paths[arm] = str(summary_path.resolve())
        records.extend(_read_episode_records(arm, summary_path))

    overall_rows = _build_overall_rows(records, cfg.arms)
    per_task_rows = _build_per_task_rows(records, cfg.arms)

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_csv_path = output_dir / "overall.csv"
    per_task_csv_path = output_dir / "per_task.csv"
    markdown_path = output_dir / "comparison.md"
    json_path = output_dir / "comparison.json"

    _write_csv(overall_csv_path, OVERALL_COLUMNS, overall_rows)
    _write_csv(per_task_csv_path, PER_TASK_COLUMNS, per_task_rows)

    markdown_text = "\n\n".join(
        [
            "# Bowl Run Comparison",
            _render_markdown_table("Overall", OVERALL_COLUMNS, overall_rows),
            _render_markdown_table("Per Task", PER_TASK_COLUMNS, per_task_rows),
        ]
    )
    markdown_path.write_text(markdown_text + "\n", encoding="utf-8")

    result = {
        "results_root": str(results_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "summary_paths": summary_paths,
        "overall": overall_rows,
        "per_task": per_task_rows,
        "files": {
            "overall_csv": str(overall_csv_path.resolve()),
            "per_task_csv": str(per_task_csv_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "json": str(json_path.resolve()),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    LOGGER.info("Wrote bowl run comparison to %s", output_dir)
    print(markdown_text)
    return result


@parser.wrap()
def compare_bowl_runs_main(cfg: BowlRunComparisonConfig) -> dict[str, Any]:
    return compare_bowl_runs(cfg)


def main() -> None:
    init_logging()
    compare_bowl_runs_main()


if __name__ == "__main__":
    main()
