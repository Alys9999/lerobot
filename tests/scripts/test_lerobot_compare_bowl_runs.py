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
from pathlib import Path

import numpy as np

from lerobot.scripts import lerobot_compare_bowl_runs as compare


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_trace_payload(*, recovered_after_failure: bool, events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "success": any(event.get("outcome") == "succeeded" for event in events),
        "metrics": {},
        "metadata": {
            "attempt_analysis": {
                "detector_version": "bowl_attempt_v1",
                "enabled": True,
                "signal_available": True,
                "recovered_after_failure": recovered_after_failure,
                "events": events,
            }
        },
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
    }


def _episode_summary(
    *,
    episode_index: int,
    task_id: int,
    success: bool,
    metrics: dict[str, float],
    trace_path: str,
    exception: str | None = None,
) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "task_episode_index": episode_index,
        "suite": "libero_spatial",
        "task_id": task_id,
        "task_name": "pick_up_the_bowl",
        "task_prompt": "pick up the bowl",
        "success": success,
        "metrics": metrics,
        "trace_path": trace_path,
        "video_path": None,
        "variation": {},
        "exception": exception,
    }


def test_compare_bowl_runs_writes_overall_and_per_task_tables(tmp_path, capsys):
    results_root = tmp_path / "runs"

    aloha_dir = results_root / "aloha"
    _write_json(
        aloha_dir / "traces" / "episode_000.json",
        _make_trace_payload(
            recovered_after_failure=True,
            events=[
                {"attempt_index": 1, "outcome": "failed"},
                {"attempt_index": 2, "outcome": "succeeded"},
            ],
        ),
    )
    _write_json(
        aloha_dir / "traces" / "episode_001.json",
        _make_trace_payload(
            recovered_after_failure=False,
            events=[{"attempt_index": 1, "outcome": "succeeded"}],
        ),
    )
    _write_json(
        aloha_dir / "summary.json",
        {
            "episodes": [
                _episode_summary(
                    episode_index=0,
                    task_id=0,
                    success=True,
                    metrics={
                        "steps": 10.0,
                        "avg_latency_ms": 20.0,
                        "attempt_count": 2.0,
                        "failed_attempt_count": 1.0,
                        "successful_attempt_count": 1.0,
                        "first_failure_to_success_s": 0.4,
                    },
                    trace_path="traces/episode_000.json",
                ),
                _episode_summary(
                    episode_index=1,
                    task_id=0,
                    success=True,
                    metrics={
                        "steps": 8.0,
                        "avg_latency_ms": 18.0,
                        "attempt_count": 1.0,
                        "failed_attempt_count": 0.0,
                        "successful_attempt_count": 1.0,
                    },
                    trace_path="traces/episode_001.json",
                ),
            ]
        },
    )

    droid_dir = results_root / "droid"
    _write_json(
        droid_dir / "traces" / "episode_000.json",
        _make_trace_payload(
            recovered_after_failure=False,
            events=[{"attempt_index": 1, "outcome": "failed"}],
        ),
    )
    _write_json(
        droid_dir / "summary.json",
        {
            "episodes": [
                _episode_summary(
                    episode_index=0,
                    task_id=0,
                    success=False,
                    metrics={
                        "steps": 12.0,
                        "avg_latency_ms": 25.0,
                        "attempt_count": 1.0,
                        "failed_attempt_count": 1.0,
                        "successful_attempt_count": 0.0,
                    },
                    trace_path="traces/episode_000.json",
                    exception="RuntimeError('policy timeout')",
                )
            ]
        },
    )

    base_dir = results_root / "base"
    _write_json(
        base_dir / "traces" / "episode_000.json",
        _make_trace_payload(
            recovered_after_failure=False,
            events=[{"attempt_index": 1, "outcome": "succeeded"}],
        ),
    )
    _write_json(
        base_dir / "summary.json",
        {
            "episodes": [
                _episode_summary(
                    episode_index=0,
                    task_id=0,
                    success=True,
                    metrics={
                        "steps": 9.0,
                        "avg_latency_ms": 19.0,
                        "attempt_count": 1.0,
                        "failed_attempt_count": 0.0,
                        "successful_attempt_count": 1.0,
                    },
                    trace_path="traces/episode_000.json",
                )
            ]
        },
    )

    result = compare.compare_bowl_runs(
        compare.BowlRunComparisonConfig(results_root=str(results_root))
    )

    overall_by_arm = {row["arm"]: row for row in result["overall"]}
    assert np.isclose(overall_by_arm["aloha"]["success_rate"], 1.0)
    assert np.isclose(overall_by_arm["aloha"]["mean_attempt_count"], 1.5)
    assert np.isclose(overall_by_arm["aloha"]["median_attempt_count"], 1.5)
    assert overall_by_arm["aloha"]["episodes_with_failure"] == 1
    assert np.isclose(overall_by_arm["aloha"]["recovered_after_failure_rate"], 1.0)
    assert np.isclose(overall_by_arm["aloha"]["mean_first_failure_to_success_s"], 0.4)
    assert np.isclose(overall_by_arm["aloha"]["avg_steps"], 9.0)
    assert np.isclose(overall_by_arm["aloha"]["avg_latency_ms"], 19.0)

    assert np.isclose(overall_by_arm["droid"]["success_rate"], 0.0)
    assert overall_by_arm["droid"]["num_exceptions"] == 1
    assert np.isclose(overall_by_arm["droid"]["recovered_after_failure_rate"], 0.0)

    assert np.isclose(overall_by_arm["base"]["success_rate"], 1.0)
    assert overall_by_arm["base"]["episodes_with_failure"] == 0
    assert overall_by_arm["base"]["recovered_after_failure_rate"] is None

    per_task_rows = result["per_task"]
    assert len(per_task_rows) == 3
    assert [row["arm"] for row in per_task_rows] == ["aloha", "droid", "base"]

    markdown_path = Path(result["files"]["markdown"])
    assert markdown_path.exists()
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "## Overall" in markdown_text
    assert "## Per Task" in markdown_text
    assert "| aloha | 2 | 1 | 1.5 | 1.5 | 1 | 1 | 0.4 | 0.4 | 9 | 19 | 0 |" in markdown_text

    overall_csv_path = Path(result["files"]["overall_csv"])
    per_task_csv_path = Path(result["files"]["per_task_csv"])
    json_path = Path(result["files"]["json"])
    assert overall_csv_path.exists()
    assert per_task_csv_path.exists()
    assert json_path.exists()

    with overall_csv_path.open("r", encoding="utf-8", newline="") as handle:
        overall_rows = list(csv.DictReader(handle))
    assert overall_rows[0]["arm"] == "aloha"
    assert overall_rows[1]["arm"] == "droid"
    assert overall_rows[2]["arm"] == "base"

    stdout = capsys.readouterr().out
    assert "# Bowl Run Comparison" in stdout
