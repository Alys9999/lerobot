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

"""CLI tool to regenerate analysis reports from an existing benchmark run.

Usage::

    python -m lerobot.scripts.lerobot_hidden_physics_report \\
        --result_dir outputs/hidden_physics/run_123456
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate Hidden-Physics benchmark analysis reports.")
    p.add_argument("--result_dir", type=str, required=True, help="Path to an existing benchmark run output.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result_dir = Path(args.result_dir)

    summary_path = result_dir / "benchmark_summary.json"
    if not summary_path.exists():
        logger.error("No benchmark_summary.json found at %s", result_dir)
        sys.exit(1)

    # Re-hydrate BenchmarkResult
    from lerobot.runtime.hidden_physics_results import (
        BenchmarkResult,
        EpisodeResult,
        FamilyResult,
        TaskResult,
    )

    with summary_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    task_results = []
    for tr_raw in raw.get("task_results", []):
        eps = [EpisodeResult(**ep) for ep in tr_raw.pop("episode_results", [])]
        task_results.append(TaskResult(**tr_raw, episode_results=eps))

    family_results = [FamilyResult(**fr) for fr in raw.get("family_results", [])]

    result = BenchmarkResult(
        benchmark_name=raw.get("benchmark_name", ""),
        policy_name=raw.get("policy_name", ""),
        policy_mode=raw.get("policy_mode", ""),
        task_results=task_results,
        family_results=family_results,
        overall_success_rate=raw.get("overall_success_rate", 0.0),
        overall_degradation=raw.get("overall_degradation", 0.0),
    )

    from lerobot.runtime.hidden_physics_analysis import (
        format_text_summary,
        generate_full_analysis,
        write_analysis_report,
        write_markdown_report,
    )

    report_path = write_analysis_report(result, result_dir)
    logger.info("Analysis report written to: %s", report_path)

    md_path = write_markdown_report(result, result_dir)
    logger.info("Markdown report written to: %s", md_path)

    print("\n" + format_text_summary(result))


if __name__ == "__main__":
    main()
