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

"""CLI entry-point for the Hidden-Physics Diagnostic Benchmark v1.

Usage (YAML + CLI overrides)::

    python -m lerobot.scripts.lerobot_hidden_physics_eval \
        --config_path configs/benchmark/hidden_physics/hidden_physics_v1_smoke.yaml \
        --families F M \
        --n_episodes_per_task 5

Usage (draccus-style, when no --config_path)::

    python -m lerobot.scripts.lerobot_hidden_physics_eval \
        --benchmark_name hidden_physics_v1 \
        --families F \
        --runtime.n_episodes_per_task 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hidden-Physics Diagnostic Benchmark v1 — evaluation runner",
    )
    p.add_argument("--config_path", type=str, default="", help="Path to a YAML benchmark config.")
    p.add_argument("--families", nargs="*", default=None, help="Filter: benchmark families (F M C P R).")
    p.add_argument("--templates", nargs="*", default=None, help="Filter: task templates (T1 T2 …).")
    p.add_argument("--task_ids", nargs="*", default=None, help="Filter: specific benchmark task IDs.")
    p.add_argument("--iid_ood_levels", nargs="*", default=None,
                   help="Filter: perturbation levels (iid_low iid_high ood_low ood_high).")
    p.add_argument("--seed_groups", nargs="*", default=None, help="Filter: seed groups (sg1 …).")
    p.add_argument("--n_episodes_per_task", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--policy_mode", type=str, default=None, choices=["native", "openpi_adapter"])
    p.add_argument("--native_policy_path", type=str, default=None)
    p.add_argument("--adapter_endpoint", type=str, default=None)
    p.add_argument("--fail_fast", action="store_true", default=False)
    p.add_argument("--write_trace", action="store_true", default=None)
    p.add_argument("--no_write_trace", dest="write_trace", action="store_false")
    p.add_argument("--write_video", action="store_true", default=None)
    p.add_argument("--device", type=str, default=None, help="Device for policy (cpu, cuda, cuda:0, …).")
    return p.parse_args(argv)


def _load_config_from_yaml(path: str):
    """Load a :class:`HiddenPhysicsBenchmarkConfig` from a YAML file."""
    from lerobot.runtime.hidden_physics_config import (
        HiddenPhysicsBenchmarkConfig,
        HiddenPhysicsRuntimeConfig,
        PolicyModeConfig,
    )

    cfg = HiddenPhysicsBenchmarkConfig()

    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    for key in ("benchmark_name", "catalog_path", "env_type"):
        if key in raw:
            setattr(cfg, key, raw[key])
    for key in ("families", "templates", "iid_ood_levels", "seed_groups", "task_ids"):
        if key in raw:
            setattr(cfg, key, list(raw[key]))

    rt_raw = raw.get("runtime", {})
    cfg.runtime = HiddenPhysicsRuntimeConfig(
        n_episodes_per_task=rt_raw.get("n_episodes_per_task", 10),
        max_steps=rt_raw.get("max_steps", 400),
        write_trace=rt_raw.get("write_trace", True),
        write_video=rt_raw.get("write_video", False),
        output_dir=rt_raw.get("output_dir", "outputs/hidden_physics"),
        run_name=rt_raw.get("run_name", ""),
        fail_fast=rt_raw.get("fail_fast", False),
        seed=rt_raw.get("seed", 42),
        parallel_tasks=rt_raw.get("parallel_tasks", 1),
    )

    pol_raw = raw.get("policy", {})
    cfg.policy = PolicyModeConfig(
        mode=pol_raw.get("mode", "native"),
        native_policy_path=pol_raw.get("native_policy_path", ""),
        native_policy_overrides=dict(pol_raw.get("native_policy_overrides", {})),
        adapter_endpoint=pol_raw.get("adapter_endpoint", ""),
        adapter_spec=pol_raw.get("adapter_spec", ""),
    )
    return cfg


def _load_config_draccus():
    """Load a :class:`HiddenPhysicsBenchmarkConfig` via draccus CLI parsing."""
    import draccus
    from lerobot.runtime.hidden_physics_config import HiddenPhysicsBenchmarkConfig

    return draccus.parse(config_class=HiddenPhysicsBenchmarkConfig)


def _load_config(args: argparse.Namespace):
    """Build config from YAML, draccus, or defaults, then apply CLI overrides."""
    if args.config_path:
        cfg = _load_config_from_yaml(args.config_path)
    else:
        try:
            cfg = _load_config_draccus()
        except Exception:
            from lerobot.runtime.hidden_physics_config import HiddenPhysicsBenchmarkConfig
            cfg = HiddenPhysicsBenchmarkConfig()

    # Apply CLI overrides
    if args.families is not None:
        cfg.families = args.families
    if args.templates is not None:
        cfg.templates = args.templates
    if args.task_ids is not None:
        cfg.task_ids = args.task_ids
    if args.iid_ood_levels is not None:
        cfg.iid_ood_levels = args.iid_ood_levels
    if args.seed_groups is not None:
        cfg.seed_groups = args.seed_groups
    if args.n_episodes_per_task is not None:
        cfg.runtime.n_episodes_per_task = args.n_episodes_per_task
    if args.max_steps is not None:
        cfg.runtime.max_steps = args.max_steps
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    if args.output_dir is not None:
        cfg.runtime.output_dir = args.output_dir
    if args.run_name is not None:
        cfg.runtime.run_name = args.run_name
    if args.fail_fast:
        cfg.runtime.fail_fast = True
    if args.write_trace is not None:
        cfg.runtime.write_trace = args.write_trace
    if args.write_video is not None:
        cfg.runtime.write_video = args.write_video
    if args.policy_mode is not None:
        cfg.policy.mode = args.policy_mode
    if args.native_policy_path is not None:
        cfg.policy.native_policy_path = args.native_policy_path
    if args.adapter_endpoint is not None:
        cfg.policy.adapter_endpoint = args.adapter_endpoint

    return cfg


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args)

    logger.info("Hidden-Physics Diagnostic Benchmark v1")
    logger.info("  benchmark_name: %s", cfg.benchmark_name)
    logger.info("  families:       %s", cfg.families or "(all)")
    logger.info("  templates:      %s", cfg.templates or "(all)")
    logger.info("  policy_mode:    %s", cfg.policy.mode)
    logger.info("  episodes/task:  %d", cfg.runtime.n_episodes_per_task)

    from lerobot.runtime.hidden_physics_analysis import (
        format_text_summary,
        write_analysis_report,
        write_markdown_report,
    )
    from lerobot.runtime.hidden_physics_executor import build_executor
    from lerobot.runtime.hidden_physics_runner import HiddenPhysicsRunner

    device = getattr(args, "device", None) or "cuda"
    executor = build_executor(cfg.policy, device=device)
    runner = HiddenPhysicsRunner(cfg)

    try:
        result = runner.run(executor)
    finally:
        executor.close()

    # Analysis reports
    run_name = cfg.runtime.run_name or f"run_{int(time.time())}"
    output_dir = Path(cfg.runtime.output_dir) / run_name
    write_analysis_report(result, output_dir)
    write_markdown_report(result, output_dir)

    # Print summary
    print("\n" + format_text_summary(result))


if __name__ == "__main__":
    main()
