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

"""Benchmark runner for the Hidden-Physics Diagnostic Benchmark v1.

Orchestrates:
  1. Loading the benchmark config and catalog.
  2. For each selected task: creating the env via the standard LIBERO API,
     setting up variations, running episodes via a :class:`PolicyExecutor`,
     and collecting results.
  3. Running *nominal* (unperturbed) episodes first, then *perturbed*
     episodes, so degradation can be computed.
  4. Aggregating episode -> task -> family -> benchmark results.
  5. Persisting traces, videos, and summaries.

The runner **never** contains policy-specific codec logic — that lives in
:mod:`hidden_physics_executor`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import EpisodeTrace
from .hidden_physics_catalog import (
    BUILTIN_PROFILES,
    BenchmarkTaskDefinition,
    filter_catalog_from_config,
    get_variation_profile_for_task,
    load_catalog,
)
from .hidden_physics_config import IID_OOD_LEVELS, HiddenPhysicsBenchmarkConfig
from .hidden_physics_executor import PolicyExecutor
from .hidden_physics_results import (
    BenchmarkResult,
    EpisodeResult,
    TaskResult,
    aggregate_benchmark_result,
    aggregate_task_result,
    write_benchmark_result,
)
from .trace import write_episode_trace
from .variation import VariationProfile, build_family_variation_profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env contract validation
# ---------------------------------------------------------------------------


class EnvContractError(RuntimeError):
    """Raised when a LIBERO env does not satisfy the expected contract."""


def _validate_env_contract(env: Any, task: BenchmarkTaskDefinition) -> None:
    """Check that *env* exposes the API surface required by the runner.

    The runner expects:
      - ``env.reset(seed=...)`` returning ``(obs, info)``
      - ``env.step(action)`` returning ``(obs, reward, terminated, truncated, info)``
      - ``info`` dict containing an ``"is_success"`` key
      - ``env.close()``
      - ``env.set_variation_profile(profile, seed)`` and
        ``env.clear_variation_profile()`` (for variation support)

    This function performs a lightweight structural check (attribute presence)
    rather than a full behavioural test.
    """
    required_methods = ("reset", "step", "close")
    for method in required_methods:
        if not callable(getattr(env, method, None)):
            raise EnvContractError(
                f"Env for task '{task.benchmark_task_id}' is missing "
                f"required method '{method}'."
            )

    variation_methods = ("set_variation_profile", "clear_variation_profile")
    for method in variation_methods:
        if not callable(getattr(env, method, None)):
            logger.warning(
                "Env for task '%s' is missing variation method '%s'. "
                "Perturbed episodes will have no effect.",
                task.benchmark_task_id,
                method,
            )


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def _write_config_json(config: Any, output_dir: Path) -> Path:
    """Write the benchmark config to ``config.json`` in *output_dir*.

    Uses :func:`dataclasses.asdict` for serialisation.  Non-serialisable
    values are coerced to ``str`` as a fallback.
    """
    config_path = output_dir / "config.json"
    try:
        payload = asdict(config)
    except Exception:
        payload = {"error": "Could not serialise config", "repr": repr(config)}

    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return config_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_variation_profile_for_task(
    task: BenchmarkTaskDefinition,
    level: str = "iid_high",
) -> VariationProfile | None:
    """Build a concrete :class:`VariationProfile` for a benchmark task at a given level."""
    profile_def = get_variation_profile_for_task(task)
    if profile_def is None:
        return None

    ranges = profile_def.ranges_for_level(level)
    if not ranges:
        return None

    # Use the per-task variation_target from the catalog entry.  This ensures
    # each template applies variations to the correct object (e.g. frying_pan
    # for T3, book for T4) rather than always targeting "bowl".
    target = task.variation_target

    return build_family_variation_profile(
        family=profile_def.family,
        profile_name=f"{profile_def.profile_name}_{level}",
        target=target,
        ranges=ranges,
    )


def _make_single_env(
    task: BenchmarkTaskDefinition,
    env_overrides: dict[str, Any] | None = None,
) -> Any:
    """Create a single (non-vectorised) LIBERO env for the given task.

    Uses the standard LIBERO ``_get_suite`` → ``LiberoEnv`` path so that
    the benchmark runner stays aligned with the rest of the LeRobot env API.
    """
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(task.suite_name)

    kwargs: dict[str, Any] = {
        "task_suite": suite,
        "task_id": task.base_task_id,
        "task_suite_name": task.suite_name,
    }
    if env_overrides:
        kwargs.update(env_overrides)

    return LiberoEnv(**kwargs)


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------


def rollout_one_episode(
    env: Any,
    executor: PolicyExecutor,
    task: BenchmarkTaskDefinition,
    variation_profile: VariationProfile | None,
    seed: int,
    max_steps: int,
    write_trace_dir: Path | None = None,
    episode_index: int = 0,
    record_video: bool = False,
    policy_mode: str = "",
) -> tuple[EpisodeTrace, EpisodeResult]:
    """Run a single episode and return (trace, result)."""
    _policy_mode = policy_mode

    # Apply variation (set before reset; reset triggers sample+apply)
    sampled_variation: dict[str, float] = {}
    if variation_profile is not None:
        env.set_variation_profile(variation_profile, seed)
    else:
        env.clear_variation_profile()

    # Inject benchmark metadata so it appears in every env info dict
    # (plan §8.1: LiberoEnv should stably record family / template /
    # benchmark_task_id alongside sampled variation values).
    if hasattr(env, "set_episode_metadata"):
        env.set_episode_metadata(
            {
                "benchmark_task_id": task.benchmark_task_id,
                "family": task.family,
                "template": task.template,
            }
        )

    # Reset
    obs, info = env.reset(seed=seed)
    if variation_profile is not None and hasattr(env, "last_variation_sample"):
        sampled_variation = dict(env.last_variation_sample or {})

    executor.reset()

    trace = EpisodeTrace(
        metadata={
            "benchmark_task_id": task.benchmark_task_id,
            "family": task.family,
            "template": task.template,
            # Canonical names (architecture spec)
            "suite": task.suite_name,
            "task_id": task.base_task_id,
            # Legacy aliases for backward compatibility
            "suite_name": task.suite_name,
            "base_task_id": task.base_task_id,
            "task_prompt": task.prompt,
            "prompt": task.prompt,  # legacy alias
            "variation_target": task.variation_target,
            "variation": sampled_variation,
            "seed": seed,
            "is_nominal": variation_profile is None,
            "policy_mode": _policy_mode,
        }
    )

    video_frames: list[np.ndarray] = []
    success = False
    step = 0
    t0 = time.time()

    for step in range(max_steps):
        action = executor.infer(obs, task_text=task.prompt, control_dt=0.02)
        obs, reward, terminated, truncated, info = env.step(action)

        trace.rewards.append(float(reward))
        trace.dones.append(bool(terminated or truncated))

        if record_video and hasattr(env, "render"):
            try:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
            except Exception:
                pass

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    elapsed = time.time() - t0
    trace.success = success
    trace.metrics["elapsed_s"] = elapsed
    trace.metrics["steps"] = step + 1

    # Persist trace
    trace_path = ""
    if write_trace_dir is not None:
        tp = write_episode_trace(trace, write_trace_dir, episode_index)
        trace_path = str(tp)

    # Persist video
    video_path = ""
    if record_video and video_frames and write_trace_dir is not None:
        video_dir = write_trace_dir.parent / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        vp = video_dir / f"episode_{episode_index:03d}.mp4"
        try:
            from lerobot.utils.io_utils import write_video
            write_video(str(vp), video_frames, fps=30)
            video_path = str(vp)
        except Exception:
            logger.debug("Could not write video for episode %d.", episode_index)

    result = EpisodeResult(
        benchmark_task_id=task.benchmark_task_id,
        family=task.family,
        template=task.template,
        episode_index=episode_index,
        success=success,
        steps=step + 1,
        elapsed_s=elapsed,
        variation=sampled_variation,
        trace_path=trace_path,
        video_path=video_path,
    )

    return trace, result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


class HiddenPhysicsRunner:
    """Drives a full benchmark run."""

    def __init__(self, config: HiddenPhysicsBenchmarkConfig) -> None:
        self.config = config

    def _run_episodes(
        self,
        env: Any,
        executor: PolicyExecutor,
        task: BenchmarkTaskDefinition,
        variation_profile: VariationProfile | None,
        n_episodes: int,
        base_seed: int,
        max_steps: int,
        trace_dir: Path | None,
        record_video: bool,
        episode_offset: int = 0,
    ) -> list[EpisodeResult]:
        """Run *n_episodes* for a single task/variation combination."""
        results: list[EpisodeResult] = []
        for ep in range(n_episodes):
            ep_seed = base_seed + ep
            try:
                _trace, ep_result = rollout_one_episode(
                    env=env,
                    executor=executor,
                    task=task,
                    variation_profile=variation_profile,
                    seed=ep_seed,
                    max_steps=max_steps,
                    write_trace_dir=trace_dir,
                    episode_index=episode_offset + ep,
                    record_video=record_video,
                    policy_mode=self.config.policy.mode,
                )
                results.append(ep_result)
                label = "nominal" if variation_profile is None else "perturbed"
                logger.info(
                    "  [%s] Episode %d/%d — success=%s, steps=%d",
                    label, ep + 1, n_episodes, ep_result.success, ep_result.steps,
                )
            except Exception:
                logger.exception("Episode %d failed for task %s.", ep, task.benchmark_task_id)
                if self.config.runtime.fail_fast:
                    raise
        return results

    @staticmethod
    def _resolve_perturbation_levels(
        config: HiddenPhysicsBenchmarkConfig,
        task: BenchmarkTaskDefinition,
    ) -> list[str]:
        """Return the list of perturbation levels to evaluate for *task*.

        If the config specifies ``iid_ood_levels``, those are used (excluding
        ``"nominal"`` which is always run separately).  Otherwise we default
        to both the task's ``iid_level`` and ``ood_level``.
        """
        if config.iid_ood_levels:
            return [lv for lv in config.iid_ood_levels if lv != "nominal"]
        # Default: evaluate both iid and ood levels defined on the task.
        levels: list[str] = []
        if task.iid_level and task.iid_level != "nominal":
            levels.append(task.iid_level)
        if task.ood_level and task.ood_level != "nominal" and task.ood_level not in levels:
            levels.append(task.ood_level)
        return levels or ["iid_high"]

    def run(self, executor: PolicyExecutor) -> BenchmarkResult:
        """Execute the benchmark and return a :class:`BenchmarkResult`.

        For each task the runner:
          1. Runs *nominal* episodes (no variation).
          2. For each perturbation level (iid_low, iid_high, ood_low,
             ood_high — filtered by config) runs *perturbed* episodes.
          3. Computes degradation per level.

        The ``TaskResult.degradation`` is computed as the *worst*
        perturbed success rate minus the nominal rate.
        """
        cfg = self.config
        rt = cfg.runtime

        # Output directory
        run_name = rt.run_name or f"run_{int(time.time())}"
        output_dir = Path(rt.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Persist config for reproducibility
        _write_config_json(cfg, output_dir)

        # Load & filter catalog
        catalog = load_catalog(cfg.catalog_path or None)
        tasks = filter_catalog_from_config(catalog, cfg)
        logger.info("Benchmark '%s': %d tasks selected.", cfg.benchmark_name, len(tasks))

        # Run each task
        all_task_results: list[TaskResult] = []

        for task in tasks:
            logger.info("Task %s [%s / %s]", task.benchmark_task_id, task.family, task.template)
            task_trace_dir = output_dir / "traces" / task.benchmark_task_id if rt.write_trace else None

            # Resolve which perturbation levels to evaluate
            perturbation_levels = self._resolve_perturbation_levels(cfg, task)

            # Create env
            try:
                env = _make_single_env(task, cfg.env_overrides)
            except Exception:
                logger.exception("Failed to create env for task %s — skipping.", task.benchmark_task_id)
                if rt.fail_fast:
                    raise
                continue

            # Validate env contract
            try:
                _validate_env_contract(env, task)
            except EnvContractError:
                logger.exception("Env contract violated for task %s — skipping.", task.benchmark_task_id)
                if rt.fail_fast:
                    raise
                env.close()
                continue

            # --- Nominal episodes (no variation) ---
            nominal_results = self._run_episodes(
                env=env,
                executor=executor,
                task=task,
                variation_profile=None,
                n_episodes=rt.n_episodes_per_task,
                base_seed=rt.seed,
                max_steps=rt.max_steps,
                trace_dir=(task_trace_dir / "nominal") if task_trace_dir else None,
                record_video=rt.write_video,
            )
            nominal_rate = (
                sum(e.success for e in nominal_results) / len(nominal_results)
                if nominal_results else 0.0
            )

            # Tag nominal episodes
            for ep in nominal_results:
                ep.iid_ood_level = "nominal"
                ep.seed_group = task.seed_group

            # --- Perturbed episodes at each level ---
            all_perturbed_results: list[EpisodeResult] = []
            level_rates: dict[str, float] = {}
            ep_offset = rt.n_episodes_per_task
            seed_offset = 10000

            for level in perturbation_levels:
                vp = _build_variation_profile_for_task(task, level=level)
                level_results = self._run_episodes(
                    env=env,
                    executor=executor,
                    task=task,
                    variation_profile=vp,
                    n_episodes=rt.n_episodes_per_task,
                    base_seed=rt.seed + seed_offset,
                    max_steps=rt.max_steps,
                    trace_dir=(task_trace_dir / level) if task_trace_dir else None,
                    record_video=rt.write_video,
                    episode_offset=ep_offset,
                )
                # Tag perturbed episodes with level and seed group
                for ep in level_results:
                    ep.iid_ood_level = level
                    ep.seed_group = task.seed_group

                rate = (
                    sum(e.success for e in level_results) / len(level_results)
                    if level_results else 0.0
                )
                level_rates[level] = rate
                all_perturbed_results.extend(level_results)
                ep_offset += rt.n_episodes_per_task
                seed_offset += 10000

                logger.info(
                    "  Level %s — success=%.2f, degradation=%+.2f",
                    level, rate, rate - nominal_rate,
                )

            # Aggregate: use the worst perturbed level for the headline
            worst_perturbed_rate = (
                min(level_rates.values()) if level_rates else 0.0
            )

            all_episodes = nominal_results + all_perturbed_results
            task_result = aggregate_task_result(
                benchmark_task_id=task.benchmark_task_id,
                family=task.family,
                template=task.template,
                suite_name=task.suite_name,
                base_task_id=task.base_task_id,
                episodes=all_episodes,
            )
            task_result.nominal_success_rate = nominal_rate
            task_result.perturbed_success_rate = worst_perturbed_rate
            task_result.degradation = worst_perturbed_rate - nominal_rate
            task_result.level_success_rates = dict(level_rates)

            all_task_results.append(task_result)
            logger.info(
                "  Task %s — nominal=%.2f, worst_perturbed=%.2f, degradation=%+.2f",
                task.benchmark_task_id, nominal_rate, worst_perturbed_rate, task_result.degradation,
            )

            env.close()

        # Aggregate benchmark result
        benchmark_result = aggregate_benchmark_result(
            benchmark_name=cfg.benchmark_name,
            task_results=all_task_results,
            policy_name=cfg.policy.native_policy_path or cfg.policy.adapter_endpoint,
            policy_mode=cfg.policy.mode,
        )

        # Persist data summaries
        paths = write_benchmark_result(benchmark_result, output_dir)

        # Write analysis and markdown reports (same as the CLI does, so
        # programmatic users also get report.md and analysis_report.json).
        try:
            from .hidden_physics_analysis import write_analysis_report, write_markdown_report
            analysis_path = write_analysis_report(benchmark_result, output_dir)
            paths["analysis_report"] = analysis_path
            md_path = write_markdown_report(benchmark_result, output_dir)
            paths["report_md"] = md_path
        except Exception:
            logger.debug("Could not write analysis/markdown reports.")

        benchmark_result.report_paths = {k: str(v) for k, v in paths.items()}
        logger.info("Benchmark complete. Results at: %s", output_dir)

        return benchmark_result
