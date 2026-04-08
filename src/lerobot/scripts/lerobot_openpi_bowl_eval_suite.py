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
import logging
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import draccus

from lerobot.configs import parser
from lerobot.scripts.lerobot_compare_bowl_runs import BowlRunComparisonConfig, compare_bowl_runs
from lerobot.scripts.lerobot_openpi_bowl_smoke import OpenPIBowlSmokeConfig, run_bowl_smoke
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenPIBowlEvalScenarioConfig:
    name: str
    smoke_config_path: str
    server_policy_config: str
    checkpoint_dir: str
    enabled: bool = True


@dataclass(slots=True)
class OpenPIBowlEvalSuiteConfig:
    openpi_repo_root: str
    output_root: str = "outputs/openpi_bowl_eval_suite"
    host: str = "127.0.0.1"
    port: int = 8000
    server_launcher: list[str] = field(default_factory=lambda: ["uv", "run", "python"])
    server_script_path: str = "scripts/serve_policy.py"
    request_timeout_s: float | None = 120.0
    startup_timeout_s: float = 300.0
    startup_poll_interval_s: float = 1.0
    shutdown_timeout_s: float = 20.0
    write_video: bool = False
    write_trace: bool | None = None
    compare_after_run: bool = True
    continue_on_failure: bool = True
    scenarios: list[OpenPIBowlEvalScenarioConfig] = field(
        default_factory=lambda: [
            OpenPIBowlEvalScenarioConfig(
                name="aloha",
                smoke_config_path="configs/smoke/openpi_aloha_bowl_variation_smoke.yaml",
                server_policy_config="pi05_aloha",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_aloha",
            ),
            OpenPIBowlEvalScenarioConfig(
                name="droid",
                smoke_config_path="configs/smoke/openpi_droid_bowl_variation_smoke.yaml",
                server_policy_config="pi05_droid",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_droid",
            ),
            OpenPIBowlEvalScenarioConfig(
                name="base",
                smoke_config_path="configs/smoke/openpi_base_bowl_variation_smoke.yaml",
                server_policy_config="pi05_libero",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_base",
            ),
        ]
    )


@dataclass(slots=True)
class _ServerProcessHandle:
    process: subprocess.Popen[str]
    log_handle: IO[str]
    log_path: Path
    command: list[str]


def _endpoint(host: str, port: int) -> str:
    return f"ws://{host}:{port}"


def _build_server_command(
    cfg: OpenPIBowlEvalSuiteConfig,
    scenario: OpenPIBowlEvalScenarioConfig,
) -> list[str]:
    return [
        *cfg.server_launcher,
        cfg.server_script_path,
        "--port",
        str(cfg.port),
        "policy:checkpoint",
        "--policy.config",
        scenario.server_policy_config,
        "--policy.dir",
        scenario.checkpoint_dir,
    ]


def _wait_for_server(
    host: str,
    port: int,
    *,
    timeout_s: float,
    poll_interval_s: float,
    process: subprocess.Popen[str],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "OpenPI server exited before becoming ready. "
                f"exit_code={process.returncode} log_path={log_path}"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for OpenPI server at {host}:{port}. "
        f"See server log at {log_path}."
    )


def _start_server_process(
    cfg: OpenPIBowlEvalSuiteConfig,
    scenario: OpenPIBowlEvalScenarioConfig,
    *,
    output_root: Path,
) -> _ServerProcessHandle:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{scenario.name}_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = _build_server_command(cfg, scenario)
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=cfg.openpi_repo_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handle = _ServerProcessHandle(
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        command=command,
    )
    try:
        _wait_for_server(
            cfg.host,
            cfg.port,
            timeout_s=cfg.startup_timeout_s,
            poll_interval_s=cfg.startup_poll_interval_s,
            process=process,
            log_path=log_path,
        )
    except Exception:
        _stop_server_process(handle, shutdown_timeout_s=cfg.shutdown_timeout_s)
        raise
    return handle


def _stop_server_process(handle: _ServerProcessHandle, *, shutdown_timeout_s: float) -> None:
    try:
        if handle.process.poll() is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=shutdown_timeout_s)
    finally:
        handle.log_handle.close()


def _load_smoke_config(config_path: str | Path) -> OpenPIBowlSmokeConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Smoke config not found: {path}")
    return draccus.parse(config_class=OpenPIBowlSmokeConfig, config_path=path, args=[])


def _prepare_smoke_config(
    cfg: OpenPIBowlEvalSuiteConfig,
    scenario: OpenPIBowlEvalScenarioConfig,
    *,
    results_root: Path,
) -> OpenPIBowlSmokeConfig:
    smoke_cfg = _load_smoke_config(scenario.smoke_config_path)
    smoke_cfg.policy.endpoint = _endpoint(cfg.host, cfg.port)
    smoke_cfg.policy.request_timeout_s = cfg.request_timeout_s
    smoke_cfg.runtime.write_video = cfg.write_video
    if cfg.write_trace is not None:
        smoke_cfg.runtime.write_trace = cfg.write_trace
    smoke_cfg.runtime.output_dir = str((results_root / scenario.name).resolve())
    return smoke_cfg


def run_openpi_bowl_eval_suite(cfg: OpenPIBowlEvalSuiteConfig) -> dict[str, Any]:
    register_third_party_plugins()

    output_root = Path(cfg.output_root).resolve()
    results_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_results: list[dict[str, Any]] = []
    completed_arms: list[str] = []

    for scenario in cfg.scenarios:
        if not scenario.enabled:
            scenario_results.append(
                {
                    "name": scenario.name,
                    "status": "skipped",
                    "reason": "disabled",
                }
            )
            continue

        LOGGER.info("Running OpenPI bowl eval scenario '%s'", scenario.name)
        started_at = time.time()
        server_handle: _ServerProcessHandle | None = None
        try:
            server_handle = _start_server_process(cfg, scenario, output_root=output_root)
            smoke_cfg = _prepare_smoke_config(cfg, scenario, results_root=results_root)
            summary = run_bowl_smoke(smoke_cfg)
            completed_arms.append(scenario.name)
            scenario_results.append(
                {
                    "name": scenario.name,
                    "status": "ok",
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "server_command": server_handle.command,
                    "server_log_path": str(server_handle.log_path.resolve()),
                    "summary_path": str((results_root / scenario.name / "summary.json").resolve()),
                    "output_dir": summary.get("output_dir"),
                    "success_rate": summary.get("success_rate"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            scenario_results.append(
                {
                    "name": scenario.name,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "error": repr(exc),
                    "server_log_path": (
                        str(server_handle.log_path.resolve()) if server_handle is not None else None
                    ),
                }
            )
            LOGGER.exception("Scenario '%s' failed", scenario.name)
            if not cfg.continue_on_failure:
                if server_handle is not None:
                    _stop_server_process(server_handle, shutdown_timeout_s=cfg.shutdown_timeout_s)
                raise
        finally:
            if server_handle is not None:
                _stop_server_process(server_handle, shutdown_timeout_s=cfg.shutdown_timeout_s)

    comparison: dict[str, Any] | None = None
    if cfg.compare_after_run and completed_arms:
        comparison = compare_bowl_runs(
            BowlRunComparisonConfig(
                results_root=str(results_root),
                arms=completed_arms,
                output_dir=str(output_root / "comparison"),
            )
        )

    summary = {
        "output_root": str(output_root),
        "results_root": str(results_root),
        "endpoint": _endpoint(cfg.host, cfg.port),
        "scenarios": scenario_results,
        "comparison": comparison,
    }

    summary_path = output_root / "suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Wrote OpenPI bowl eval suite summary to %s", summary_path)
    return summary


@parser.wrap()
def openpi_bowl_eval_suite_main(cfg: OpenPIBowlEvalSuiteConfig) -> dict[str, Any]:
    return run_openpi_bowl_eval_suite(cfg)


def main() -> None:
    init_logging()
    openpi_bowl_eval_suite_main()


if __name__ == "__main__":
    main()
