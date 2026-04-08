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

from pathlib import Path
from types import SimpleNamespace

from lerobot.scripts import lerobot_openpi_bowl_eval_suite as suite
from lerobot.scripts.lerobot_openpi_bowl_smoke import OpenPIBowlSmokeConfig


def test_build_server_command_uses_expected_openpi_checkpoint_contract():
    cfg = suite.OpenPIBowlEvalSuiteConfig(
        openpi_repo_root="/openpi",
        server_launcher=["uv", "run", "python"],
        port=8123,
        scenarios=[
            suite.OpenPIBowlEvalScenarioConfig(
                name="base",
                smoke_config_path="configs/smoke/openpi_base_bowl_variation_smoke.yaml",
                server_policy_config="pi05_libero",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_base",
            )
        ],
    )

    command = suite._build_server_command(cfg, cfg.scenarios[0])

    assert command == [
        "uv",
        "run",
        "python",
        "scripts/serve_policy.py",
        "--port",
        "8123",
        "policy:checkpoint",
        "--policy.config",
        "pi05_libero",
        "--policy.dir",
        "gs://openpi-assets/checkpoints/pi05_base",
    ]


def test_run_openpi_bowl_eval_suite_runs_servers_sequentially_and_compares_outputs(
    tmp_path,
    monkeypatch,
):
    openpi_root = tmp_path / "openpi"
    openpi_root.mkdir()

    cfg = suite.OpenPIBowlEvalSuiteConfig(
        openpi_repo_root=str(openpi_root),
        output_root=str(tmp_path / "outputs"),
        request_timeout_s=90.0,
        write_video=False,
        continue_on_failure=True,
        scenarios=[
            suite.OpenPIBowlEvalScenarioConfig(
                name="aloha",
                smoke_config_path="configs/smoke/openpi_aloha_bowl_variation_smoke.yaml",
                server_policy_config="pi05_aloha",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_aloha",
            ),
            suite.OpenPIBowlEvalScenarioConfig(
                name="droid",
                smoke_config_path="configs/smoke/openpi_droid_bowl_variation_smoke.yaml",
                server_policy_config="pi05_droid",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_droid",
            ),
            suite.OpenPIBowlEvalScenarioConfig(
                name="base",
                smoke_config_path="configs/smoke/openpi_base_bowl_variation_smoke.yaml",
                server_policy_config="pi05_libero",
                checkpoint_dir="gs://openpi-assets/checkpoints/pi05_base",
            ),
        ],
    )

    start_calls: list[str] = []
    stop_calls: list[str] = []
    smoke_calls: list[tuple[str, str, float | None, bool]] = []
    compare_calls: list[tuple[str, list[str], str | None]] = []

    monkeypatch.setattr(suite, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(suite, "_load_smoke_config", lambda _path: OpenPIBowlSmokeConfig())

    def fake_start_server_process(
        _cfg: suite.OpenPIBowlEvalSuiteConfig,
        scenario: suite.OpenPIBowlEvalScenarioConfig,
        *,
        output_root: Path,
    ):
        start_calls.append(scenario.name)
        log_path = output_root / "logs" / f"{scenario.name}_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        return SimpleNamespace(
            command=["uv", "run", "python", "scripts/serve_policy.py"],
            log_path=log_path,
            scenario_name=scenario.name,
        )

    def fake_stop_server_process(handle, *, shutdown_timeout_s: float) -> None:
        del shutdown_timeout_s
        stop_calls.append(handle.scenario_name)

    def fake_run_bowl_smoke(smoke_cfg: OpenPIBowlSmokeConfig) -> dict[str, object]:
        scenario_name = Path(smoke_cfg.runtime.output_dir).name
        smoke_calls.append(
            (
                scenario_name,
                smoke_cfg.policy.endpoint,
                smoke_cfg.policy.request_timeout_s,
                smoke_cfg.runtime.write_video,
            )
        )
        if scenario_name == "droid":
            raise RuntimeError("synthetic droid failure")

        output_dir = Path(smoke_cfg.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text("{}", encoding="utf-8")
        return {
            "output_dir": str(output_dir),
            "success_rate": 1.0,
        }

    def fake_compare_bowl_runs(compare_cfg) -> dict[str, object]:
        compare_calls.append((compare_cfg.results_root, list(compare_cfg.arms), compare_cfg.output_dir))
        return {"overall": [], "per_task": []}

    monkeypatch.setattr(suite, "_start_server_process", fake_start_server_process)
    monkeypatch.setattr(suite, "_stop_server_process", fake_stop_server_process)
    monkeypatch.setattr(suite, "run_bowl_smoke", fake_run_bowl_smoke)
    monkeypatch.setattr(suite, "compare_bowl_runs", fake_compare_bowl_runs)

    result = suite.run_openpi_bowl_eval_suite(cfg)

    assert start_calls == ["aloha", "droid", "base"]
    assert stop_calls == ["aloha", "droid", "base"]
    assert smoke_calls == [
        ("aloha", "ws://127.0.0.1:8000", 90.0, False),
        ("droid", "ws://127.0.0.1:8000", 90.0, False),
        ("base", "ws://127.0.0.1:8000", 90.0, False),
    ]
    assert compare_calls == [
        (
            str((Path(cfg.output_root).resolve() / "runs")),
            ["aloha", "base"],
            str(Path(cfg.output_root).resolve() / "comparison"),
        )
    ]

    scenario_status = {item["name"]: item["status"] for item in result["scenarios"]}
    assert scenario_status == {
        "aloha": "ok",
        "droid": "failed",
        "base": "ok",
    }
    assert result["comparison"] == {"overall": [], "per_task": []}
    assert Path(cfg.output_root, "suite_summary.json").exists()
