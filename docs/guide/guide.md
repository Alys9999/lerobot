# OpenPI + LeRobot Bowl Smoke Guide

This guide collects the environment preparation, install, checkpoint download, and run commands used for the local OpenPI JAX + LeRobot LIBERO bowl smoke flow.

## Repos

- LeRobot repo: `/workspace/lerobot`
- OpenPI repo: `/workspace/openpi/openpi`

## 1. Prepare Environments

OpenPI uses its own `uv` environment pinned to Python 3.12:

```bash
cd /workspace/openpi/openpi
uv python install 3.12
uv venv --python 3.12
uv sync
```

LeRobot needs its own Python 3.12 `uv` environment plus the LIBERO and OpenPI client extras for this flow:

```bash
cd /workspace/lerobot
uv python install 3.12
uv venv --python 3.12
uv sync --extra libero --extra openpi-client-dep
```

Optional quick verification in the LeRobot environment:

```bash
cd /workspace/lerobot
uv run python - <<'PY'
import importlib.util
mods = ["libero.libero", "msgpack", "websockets.sync.client"]
for m in mods:
    print(m, "OK" if importlib.util.find_spec(m) else "MISSING")
PY
```

## 2. Download the OpenPI JAX LIBERO Checkpoint

Download the official OpenPI JAX LIBERO fine-tuned checkpoint:

```bash
cd /workspace/openpi/openpi
uv run python -u - <<'PY'
from openpi.shared.download import maybe_download
path = maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
print(path)
PY
```

The downloaded local path is:

```text
/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

## 3. Start the OpenPI Policy Server

Run the server with the downloaded checkpoint:

```bash
cd /workspace/openpi/openpi
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

Notes:

- `--port` is a top-level argument, so it must appear before `policy:checkpoint`.
- Keep this process running in its own terminal while you run the LeRobot client.

## 4. Run the LeRobot Bowl Smoke

Use the checked-in smoke config:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_bowl_variation_smoke.yaml
```

A safer first run is to extend the request timeout and disable video:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_bowl_variation_smoke.yaml \
  --policy.request_timeout_s 120 \
  --runtime.write_video false
```

## 4.1 Run a Task Series with Variation

To run the same variation-enabled smoke flow across a series of bowl tasks, use the checked-in multi-task config:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_bowl_variation_task_series_smoke.yaml \
  --policy.request_timeout_s 120
```

This config currently runs:

- `libero_spatial` task `0`
- `libero_spatial` task `1`
- `libero_spatial` task `2`

and applies the same variation profile independently to each task episode.

## 4.2 Run the Base pi05 Checkpoint on Bowl Tasks

To probe the raw `pi05_base` checkpoint against the same LIBERO bowl contract, start the server with the LIBERO policy config but point it at the base checkpoint:

```bash
cd /workspace/openpi/openpi
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir gs://openpi-assets/checkpoints/pi05_base
```

Then run the checked-in base smoke config:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_base_bowl_variation_smoke.yaml \
  --policy.request_timeout_s 120
```

For the multi-task variant:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_base_bowl_variation_task_series_smoke.yaml \
  --policy.request_timeout_s 120
```

## 4.3 Experimental DROID Contract on Bowl Tasks

The repository also includes DROID-shaped bowl smoke configs:

- `/workspace/lerobot/configs/smoke/openpi_droid_bowl_variation_smoke.yaml`
- `/workspace/lerobot/configs/smoke/openpi_droid_bowl_variation_task_series_smoke.yaml`

These use the official OpenPI DROID request schema (`observation/exterior_image_1_left`, `observation/wrist_image_left`, `observation/joint_position`, `observation/gripper_position`) against a LIBERO bowl environment. Start the server with:

```bash
cd /workspace/openpi/openpi
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_droid \
  --policy.dir gs://openpi-assets/checkpoints/pi05_droid
```

Then run:

```bash
cd /workspace/lerobot
uv run lerobot-openpi-bowl-smoke \
  --config_path /workspace/lerobot/configs/smoke/openpi_droid_bowl_variation_smoke.yaml \
  --policy.request_timeout_s 120
```

Note: this path validates the official DROID input contract, but it still projects the returned 8D DROID action chunk onto the first 7 dimensions expected by the LIBERO bowl environment. Treat it as an experimental cross-embodiment smoke path rather than a semantically equivalent DROID evaluation.

## 5. Outputs

The smoke run writes outputs under:

```text
/workspace/lerobot/outputs/openpi_bowl_smoke
```

Key artifacts:

- `summary.json`
- `traces/episode_*.json`
- `videos/episode_*.mp4` if video is enabled

Attempt and recovery metrics are written into each episode trace as:

- numeric aggregates in `metrics` such as `attempt_count`, `failed_attempt_count`, `successful_attempt_count`, `first_attempt_to_success_s`, and `first_failure_to_success_s`
- structured per-attempt details in `metadata.attempt_analysis`

## 5.1 Compare ALOHA, DROID, and Base Runs

If you have already written the three run outputs under one directory, for example:

```text
/workspace/lerobot/outputs/bowl_compare_inputs/
  aloha/summary.json
  droid/summary.json
  base/summary.json
```

run the comparison script:

```bash
cd /workspace/lerobot
uv run python -m lerobot.scripts.lerobot_compare_bowl_runs \
  --results_root /workspace/lerobot/outputs/bowl_compare_inputs
```

This writes:

- `comparison/comparison.md`
- `comparison/overall.csv`
- `comparison/per_task.csv`
- `comparison/comparison.json`

under the same `results_root` by default.

## 5.2 Run All Three Sequentially

The repo also includes an orchestration script and example config that will:

1. start the corresponding OpenPI policy server
2. run the matching LeRobot bowl smoke
3. stop the server
4. continue to the next arm
5. write a final comparison table

Example:

```bash
cd /workspace/lerobot
uv run python -m lerobot.scripts.lerobot_openpi_bowl_eval_suite \
  --config_path /workspace/lerobot/configs/smoke/openpi_bowl_eval_suite.yaml
```

By default this writes:

- `outputs/openpi_bowl_eval_suite/runs/aloha`
- `outputs/openpi_bowl_eval_suite/runs/droid`
- `outputs/openpi_bowl_eval_suite/runs/base`
- `outputs/openpi_bowl_eval_suite/comparison`
- `outputs/openpi_bowl_eval_suite/suite_summary.json`

Notes:

- `openpi_repo_root` in the example config must point at your local OpenPI checkout.
- the `base` path intentionally serves `pi05_base` with `--policy.config pi05_libero`
- if your OpenPI checkout uses a different ALOHA config name or checkpoint path than `pi05_aloha`, override `scenarios[0].server_policy_config` and `scenarios[0].checkpoint_dir` in the suite config

## 6. Useful Checks

Check that the OpenPI environment has the required packages:

```bash
cd /workspace/openpi/openpi
uv run python - <<'PY'
import importlib.util
mods = ["openpi", "openpi_client", "lerobot", "msgpack", "websockets.sync.client"]
for m in mods:
    print(m, "OK" if importlib.util.find_spec(m) else "MISSING")
PY
```

Check that the LeRobot environment has the required smoke dependencies:

```bash
cd /workspace/lerobot
uv run python - <<'PY'
import importlib.util
mods = ["lerobot", "libero.libero", "msgpack", "websockets.sync.client"]
for m in mods:
    print(m, "OK" if importlib.util.find_spec(m) else "MISSING")
PY
```

## 7. Common Issues

If you see:

```text
ImportError: LIBERO is required ...
```

install the LeRobot extras:

```bash
cd /workspace/lerobot
uv sync --extra libero --extra openpi-client-dep
```

If you see:

```text
Unrecognized options: --port=8000
```

use:

```bash
--port 8000
```

instead of:

```bash
--port=8000
```

If the first OpenPI request takes a long time, increase:

```bash
--policy.request_timeout_s 120
```

or a larger value.
