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

## 5. Outputs

The smoke run writes outputs under:

```text
/workspace/lerobot/outputs/openpi_bowl_smoke
```

Key artifacts:

- `summary.json`
- `traces/episode_*.json`
- `videos/episode_*.mp4` if video is enabled

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
