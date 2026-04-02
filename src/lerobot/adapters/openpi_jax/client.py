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

import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


class OpenPIJaxTransportError(RuntimeError):
    pass


class OpenPIJaxClientProtocol(Protocol):
    server_metadata: dict[str, Any]

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class OpenPIJaxClientConfig:
    transport: str = "websocket"
    endpoint: str = "ws://127.0.0.1:8000"
    timeout_s: float = 10.0
    request_timeout_s: float | None = None
    connect_retry_interval_s: float = 1.0
    api_key: str | None = None
    mock_action_scale: float = 0.25

    @property
    def resolved_endpoint(self) -> str:
        if self.endpoint.startswith("ws://") or self.endpoint.startswith("wss://"):
            return self.endpoint
        return f"ws://{self.endpoint}"

    @property
    def resolved_transport(self) -> str:
        return self.transport.strip().lower()


class _MsgpackNumpyCodec:
    def __init__(self) -> None:
        try:
            self._msgpack = importlib.import_module("msgpack")
        except ImportError as exc:
            raise ImportError(
                "OpenPI JAX client requires the optional msgpack dependency. "
                "Install with `pip install -e .[openpi-client-dep]`."
            ) from exc

    def packb(self, payload: dict[str, Any]) -> bytes:
        return self._msgpack.packb(payload, default=self._encode_numpy, use_bin_type=True)

    def unpackb(self, payload: bytes) -> dict[str, Any]:
        return self._msgpack.unpackb(payload, object_hook=self._decode_numpy, raw=False)

    def _encode_numpy(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
            raise TypeError(f"Unsupported dtype for OpenPI JAX transport: {obj.dtype}")

        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": np.ascontiguousarray(obj).tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }

        return obj

    def _decode_numpy(self, obj: dict[Any, Any]) -> Any:
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])

        return obj


class WebsocketOpenPIJaxClient(OpenPIJaxClientProtocol):
    def __init__(self, config: OpenPIJaxClientConfig):
        self.config = config
        self.server_metadata: dict[str, Any] = {}
        self._codec = _MsgpackNumpyCodec()
        try:
            self._ws_client = importlib.import_module("websockets.sync.client")
        except ImportError as exc:
            raise ImportError(
                "OpenPI JAX client requires the optional websockets dependency. "
                "Install with `pip install -e .[openpi-client-dep]`."
            ) from exc
        self._conn = None
        self._ensure_connected()

    def _ensure_connected(self) -> None:
        if self._conn is not None:
            return

        deadline = time.monotonic() + self.config.timeout_s if self.config.timeout_s is not None else None
        headers = {"Authorization": f"Api-Key {self.config.api_key}"} if self.config.api_key else None
        last_error: Exception | None = None
        while deadline is None or time.monotonic() < deadline:
            try:
                self._conn = self._ws_client.connect(
                    self.config.resolved_endpoint,
                    compression=None,
                    max_size=None,
                    open_timeout=self.config.timeout_s,
                    close_timeout=self.config.timeout_s,
                    additional_headers=headers,
                )
                metadata = self._conn.recv(self.config.timeout_s)
                if isinstance(metadata, str):
                    raise OpenPIJaxTransportError(
                        f"Expected binary metadata during OpenPI JAX handshake, got text: {metadata}"
                    )
                self.server_metadata = self._codec.unpackb(metadata)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._conn = None
                LOGGER.warning("OpenPI JAX websocket connect failed: %s", exc)
                time.sleep(self.config.connect_retry_interval_s)

        raise OpenPIJaxTransportError(
            f"Failed to connect to OpenPI JAX server at {self.config.resolved_endpoint}."
        ) from last_error

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        payload = self._codec.packb(observation)
        try:
            self._conn.send(payload)
            response = self._conn.recv(self.config.request_timeout_s)
        except TimeoutError as exc:
            self.close()
            timeout = self.config.request_timeout_s
            raise OpenPIJaxTransportError(
                "Timed out waiting for OpenPI JAX inference response"
                f"{f' after {timeout:.1f}s' if timeout is not None else ''}. "
                "The first request may spend extra time compiling the model; "
                "increase `policy.request_timeout_s` or set it to null to wait indefinitely."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise OpenPIJaxTransportError("OpenPI JAX websocket inference failed.") from exc

        if isinstance(response, str):
            raise OpenPIJaxTransportError(f"OpenPI JAX server returned text error:\n{response}")
        return self._codec.unpackb(response)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        finally:
            self._conn = None


class MockOpenPIJaxClient(OpenPIJaxClientProtocol):
    def __init__(self, config: OpenPIJaxClientConfig, *, action_dim: int = 7, action_horizon: int = 10):
        self.config = config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.server_metadata: dict[str, Any] = {
            "transport": "mock",
            "endpoint": config.endpoint,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
        }
        self._timestep = 0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        time_index = np.arange(self.action_horizon, dtype=np.float32) + float(self._timestep)
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)

        if self.action_dim > 0:
            actions[:, 0] = 0.55 * np.sin(time_index / 7.0)
        if self.action_dim > 1:
            actions[:, 1] = 0.35 * np.cos(time_index / 11.0)
        if self.action_dim > 2:
            actions[:, 2] = 0.25 * np.sin(time_index / 5.0)
        if self.action_dim > 6:
            actions[:, 6] = np.where(((time_index // 15).astype(np.int32) % 2) == 0, -1.0, 1.0)

        if self.action_dim > 1:
            actions[:, : min(self.action_dim, 6)] *= float(self.config.mock_action_scale)

        self._timestep += self.action_horizon
        return {"actions": np.clip(actions, -1.0, 1.0)}

    def reset(self) -> None:
        self._timestep = 0

    def close(self) -> None:
        return None


def make_openpi_jax_client(
    config: OpenPIJaxClientConfig,
    *,
    action_dim: int = 7,
    action_horizon: int = 10,
) -> OpenPIJaxClientProtocol:
    if config.resolved_transport == "websocket":
        return WebsocketOpenPIJaxClient(config)
    if config.resolved_transport == "mock":
        return MockOpenPIJaxClient(config, action_dim=action_dim, action_horizon=action_horizon)
    raise ValueError(
        f"Unsupported OpenPI JAX transport '{config.transport}'. Expected one of: websocket, mock."
    )
