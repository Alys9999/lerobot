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

import msgpack
import numpy as np

from lerobot.adapters.openpi_jax.client import OpenPIJaxClientConfig, _MsgpackNumpyCodec, make_openpi_jax_client
from lerobot.adapters.openpi_jax.spec import OpenPIJaxLiberoSpec


def test_make_openpi_jax_client_supports_mock_transport():
    spec = OpenPIJaxLiberoSpec()
    client = make_openpi_jax_client(
        OpenPIJaxClientConfig(transport="mock"),
        action_dim=spec.action_dim,
        action_horizon=spec.action_horizon,
    )

    output = client.infer(
        {
            spec.prompt_remote_key: "pick up the bowl",
            spec.state_remote_key: np.zeros(spec.state_dim, dtype=np.float32),
        }
    )

    assert output["actions"].shape == (spec.action_horizon, spec.action_dim)
    assert output["actions"].dtype == np.float32


def test_msgpack_codec_uses_openpi_numpy_markers():
    codec = _MsgpackNumpyCodec()
    payload = {"observation/image": np.zeros((4, 5, 3), dtype=np.uint8), "score": np.float32(1.5)}

    packed = codec.packb(payload)
    unpacked_raw = msgpack.unpackb(packed, raw=False)
    image_entry = unpacked_raw.get("observation/image", unpacked_raw.get(b"observation/image"))
    score_entry = unpacked_raw.get("score", unpacked_raw.get(b"score"))

    assert image_entry is not None
    assert score_entry is not None
    assert image_entry[b"__ndarray__"] is True
    assert image_entry[b"shape"] == [4, 5, 3]
    assert score_entry[b"__npgeneric__"] is True

    round_trip = codec.unpackb(packed)
    assert round_trip["observation/image"].shape == (4, 5, 3)
    assert round_trip["score"] == np.float32(1.5)
