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

import time

from lerobot.runtime.compatibility import validate_action_command_for_spec
from lerobot.runtime.contracts import PolicyRequest, PolicyResponse

from .client import OpenPIJaxClientProtocol
from .input_codec_libero import OpenPIJaxLiberoInputCodec
from .output_codec_libero import OpenPIJaxLiberoOutputCodec
from .spec import OpenPIJaxLiberoSpec


class OpenPIJaxAdapter:
    def __init__(
        self,
        client: OpenPIJaxClientProtocol,
        spec: OpenPIJaxLiberoSpec | None = None,
        input_codec: OpenPIJaxLiberoInputCodec | None = None,
        output_codec: OpenPIJaxLiberoOutputCodec | None = None,
    ):
        self.client = client
        self.spec = spec or OpenPIJaxLiberoSpec()
        self.input_codec = input_codec or OpenPIJaxLiberoInputCodec(self.spec)
        self.output_codec = output_codec or OpenPIJaxLiberoOutputCodec(self.spec)

    def infer(self, req: PolicyRequest) -> PolicyResponse:
        start = time.perf_counter()
        encoded = self.input_codec.encode(req)
        raw_output = self.client.infer(encoded)
        latency_ms = (time.perf_counter() - start) * 1000.0
        action = self.output_codec.decode(raw_output, req)
        validate_action_command_for_spec(action, self.spec)
        return PolicyResponse(
            action=action,
            raw_output=raw_output,
            latency_ms=latency_ms,
            debug_info={
                "server_metadata": dict(getattr(self.client, "server_metadata", {})),
                "request_keys": sorted(encoded.keys()),
            },
        )

    def reset(self) -> None:
        self.client.reset()

    def close(self) -> None:
        self.client.close()
