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

"""Policy executor abstraction for the Hidden-Physics Diagnostic Benchmark.

Provides a :class:`PolicyExecutor` protocol that unifies LeRobot-native
and adapter-backed (e.g. OpenPI) policy execution behind a single
``reset / infer / close`` interface so the benchmark runner never needs to
know which policy family it is driving.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import numpy as np

from .hidden_physics_config import PolicyModeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PolicyExecutor(Protocol):
    """Minimal interface that the benchmark runner calls."""

    def reset(self) -> None:
        """Prepare for a new episode (clear action queues, etc.)."""
        ...

    def infer(
        self,
        observation: dict[str, Any],
        *,
        task_text: str,
        control_dt: float,
    ) -> np.ndarray:
        """Return an action array from *observation*."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# Native policy executor
# ---------------------------------------------------------------------------


class NativePolicyExecutor:
    """Wraps a LeRobot-native policy behind the :class:`PolicyExecutor` API.

    Construction is deferred to :meth:`build` so that heavy imports happen
    only when this executor is actually requested.

    The inference pipeline mirrors ``lerobot_eval.py``:
      1. ``preprocess_observation`` — numpy gym obs → LeRobot tensor format
      2. task text injection (``observation["task"]``)
      3. ``env_preprocessor`` — e.g. ``LiberoProcessorStep``
      4. ``preprocessor`` — policy-specific normalisation
      5. ``policy.select_action``
      6. ``postprocessor`` / ``env_postprocessor``
    """

    def __init__(
        self,
        policy: Any,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        env_preprocessor: Any | None = None,
        env_postprocessor: Any | None = None,
        device: str = "cuda",
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._env_preprocessor = env_preprocessor
        self._env_postprocessor = env_postprocessor
        self._device = device

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def infer(
        self,
        observation: dict[str, Any],
        *,
        task_text: str,
        control_dt: float,
    ) -> np.ndarray:
        import torch

        obs = dict(observation)

        # Step 1: preprocess_observation — convert numpy gym obs → LeRobot
        # tensor format (images BCHW float, state → tensors).
        try:
            from lerobot.envs.utils import preprocess_observation
            obs = preprocess_observation(obs)
        except ImportError:
            logger.debug("preprocess_observation not available — passing raw obs.")

        # Step 2: Inject task text.  lerobot_eval.py uses add_envs_task()
        # which requires a vectorised env.  We emulate it here for the
        # single-env (non-vectorised) case.
        if "task" not in obs:
            obs["task"] = [task_text] if task_text else [""]

        # Step 3: env preprocessor (e.g. LiberoProcessorStep)
        if self._env_preprocessor is not None:
            obs = self._env_preprocessor(obs)

        # Step 4: policy preprocessor (normalisation, etc.)
        if self._preprocessor is not None:
            obs = self._preprocessor(obs)

        # ensure tensors are on the correct device
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                obs[key] = val.to(self._device)

        with torch.inference_mode():
            action = self._policy.select_action(obs)

        # policy postprocessor
        if self._postprocessor is not None:
            action = self._postprocessor(action)

        # env postprocessor
        if self._env_postprocessor is not None:
            action = self._env_postprocessor(action)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return np.asarray(action)

    def close(self) -> None:
        del self._policy

    @classmethod
    def build(cls, config: PolicyModeConfig, device: str = "cuda") -> NativePolicyExecutor:
        """Construct from a :class:`PolicyModeConfig`.

        Imports ``make_policy`` and ``make_pre_post_processors`` lazily.
        """
        from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

        # Build policy config from pretrained path
        policy_cfg = make_policy_config("pretrained")
        if config.native_policy_path:
            policy_cfg.pretrained_path = config.native_policy_path
        for key, val in config.native_policy_overrides.items():
            if hasattr(policy_cfg, key):
                setattr(policy_cfg, key, val)

        policy = make_policy(policy_cfg)
        policy.to(device)
        policy.eval()

        # Attempt to build pre/post processors
        preprocessor = postprocessor = None
        env_preprocessor = env_postprocessor = None
        try:
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg)
        except Exception:
            logger.debug("Could not build policy pre/post processors — using identity.")
        try:
            from lerobot.envs.factory import make_env_pre_post_processors
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(policy_cfg)
        except Exception:
            logger.debug("Could not build env pre/post processors — using identity.")

        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            device=device,
        )


# ---------------------------------------------------------------------------
# OpenPI adapter executor
# ---------------------------------------------------------------------------


class OpenPIAdapterExecutor:
    """Wraps an OpenPI JAX adapter behind the :class:`PolicyExecutor` API.

    The inference pipeline applies ``LiberoProcessorStep`` (when available)
    to preprocess the raw observation into the format OpenPI models expect
    (flipped images, flattened robot state), then routes through the codec
    layer.

    Construction is deferred to :meth:`build`.
    """

    def __init__(
        self,
        adapter: Any,
        input_codec: Any,
        output_codec: Any,
        spec: Any,
        env_processor: Any | None = None,
    ) -> None:
        self._adapter = adapter
        self._input_codec = input_codec
        self._output_codec = output_codec
        self._spec = spec
        self._env_processor = env_processor
        self._action_queue: list[np.ndarray] = []

    def reset(self) -> None:
        self._action_queue.clear()

    def infer(
        self,
        observation: dict[str, Any],
        *,
        task_text: str,
        control_dt: float,
    ) -> np.ndarray:
        # If we still have queued actions from a chunked prediction, pop one.
        if self._action_queue:
            return self._action_queue.pop(0)

        from .contracts import ObservationPacket, PolicyRequest, RobotSpec, RuntimeSpec, TaskSpec

        obs = dict(observation)

        # Step 1: preprocess_observation (numpy gym → LeRobot tensors)
        try:
            from lerobot.envs.utils import preprocess_observation
            obs = preprocess_observation(obs)
        except ImportError:
            logger.debug("preprocess_observation not available — using raw obs.")

        # Step 2: Inject task text
        if "task" not in obs:
            obs["task"] = [task_text] if task_text else [""]

        # Step 3: LiberoProcessorStep (flips images, flattens robot state)
        if self._env_processor is not None:
            try:
                obs = self._env_processor(obs)
            except Exception:
                logger.debug("LiberoProcessorStep failed — using pre-processed obs.")

        # Build observation packet from processed data
        images = {}
        robot_state = {}
        for k, v in obs.items():
            if k == "task":
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 3:
                images[k] = v
            elif isinstance(v, np.ndarray):
                robot_state[k] = v
            elif hasattr(v, "numpy"):  # torch.Tensor
                arr = v.detach().cpu().numpy()
                if arr.ndim >= 3:
                    images[k] = arr
                else:
                    robot_state[k] = arr

        packet = ObservationPacket(
            timestamp=time.time(),
            episode_id="",
            step_id=0,
            images=images,
            robot_state=robot_state,
            task_text=task_text,
            task_id="",
            robot_id="",
            embodiment_id="",
            backend_id="libero",
        )

        request = PolicyRequest(
            observation=packet,
            robot_spec=RobotSpec(robot_id="", action_dim=7),
            task_spec=TaskSpec(task_suite="", task_id="", prompt=task_text),
            runtime_spec=RuntimeSpec(control_dt=control_dt),
        )

        # Encode -> infer -> decode
        encoded = self._input_codec.encode(request)
        raw_output = self._adapter.infer(encoded)
        action_cmd = self._output_codec.decode(raw_output)

        actions = np.asarray(action_cmd.values)
        if actions.ndim == 2 and actions.shape[0] > 1:
            # Chunked output: return first, queue rest
            self._action_queue = [actions[i] for i in range(1, actions.shape[0])]
            return actions[0]
        return actions.flatten()

    def close(self) -> None:
        if hasattr(self._adapter, "close"):
            self._adapter.close()

    @classmethod
    def build(cls, config: PolicyModeConfig) -> OpenPIAdapterExecutor:
        """Construct from a :class:`PolicyModeConfig`."""
        from lerobot.adapters.openpi_jax.adapter import OpenPIJaxAdapter
        from lerobot.adapters.openpi_jax.input_codec_libero import OpenPIJaxLiberoInputCodec
        from lerobot.adapters.openpi_jax.output_codec_libero import OpenPIJaxLiberoOutputCodec
        from lerobot.adapters.openpi_jax.spec import OpenPIJaxLiberoSpec

        spec = OpenPIJaxLiberoSpec()
        adapter = OpenPIJaxAdapter(endpoint=config.adapter_endpoint)
        input_codec = OpenPIJaxLiberoInputCodec(spec=spec)
        output_codec = OpenPIJaxLiberoOutputCodec(spec=spec)

        # Load LiberoProcessorStep if available
        env_processor = None
        try:
            from lerobot.processor.env_processor import LiberoProcessorStep
            env_processor = LiberoProcessorStep()
            logger.info("LiberoProcessorStep loaded for OpenPI adapter preprocessing.")
        except ImportError:
            logger.debug("LiberoProcessorStep not available — adapter will use raw obs.")

        return cls(
            adapter=adapter,
            input_codec=input_codec,
            output_codec=output_codec,
            spec=spec,
            env_processor=env_processor,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_executor(config: PolicyModeConfig, device: str = "cuda") -> PolicyExecutor:
    """Build the appropriate executor based on ``config.mode``."""
    if config.mode == "native":
        return NativePolicyExecutor.build(config, device=device)
    elif config.mode == "openpi_adapter":
        return OpenPIAdapterExecutor.build(config)
    else:
        raise ValueError(f"Unknown policy mode '{config.mode}'. Expected 'native' or 'openpi_adapter'.")
