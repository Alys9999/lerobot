#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):
    """
    Processes LIBERO observations into the LeRobot format.

    This step handles the specific observation structure from LIBERO environments,
    which includes nested robot_state dictionaries and image observations.

    **State Processing:**
    -   Processes the `robot_state` dictionary which contains nested end-effector,
        gripper, and joint information.
    -   Extracts and concatenates:
        - End-effector position (3D)
        - End-effector quaternion converted to axis-angle (3D)
        - Gripper joint positions (2D)
    -   Maps the concatenated state to `"observation.state"`.

    **Image Processing:**
    -   Rotates images by 180 degrees by flipping both height and width dimensions.
    -   This accounts for the HuggingFaceVLA/libero camera orientation convention.
    """

    image_flip: bool = True
    state_components: list[str] = field(
        default_factory=lambda: ["eef_pos", "eef_axis_angle", "gripper_qpos"]
    )
    state_output_key: str = OBS_STATE

    def __post_init__(self) -> None:
        normalized_components: list[str] = []
        for component in self.state_components:
            normalized = self._normalize_state_component_name(component)
            if normalized not in normalized_components:
                normalized_components.append(normalized)
        self.state_components = normalized_components

    def _process_observation(self, observation):
        """
        Processes both image and robot_state observations from LIBERO.
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                if self.image_flip:
                    # Flip both H and W to match the OpenPI/LIBERO camera convention.
                    img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # Process robot_state into a flat state vector
        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)
            state = self._build_state_tensor(robot_state)

            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[self.state_output_key] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transforms feature keys from the LIBERO format to the LeRobot standard.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # copy over non-STATE features
        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        # rebuild STATE features
        state_feats = {}

        # add our new flattened state
        state_feats[self.state_output_key] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self._get_state_dim(),),
        )

        new_features[FeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _build_state_tensor(self, robot_state: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        eef = robot_state["eef"]
        gripper = robot_state["gripper"]
        joints = robot_state["joints"]

        component_tensors: dict[str, torch.Tensor] = {
            "eef_pos": eef["pos"],
            "eef_quat": eef["quat"],
            "eef_axis_angle": self._quat2axisangle(eef["quat"]),
            "eef_mat": eef["mat"].reshape(eef["mat"].shape[0], -1),
            "gripper_qpos": gripper["qpos"],
            "gripper_qvel": gripper["qvel"],
            "joints_pos": joints["pos"],
            "joints_vel": joints["vel"],
        }

        state_parts = [component_tensors[name] for name in self.state_components]
        if not state_parts:
            raise ValueError("LiberoProcessorStep requires at least one state component.")
        return torch.cat(state_parts, dim=-1)

    def _get_state_dim(self) -> int:
        component_dims = {
            "eef_pos": 3,
            "eef_quat": 4,
            "eef_axis_angle": 3,
            "eef_mat": 9,
            "gripper_qpos": 2,
            "gripper_qvel": 2,
            "joints_pos": 7,
            "joints_vel": 7,
        }
        return sum(component_dims[name] for name in self.state_components)

    @property
    def state_dim(self) -> int:
        return self._get_state_dim()

    @staticmethod
    def _normalize_state_component_name(name: str) -> str:
        normalized = name.strip().lower()
        aliases = {
            "eef_axisangle": "eef_axis_angle",
            "eef_rotvec": "eef_axis_angle",
            "joint_pos": "joints_pos",
            "joint_positions": "joints_pos",
            "joint_vel": "joints_vel",
            "joint_velocities": "joints_vel",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "eef_pos",
            "eef_quat",
            "eef_axis_angle",
            "eef_mat",
            "gripper_qpos",
            "gripper_qvel",
            "joints_pos",
            "joints_vel",
        }
        if normalized not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"Unsupported LIBERO state component '{name}'. Expected one of: {allowed_display}.")
        return normalized

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched quaternions to axis-angle format.
        Only accepts torch tensors of shape (B, 4).

        Args:
            quat (Tensor): (B, 4) tensor of quaternions in (x, y, z, w) format

        Returns:
            Tensor: (B, 3) axis-angle vectors

        Raises:
            TypeError: if input is not a torch tensor
            ValueError: if shape is not (B, 4)
        """

        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):
    """
    Processes IsaacLab Arena observations into LeRobot format.

    **State Processing:**
    - Extracts state components from obs["policy"] based on `state_keys`.
    - Concatenates into a flat vector mapped to "observation.state".

    **Image Processing:**
    - Extracts images from obs["camera_obs"] based on `camera_keys`.
    - Converts from (B, H, W, C) uint8 to (B, C, H, W) float32 [0, 1].
    - Maps to "observation.images.<camera_name>".
    """

    # Configurable from IsaacLabEnv config / cli args: --env.state_keys="robot_joint_pos,left_eef_pos"
    state_keys: tuple[str, ...]

    # Configurable from IsaacLabEnv config / cli args: --env.camera_keys="robot_pov_cam_rgb"
    camera_keys: tuple[str, ...]

    def _process_observation(self, observation):
        """
        Processes both image and policy state observations from IsaacLab Arena.
        """
        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

        # Process policy state -> observation.state
        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]

            # Collect state components in order
            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]
                    # Flatten extra dims: (B, N, M) -> (B, N*M)
                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Not used for policy evaluation."""
        return features

    def observation(self, observation):
        return self._process_observation(observation)
