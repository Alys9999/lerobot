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

from .compatibility import CompatibilityError, validate_action_command_for_spec, validate_openpi_jax_policy_request
from .contracts import (
    ActionCommand,
    EpisodeTrace,
    ObservationPacket,
    PolicyRequest,
    PolicyResponse,
    RobotSpec,
    RuntimeSpec,
    TaskSpec,
)
from .hidden_physics_catalog import (
    BenchmarkTaskDefinition,
    CatalogValidationError,
    V1_CATALOG,
    filter_catalog,
    load_catalog,
    validate_catalog,
)
from .hidden_physics_config import (
    HiddenPhysicsBenchmarkConfig,
    HiddenPhysicsRuntimeConfig,
    PolicyModeConfig,
)
from .hidden_physics_results import (
    BenchmarkResult,
    EpisodeResult,
    FamilyResult,
    TaskResult,
)
from .trace import read_episode_trace_summary, write_episode_trace
from .variation import VariationConfig, VariationProfile, build_family_variation_profile, build_variation_profile

__all__ = [
    "ActionCommand",
    "BenchmarkResult",
    "BenchmarkTaskDefinition",
    "CatalogValidationError",
    "CompatibilityError",
    "EpisodeResult",
    "EpisodeTrace",
    "FamilyResult",
    "HiddenPhysicsBenchmarkConfig",
    "HiddenPhysicsRuntimeConfig",
    "ObservationPacket",
    "PolicyModeConfig",
    "PolicyRequest",
    "PolicyResponse",
    "RobotSpec",
    "RuntimeSpec",
    "TaskResult",
    "TaskSpec",
    "V1_CATALOG",
    "VariationConfig",
    "VariationProfile",
    "build_family_variation_profile",
    "build_variation_profile",
    "filter_catalog",
    "load_catalog",
    "validate_catalog",
    "read_episode_trace_summary",
    "validate_action_command_for_spec",
    "validate_openpi_jax_policy_request",
    "write_episode_trace",
]
