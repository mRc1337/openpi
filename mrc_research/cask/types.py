"""JAX pytree contracts that separate training-only and deployment data."""

from __future__ import annotations

from typing import Any

from flax import struct

Array = Any


@struct.dataclass
class TrainingSegmentBatch:
    """A full demonstrated segment. This type is forbidden in deployment APIs."""

    z_features: Array
    r_features: Array
    native_actions: Array
    canonical_action_delta: Array
    time_mask: Array
    feature_mask: Array
    sensor_mask: Array
    outcome_target: Array | None = None
    contact_state_target: Array | None = None
    contact_event_target: Array | None = None
    subgoal_target: Array | None = None


@struct.dataclass
class PolicyContext:
    """Causal inputs available at a deployment decision boundary."""

    context_embedding: Array
    skill_history: Array
    skill_history_mask: Array
    observed_event: Array
    elapsed_time: Array


@struct.dataclass
class SkillCondition:
    """The only CASK object accepted by the Pi action executor."""

    z_coordinates: Array
    residual: Array
    subgoal_embedding: Array
    observed_event: Array
    validity: Array
    token_mask: Array


@struct.dataclass
class SkillPosterior:
    z_continuous: Array
    z_quantized: Array
    z_index: Array
    residual_mean: Array
    residual_logvar: Array
    residual_sample: Array


@struct.dataclass
class SkillPriorOutput:
    z_logits: Array
    z_coordinates: Array
    residual_mean: Array
    residual_logvar: Array


@struct.dataclass
class SubgoalProposals:
    geometry: Array
    visual: Array
    contact_logits: Array
    embedding: Array
    score: Array
    feasibility: Array
    modality_mask: Array
