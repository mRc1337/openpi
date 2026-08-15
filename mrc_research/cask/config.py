"""Typed configuration for the CASK model family.

The defaults mirror model_design.md v0.2. Data- and robot-dependent values are
intentionally optional so that a missing M0 decision cannot silently become a
made-up experimental constant.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SegmentEncoderConfig:
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_segment_steps: int = 128

    def validate(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("Segment encoder width and depth must be positive.")
        if self.width % self.num_heads != 0:
            raise ValueError("Segment encoder width must be divisible by num_heads.")
        if self.max_segment_steps <= 0:
            raise ValueError("max_segment_steps must be positive.")


@dataclasses.dataclass(frozen=True)
class HybridSkillConfig:
    fsq_levels: tuple[int, ...] = (3, 3, 3, 3)
    residual_dim: int = 16
    residual_dropout_rate: float = 0.2

    def validate(self) -> None:
        if not self.fsq_levels:
            raise ValueError("At least one FSQ coordinate is required.")
        if any(level < 3 or level % 2 == 0 for level in self.fsq_levels):
            raise ValueError("The first implementation supports odd FSQ levels >= 3.")
        if self.residual_dim <= 0:
            raise ValueError("residual_dim must be positive.")
        if not 0.0 <= self.residual_dropout_rate < 1.0:
            raise ValueError("residual_dropout_rate must be in [0, 1).")

    @property
    def discrete_dim(self) -> int:
        return len(self.fsq_levels)

    @property
    def num_combinations(self) -> int:
        result = 1
        for level in self.fsq_levels:
            result *= level
        return result


@dataclasses.dataclass(frozen=True)
class SubgoalConfig:
    embedding_dim: int = 256
    visual_target_dim: int = 128
    contact_target_dim: int = 7
    num_proposals: int = 3
    geometry_dim: int | None = None

    def validate(self) -> None:
        if self.embedding_dim <= 0 or self.num_proposals <= 0:
            raise ValueError("Subgoal dimensions and proposal count must be positive.")
        if self.geometry_dim is not None and self.geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive when configured.")


@dataclasses.dataclass(frozen=True)
class PiExecutorConfig:
    base_variant: str = "pi05"
    pi_action_dim: int = 32
    action_horizon: int | None = None
    observed_event_dim: int = 7
    condition_validity_dim: int = 3
    enable_skill_adarms: bool = False
    native_action_slots: tuple[int, ...] | None = None

    def validate(self) -> None:
        if self.base_variant not in {"pi0", "pi05"}:
            raise ValueError("CASK's first executor supports only pi0 or pi05 flow models.")
        if self.pi_action_dim != 32:
            raise ValueError("The pinned OpenPI checkpoint contract uses 32 action slots.")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive when configured.")
        if self.native_action_slots is not None:
            if len(set(self.native_action_slots)) != len(self.native_action_slots):
                raise ValueError("native_action_slots must be unique.")
            if any(slot < 0 or slot >= self.pi_action_dim for slot in self.native_action_slots):
                raise ValueError("A native action slot is outside the Pi action space.")


@dataclasses.dataclass(frozen=True)
class TimingConfig:
    skill_rate_hz: float = 10.0
    control_rate_hz: float | None = None
    min_skill_seconds: float | None = None
    max_skill_seconds: float | None = None
    execute_prefix_steps: int | None = None

    def validate(self) -> None:
        if self.skill_rate_hz <= 0:
            raise ValueError("skill_rate_hz must be positive.")
        optional_positive = {
            "control_rate_hz": self.control_rate_hz,
            "min_skill_seconds": self.min_skill_seconds,
            "max_skill_seconds": self.max_skill_seconds,
            "execute_prefix_steps": self.execute_prefix_steps,
        }
        for name, value in optional_positive.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured.")
        if (
            self.min_skill_seconds is not None
            and self.max_skill_seconds is not None
            and self.min_skill_seconds >= self.max_skill_seconds
        ):
            raise ValueError("min_skill_seconds must be less than max_skill_seconds.")


@dataclasses.dataclass(frozen=True)
class CaskConfig:
    segment_encoder: SegmentEncoderConfig = dataclasses.field(default_factory=SegmentEncoderConfig)
    skill: HybridSkillConfig = dataclasses.field(default_factory=HybridSkillConfig)
    subgoal: SubgoalConfig = dataclasses.field(default_factory=SubgoalConfig)
    executor: PiExecutorConfig = dataclasses.field(default_factory=PiExecutorConfig)
    timing: TimingConfig = dataclasses.field(default_factory=TimingConfig)

    def validate(self, *, require_m0_fields: bool = False) -> None:
        self.segment_encoder.validate()
        self.skill.validate()
        self.subgoal.validate()
        self.executor.validate()
        self.timing.validate()
        if require_m0_fields:
            missing = []
            if self.subgoal.geometry_dim is None:
                missing.append("subgoal.geometry_dim")
            if self.executor.native_action_slots is None:
                missing.append("executor.native_action_slots")
            if self.executor.action_horizon is None:
                missing.append("executor.action_horizon")
            if self.timing.control_rate_hz is None:
                missing.append("timing.control_rate_hz")
            if self.timing.min_skill_seconds is None:
                missing.append("timing.min_skill_seconds")
            if self.timing.max_skill_seconds is None:
                missing.append("timing.max_skill_seconds")
            if missing:
                raise ValueError("M0 has not frozen required fields: " + ", ".join(missing))
