"""Mask and action-slot utilities shared by CASK core and the Pi executor."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ActionSlotMap:
    """A reversible mapping from native action coordinates into Pi's 32 slots."""

    native_to_pi: tuple[int, ...]
    pi_dim: int = 32

    def __post_init__(self) -> None:
        if not self.native_to_pi:
            raise ValueError("native_to_pi cannot be empty.")
        if len(set(self.native_to_pi)) != len(self.native_to_pi):
            raise ValueError("Each native action dimension needs a unique Pi slot.")
        if any(index < 0 or index >= self.pi_dim for index in self.native_to_pi):
            raise ValueError("A native action coordinate maps outside the Pi action space.")

    @classmethod
    def from_sequence(cls, slots: Sequence[int], *, pi_dim: int = 32) -> "ActionSlotMap":
        return cls(tuple(int(slot) for slot in slots), pi_dim=pi_dim)

    @property
    def native_dim(self) -> int:
        return len(self.native_to_pi)

    def pack(self, native_actions):
        if native_actions.shape[-1] != self.native_dim:
            raise ValueError(
                f"Expected native action dim {self.native_dim}, got {native_actions.shape[-1]}."
            )
        packed = jnp.zeros((*native_actions.shape[:-1], self.pi_dim), dtype=native_actions.dtype)
        return packed.at[..., jnp.asarray(self.native_to_pi)].set(native_actions)

    def unpack(self, pi_actions):
        if pi_actions.shape[-1] != self.pi_dim:
            raise ValueError(f"Expected Pi action dim {self.pi_dim}, got {pi_actions.shape[-1]}.")
        return pi_actions[..., jnp.asarray(self.native_to_pi)]

    def pack_validity(self, native_validity):
        if native_validity.shape[-1] != self.native_dim:
            raise ValueError(
                f"Expected native validity dim {self.native_dim}, got {native_validity.shape[-1]}."
            )
        packed = jnp.zeros((*native_validity.shape[:-1], self.pi_dim), dtype=jnp.bool_)
        return packed.at[..., jnp.asarray(self.native_to_pi)].set(native_validity.astype(jnp.bool_))


def masked_mean(values, mask, *, axis=None, keepdims: bool = False):
    mask = jnp.asarray(mask, dtype=values.dtype)
    numerator = jnp.sum(values * mask, axis=axis, keepdims=keepdims)
    denominator = jnp.sum(mask, axis=axis, keepdims=keepdims)
    return numerator / jnp.maximum(denominator, 1)


def mask_flow_path(actions, noise, action_mask):
    """Mask all analytic flow-path tensors before they reach the action projection."""

    mask = jnp.asarray(action_mask, dtype=actions.dtype)
    if actions.shape != noise.shape or actions.shape != mask.shape:
        raise ValueError(
            f"actions, noise, and action_mask must match; got {actions.shape}, "
            f"{noise.shape}, and {mask.shape}."
        )
    actions = actions * mask
    noise = noise * mask
    return actions, noise, noise - actions


def masked_flow_mse(predicted_velocity, target_velocity, action_mask):
    """Return per-action-token MSE and a time-valid mask.

    The reduction is over valid action coordinates, not the fixed 32D Pi width.
    """

    mask = jnp.asarray(action_mask, dtype=predicted_velocity.dtype)
    if predicted_velocity.shape != target_velocity.shape or predicted_velocity.shape != mask.shape:
        raise ValueError("Velocity tensors and action_mask must have identical shapes.")
    squared_error = jnp.square(predicted_velocity - target_velocity)
    per_timestep = masked_mean(squared_error, mask, axis=-1)
    time_valid = jnp.any(action_mask, axis=-1)
    return per_timestep, time_valid


def reduce_flow_loss(per_timestep, time_valid):
    return masked_mean(per_timestep, time_valid, axis=None)
