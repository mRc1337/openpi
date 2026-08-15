"""Deployment-safe causal termination hazard."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from cask.models.segment_encoder import TemporalEncoder


class CausalTerminationHead(nn.Module):
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 512

    @nn.compact
    def __call__(
        self,
        causal_context_sequence,
        event_sequence,
        z_coordinates,
        residual,
        subgoal_embedding,
        elapsed_time,
        time_mask,
        *,
        deterministic: bool,
    ):
        if causal_context_sequence.shape[:2] != event_sequence.shape[:2]:
            raise ValueError("Causal context and event sequence must share [B,L].")
        batch_size, length = causal_context_sequence.shape[:2]
        if elapsed_time.shape != (batch_size, length, 1):
            raise ValueError("elapsed_time must have shape [B,L,1].")
        constant_condition = jnp.concatenate(
            [z_coordinates, residual, subgoal_embedding],
            axis=-1,
        )
        constant_condition = jnp.broadcast_to(
            constant_condition[:, None, :],
            (batch_size, length, constant_condition.shape[-1]),
        )
        features = jnp.concatenate(
            [
                causal_context_sequence,
                event_sequence,
                constant_condition,
                elapsed_time,
            ],
            axis=-1,
        )
        hidden = TemporalEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            causal=True,
            name="causal_tower",
        )(features, time_mask, deterministic=deterministic)
        return nn.Dense(1, name="hazard_logit")(hidden)[..., 0]
