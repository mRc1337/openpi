"""Small conditional flow decoder used before integrating the Pi executor."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from cask.models.segment_encoder import TemporalEncoder


def scalar_sincos_embedding(value, width: int, *, min_period: float = 4e-3, max_period: float = 4.0):
    if width % 2 != 0:
        raise ValueError("The time embedding width must be even.")
    fraction = jnp.linspace(0.0, 1.0, width // 2)
    period = min_period * (max_period / min_period) ** fraction
    phase = value[:, None] / period[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


class MVPFlowDecoder(nn.Module):
    action_dim: int
    condition_dim: int
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 256

    @nn.compact
    def __call__(
        self,
        noisy_actions,
        flow_time,
        initial_state,
        skill_condition,
        action_time_mask,
        *,
        deterministic: bool,
    ):
        if noisy_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {noisy_actions.shape[-1]}."
            )
        if skill_condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition_dim={self.condition_dim}, got {skill_condition.shape[-1]}."
            )
        horizon = noisy_actions.shape[1]
        time_embedding = scalar_sincos_embedding(flow_time, self.width)
        global_condition = jnp.concatenate(
            [initial_state, skill_condition, time_embedding],
            axis=-1,
        )
        repeated_condition = jnp.broadcast_to(
            global_condition[:, None, :],
            (noisy_actions.shape[0], horizon, global_condition.shape[-1]),
        )
        features = jnp.concatenate([noisy_actions, repeated_condition], axis=-1)
        hidden = TemporalEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            causal=False,
            name="decoder",
        )(features, action_time_mask, deterministic=deterministic)
        return nn.Dense(self.action_dim, name="velocity_head")(hidden)

    def flow_training_inputs(self, rng, actions, action_mask):
        mask = action_mask.astype(actions.dtype)
        actions = actions * mask
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape) * mask
        flow_time = (
            jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        )
        expanded_time = flow_time[..., None, None]
        noisy_actions = (
            expanded_time * noise + (1 - expanded_time) * actions
        ) * mask
        target_velocity = (noise - actions) * mask
        return noisy_actions, flow_time, target_velocity
