"""Skill-token extension for the pinned OpenPI Pi0/Pi0.5 flow executor."""

from __future__ import annotations

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from cask.masking import masked_flow_mse
from cask.pi_layout import skill_insert_index
from cask.types import SkillCondition
from openpi.models import model as openpi_model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
import openpi.models.gemma as gemma


@dataclasses.dataclass(frozen=True)
class CaskPi0Config(pi0_config.Pi0Config):
    skill_code_dim: int = 4
    skill_residual_dim: int = 16
    skill_subgoal_dim: int = 256
    skill_observed_event_dim: int = 7
    skill_validity_dim: int = 3

    def create(self, rng):
        return CaskPi0(self, rngs=nnx.Rngs(rng))


class SkillTokenProjector(nnx.Module):
    def __init__(self, config: CaskPi0Config, width: int, *, rngs: nnx.Rngs):
        instance_dim = (
            config.skill_residual_dim
            + config.skill_subgoal_dim
            + config.skill_observed_event_dim
            + config.skill_validity_dim
        )
        self.mode_in = nnx.Linear(config.skill_code_dim, width, rngs=rngs)
        self.mode_out = nnx.Linear(width, width, rngs=rngs)
        self.instance_in = nnx.Linear(instance_dim, width, rngs=rngs)
        self.instance_out = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, condition: SkillCondition):
        mode = self.mode_out(nnx.swish(self.mode_in(condition.z_coordinates)))
        instance_input = jnp.concatenate(
            [
                condition.residual,
                condition.subgoal_embedding,
                condition.observed_event,
                condition.validity,
            ],
            axis=-1,
        )
        instance = self.instance_out(nnx.swish(self.instance_in(instance_input)))
        return mode[:, None, :], instance[:, None, :]


class CaskPi0(Pi0):
    """Pi0/Pi0.5 with an opt-in CASK path.

    The inherited compute_loss and sample_actions remain untouched and are the
    direct-Pi baseline. CASK training calls compute_cask_loss explicitly.
    """

    def __init__(self, config: CaskPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        action_expert_width = gemma.get_config(config.action_expert_variant).width
        self.skill_projector = SkillTokenProjector(config, action_expert_width, rngs=rngs)

    def embed_cask_suffix(
        self,
        observation: openpi_model.Observation,
        noisy_actions,
        timestep,
        condition: SkillCondition,
    ):
        base_tokens, base_input_mask, base_ar_mask, adarms_cond = super().embed_suffix(
            observation, noisy_actions, timestep
        )
        mode_token, instance_token = self.skill_projector(condition)
        skill_tokens = jnp.concatenate([mode_token, instance_token], axis=1)

        if condition.token_mask.shape != (noisy_actions.shape[0], 2):
            raise ValueError(
                "SkillCondition.token_mask must have shape [batch, 2], "
                f"got {condition.token_mask.shape}."
            )

        insert_at = skill_insert_index(pi05=self.pi05)
        tokens = jnp.concatenate(
            [base_tokens[:, :insert_at], skill_tokens, base_tokens[:, insert_at:]],
            axis=1,
        )
        input_mask = jnp.concatenate(
            [
                base_input_mask[:, :insert_at],
                condition.token_mask.astype(jnp.bool_),
                base_input_mask[:, insert_at:],
            ],
            axis=1,
        )
        ar_mask = jnp.concatenate(
            [
                base_ar_mask[:insert_at],
                jnp.asarray([True, False], dtype=jnp.bool_),
                base_ar_mask[insert_at:],
            ],
            axis=0,
        )
        return tokens, input_mask, ar_mask, adarms_cond

    def compute_cask_loss(
        self,
        rng,
        observation: openpi_model.Observation,
        actions,
        condition: SkillCondition,
        action_mask,
        *,
        train: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = openpi_model.preprocess_observation(preprocess_rng, observation, train=train)

        if actions.shape != action_mask.shape:
            raise ValueError(
                f"actions and action_mask must match; got {actions.shape} and {action_mask.shape}."
            )
        mask = action_mask.astype(actions.dtype)
        actions = actions * mask
        noise = jax.random.normal(noise_rng, actions.shape) * mask
        batch_shape = actions.shape[:-2]
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = (time_expanded * noise + (1 - time_expanded) * actions) * mask
        target_velocity = (noise - actions) * mask

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_cask_suffix(
            observation, x_t, time, condition
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        predicted_velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :]) * mask
        return masked_flow_mse(predicted_velocity, target_velocity, action_mask)

    def sample_cask_actions(
        self,
        rng,
        observation: openpi_model.Observation,
        condition: SkillCondition,
        action_mask,
        *,
        num_steps: int = 10,
        noise=None,
    ):
        observation = openpi_model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        if action_mask.shape != expected_shape:
            raise ValueError(f"Expected action_mask shape {expected_shape}, got {action_mask.shape}.")
        if noise is None:
            noise = jax.random.normal(rng, expected_shape)
        if noise.shape != expected_shape:
            raise ValueError(f"Expected noise shape {expected_shape}, got {noise.shape}.")
        mask = action_mask.astype(noise.dtype)
        noise = noise * mask
        dt = -1.0 / num_steps

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attention_mask,
            positions=prefix_positions,
        )

        def step(carry):
            x_t, time = carry
            x_t = x_t * mask
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_cask_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                condition,
            )
            suffix_attention_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(
                prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attention_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attention_mask],
                axis=-1,
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attention_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise AssertionError("The cached-prefix call unexpectedly returned prefix outputs.")
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :]) * mask
            next_x = (x_t + dt * velocity) * mask
            return next_x, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions * mask
