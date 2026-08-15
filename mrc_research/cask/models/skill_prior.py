"""Causal prior p(z, r | context, grounded subgoal)."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from cask.types import SkillPriorOutput


class SkillPrior(nn.Module):
    fsq_levels: tuple[int, ...] = (3, 3, 3, 3)
    residual_dim: int = 16
    width: int = 256
    depth: int = 2

    @nn.compact
    def __call__(
        self,
        context_embedding,
        subgoal_embedding,
        *,
        teacher_integer_codes=None,
    ) -> SkillPriorOutput:
        if len(set(self.fsq_levels)) != 1:
            raise ValueError("The initial autoregressive prior expects equal FSQ levels.")
        level = self.fsq_levels[0]
        half = (level - 1) // 2

        hidden = jnp.concatenate([context_embedding, subgoal_embedding], axis=-1)
        for layer in range(self.depth):
            hidden = nn.Dense(self.width, name=f"context_mlp_{layer}")(hidden)
            hidden = nn.gelu(hidden)

        batch_size = hidden.shape[0]
        previous_onehot = jnp.zeros(
            (batch_size, len(self.fsq_levels) * level),
            dtype=hidden.dtype,
        )
        logits = []
        chosen_codes = []
        for coordinate in range(len(self.fsq_levels)):
            coordinate_hidden = jnp.concatenate([hidden, previous_onehot], axis=-1)
            coordinate_hidden = nn.Dense(
                self.width,
                name=f"coordinate_{coordinate}_hidden",
            )(coordinate_hidden)
            coordinate_hidden = nn.gelu(coordinate_hidden)
            coordinate_logits = nn.Dense(
                level,
                name=f"coordinate_{coordinate}_logits",
            )(coordinate_hidden)
            logits.append(coordinate_logits)

            if teacher_integer_codes is None:
                digit = jnp.argmax(coordinate_logits, axis=-1)
            else:
                digit = teacher_integer_codes[:, coordinate].astype(jnp.int32) + half
            digit = jnp.clip(digit, 0, level - 1)
            chosen_codes.append(digit - half)
            onehot = jax_one_hot(digit, level, hidden.dtype)
            start = coordinate * level
            previous_onehot = previous_onehot.at[:, start : start + level].set(onehot)

        integer_codes = jnp.stack(chosen_codes, axis=-1)
        z_coordinates = integer_codes.astype(hidden.dtype) / half
        residual_hidden = jnp.concatenate([hidden, z_coordinates], axis=-1)
        residual_hidden = nn.Dense(self.width, name="residual_hidden")(residual_hidden)
        residual_hidden = nn.gelu(residual_hidden)
        residual_mean = nn.Dense(self.residual_dim, name="residual_mean")(residual_hidden)
        residual_logvar = jnp.clip(
            nn.Dense(self.residual_dim, name="residual_logvar")(residual_hidden),
            -10.0,
            5.0,
        )
        return SkillPriorOutput(
            z_logits=jnp.stack(logits, axis=1),
            z_coordinates=z_coordinates,
            residual_mean=residual_mean,
            residual_logvar=residual_logvar,
        )


def jax_one_hot(indices, num_classes: int, dtype):
    return jnp.eye(num_classes, dtype=dtype)[indices]
