"""Trajectory posterior q(z, r | segment) with restricted z/r views."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from cask.models.fsq import FiniteScalarQuantizer
from cask.models.segment_encoder import SegmentEncoder
from cask.types import SkillPosterior


class HybridSkillTokenizer(nn.Module):
    fsq_levels: tuple[int, ...] = (3, 3, 3, 3)
    residual_dim: int = 16
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 128
    residual_dropout_rate: float = 0.2

    @nn.compact
    def __call__(
        self,
        z_features,
        r_features,
        time_mask,
        *,
        deterministic: bool,
        sample_residual: bool = True,
    ) -> SkillPosterior:
        if z_features.shape[:2] != r_features.shape[:2]:
            raise ValueError("z_features and r_features must share [B,L].")
        if time_mask.shape != z_features.shape[:2]:
            raise ValueError("time_mask must match the segment batch and length.")

        z_summary = SegmentEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            name="z_encoder",
        )(z_features, time_mask, deterministic=deterministic)
        z_prequantized = nn.Dense(len(self.fsq_levels), name="z_projection")(z_summary)
        fsq = FiniteScalarQuantizer(self.fsq_levels, name="fsq")(z_prequantized)

        repeated_z = jnp.broadcast_to(
            fsq.quantized[:, None, :],
            (*r_features.shape[:2], len(self.fsq_levels)),
        )
        r_input = jnp.concatenate([r_features, repeated_z], axis=-1)
        r_summary = SegmentEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            name="r_encoder",
        )(r_input, time_mask, deterministic=deterministic)
        residual_mean = nn.Dense(self.residual_dim, name="residual_mean")(r_summary)
        residual_logvar = jnp.clip(
            nn.Dense(self.residual_dim, name="residual_logvar")(r_summary),
            -10.0,
            5.0,
        )

        if sample_residual:
            epsilon = jax.random.normal(self.make_rng("latent"), residual_mean.shape)
            residual_sample = residual_mean + jnp.exp(0.5 * residual_logvar) * epsilon
        else:
            residual_sample = residual_mean

        if not deterministic and self.residual_dropout_rate > 0:
            keep_probability = 1.0 - self.residual_dropout_rate
            keep = jax.random.bernoulli(
                self.make_rng("residual_dropout"),
                keep_probability,
                (residual_sample.shape[0], 1),
            )
            residual_sample = (
                residual_sample * keep.astype(residual_sample.dtype) / keep_probability
            )

        return SkillPosterior(
            z_continuous=fsq.continuous,
            z_quantized=fsq.quantized,
            z_index=fsq.index,
            residual_mean=residual_mean,
            residual_logvar=residual_logvar,
            residual_sample=residual_sample,
        )
