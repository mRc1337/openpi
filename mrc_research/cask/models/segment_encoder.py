"""Masked temporal Transformer used by the posterior and event models."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    width: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, attention_mask, *, deterministic: bool):
        residual = inputs
        hidden = nn.LayerNorm(name="attention_norm")(inputs)
        hidden = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=self.dropout_rate,
            name="self_attention",
        )(
            hidden,
            hidden,
            mask=attention_mask,
            deterministic=deterministic,
        )
        hidden = nn.Dropout(rate=self.dropout_rate, name="attention_dropout")(
            hidden,
            deterministic=deterministic,
        )
        inputs = residual + hidden

        residual = inputs
        hidden = nn.LayerNorm(name="mlp_norm")(inputs)
        hidden = nn.Dense(self.width * self.mlp_ratio, name="mlp_in")(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(rate=self.dropout_rate, name="mlp_hidden_dropout")(
            hidden,
            deterministic=deterministic,
        )
        hidden = nn.Dense(self.width, name="mlp_out")(hidden)
        hidden = nn.Dropout(rate=self.dropout_rate, name="mlp_output_dropout")(
            hidden,
            deterministic=deterministic,
        )
        return residual + hidden


class TemporalEncoder(nn.Module):
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 128
    causal: bool = False

    @nn.compact
    def __call__(self, features, time_mask, *, deterministic: bool):
        if features.ndim != 3:
            raise ValueError(f"Expected features [B,L,D], got {features.shape}.")
        if time_mask.shape != features.shape[:2]:
            raise ValueError(
                f"time_mask must match [B,L]; got {time_mask.shape} for {features.shape}."
            )
        if features.shape[1] > self.max_steps:
            raise ValueError(
                f"Segment length {features.shape[1]} exceeds configured max_steps={self.max_steps}."
            )
        if self.width % self.num_heads != 0:
            raise ValueError("width must be divisible by num_heads.")

        hidden = nn.Dense(self.width, name="input_projection")(features)
        position = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_steps, self.width),
        )
        hidden = hidden + position[None, : features.shape[1]]
        hidden = hidden * time_mask[..., None].astype(hidden.dtype)

        attention_mask = nn.make_attention_mask(time_mask, time_mask, dtype=jnp.bool_)
        if self.causal:
            causal_mask = nn.make_causal_mask(time_mask, dtype=jnp.bool_)
            attention_mask = nn.combine_masks(attention_mask, causal_mask)

        for layer in range(self.depth):
            hidden = TransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f"block_{layer}",
            )(hidden, attention_mask, deterministic=deterministic)
            hidden = hidden * time_mask[..., None].astype(hidden.dtype)
        return nn.LayerNorm(name="output_norm")(hidden)


class MaskedAttentionPool(nn.Module):
    @nn.compact
    def __call__(self, sequence, time_mask):
        scores = nn.Dense(1, name="score")(sequence)[..., 0]
        scores = jnp.where(time_mask, scores, jnp.finfo(scores.dtype).min)
        weights = nn.softmax(scores, axis=-1)
        weights = jnp.where(time_mask, weights, 0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
        return jnp.sum(sequence * weights[..., None], axis=1)


class SegmentEncoder(nn.Module):
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 128

    @nn.compact
    def __call__(self, features, time_mask, *, deterministic: bool):
        sequence = TemporalEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            causal=False,
            name="temporal",
        )(features, time_mask, deterministic=deterministic)
        return MaskedAttentionPool(name="pool")(sequence, time_mask)
