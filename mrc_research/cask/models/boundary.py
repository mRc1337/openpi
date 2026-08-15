"""Offline bidirectional event-boundary posterior."""

from __future__ import annotations

from flax import linen as nn

from cask.models.segment_encoder import TemporalEncoder


class OfflineBoundaryPosterior(nn.Module):
    width: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout_rate: float = 0.1
    max_steps: int = 1024

    @nn.compact
    def __call__(self, frame_event_features, time_mask, *, deterministic: bool):
        hidden = TemporalEncoder(
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            max_steps=self.max_steps,
            causal=False,
            name="bidirectional_tower",
        )(frame_event_features, time_mask, deterministic=deterministic)
        return nn.Dense(1, name="boundary_logit")(hidden)[..., 0]
