"""Masked multimodal grounded-subgoal proposals and feasibility."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from cask.types import SubgoalProposals


class GroundedSubgoalHead(nn.Module):
    geometry_dim: int
    visual_dim: int = 128
    contact_dim: int = 7
    embedding_dim: int = 256
    num_proposals: int = 3

    @nn.compact
    def __call__(self, context_embedding, modality_mask) -> SubgoalProposals:
        if modality_mask.shape != (context_embedding.shape[0], 3):
            raise ValueError("modality_mask must be [B,3] for geometry/visual/contact.")
        query = self.param(
            "proposal_queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_proposals, self.embedding_dim),
        )
        context = nn.Dense(self.embedding_dim, name="context_projection")(context_embedding)
        hidden = nn.tanh(context[:, None, :] + query[None, :, :])

        geometry = nn.Dense(self.geometry_dim, name="geometry_head")(hidden)
        visual = nn.Dense(self.visual_dim, name="visual_head")(hidden)
        contact_logits = nn.Dense(self.contact_dim, name="contact_head")(hidden)
        score = nn.Dense(1, name="score_head")(hidden)[..., 0]
        feasibility_logit = nn.Dense(1, name="feasibility_head")(hidden)[..., 0]

        geometry_embedding = nn.Dense(self.embedding_dim, name="geometry_projection")(geometry)
        visual_embedding = nn.Dense(self.embedding_dim, name="visual_projection")(visual)
        contact_embedding = nn.Dense(
            self.embedding_dim,
            name="contact_projection",
        )(contact_logits)
        modality_embeddings = jnp.stack(
            [geometry_embedding, visual_embedding, contact_embedding],
            axis=2,
        )
        mask = modality_mask[:, None, :, None].astype(modality_embeddings.dtype)
        fused = jnp.sum(modality_embeddings * mask, axis=2) / jnp.maximum(
            jnp.sum(mask, axis=2),
            1,
        )
        mask_embedding_table = self.param(
            "modality_presence_embedding",
            nn.initializers.normal(stddev=0.02),
            (3, self.embedding_dim),
        )
        presence_embedding = jnp.einsum(
            "bm,md->bd",
            modality_mask.astype(fused.dtype),
            mask_embedding_table,
        )
        fused = nn.LayerNorm(name="fusion_norm")(fused + presence_embedding[:, None, :])
        has_target = jnp.any(modality_mask, axis=-1)[:, None]
        feasibility = jnp.where(has_target, jax.nn.sigmoid(feasibility_logit), 0)
        return SubgoalProposals(
            geometry=geometry,
            visual=visual,
            contact_logits=contact_logits,
            embedding=fused,
            score=score,
            feasibility=feasibility,
            modality_mask=modality_mask,
        )
