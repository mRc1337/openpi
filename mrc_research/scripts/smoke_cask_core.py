"""Small CPU-compatible runtime smoke test for CASK core modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cask.models.boundary import OfflineBoundaryPosterior
from cask.models.hybrid_tokenizer import HybridSkillTokenizer
from cask.models.mvp_flow_decoder import MVPFlowDecoder
from cask.models.skill_prior import SkillPrior
from cask.models.subgoal import GroundedSubgoalHead
from cask.models.termination import CausalTerminationHead


def main() -> None:
    key = jax.random.key(0)
    batch, length, width = 2, 6, 16
    time_mask = jnp.asarray(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )

    tokenizer = HybridSkillTokenizer(
        width=width,
        depth=2,
        num_heads=4,
        max_steps=length,
        dropout_rate=0.0,
    )
    z_features = jnp.ones((batch, length, 12))
    r_features = jnp.ones((batch, length, 20))
    variables = tokenizer.init(
        {"params": key},
        z_features,
        r_features,
        time_mask,
        deterministic=True,
        sample_residual=False,
    )
    posterior = tokenizer.apply(
        variables,
        z_features,
        r_features,
        time_mask,
        deterministic=True,
        sample_residual=False,
    )
    assert posterior.z_quantized.shape == (batch, 4)
    assert posterior.residual_mean.shape == (batch, 16)

    context = jnp.ones((batch, width))
    subgoal = jnp.ones((batch, width))
    prior = SkillPrior(width=width, depth=1)
    variables = prior.init(
        key,
        context,
        subgoal,
        teacher_integer_codes=jnp.zeros((batch, 4), dtype=jnp.int32),
    )
    prior_output = prior.apply(variables, context, subgoal)
    assert prior_output.z_logits.shape == (batch, 4, 3)

    boundary = OfflineBoundaryPosterior(
        width=width,
        depth=2,
        num_heads=4,
        max_steps=length,
        dropout_rate=0.0,
    )
    event_features = jnp.ones((batch, length, 10))
    variables = boundary.init(
        key,
        event_features,
        time_mask,
        deterministic=True,
    )
    assert boundary.apply(
        variables,
        event_features,
        time_mask,
        deterministic=True,
    ).shape == (batch, length)

    termination = CausalTerminationHead(
        width=width,
        depth=2,
        num_heads=4,
        max_steps=length,
        dropout_rate=0.0,
    )
    termination_args = (
        jnp.ones((batch, length, width)),
        event_features,
        jnp.ones((batch, 4)),
        jnp.ones((batch, 16)),
        subgoal,
        jnp.arange(length, dtype=jnp.float32)[None, :, None].repeat(batch, axis=0),
        time_mask,
    )
    variables = termination.init(key, *termination_args, deterministic=True)
    assert termination.apply(
        variables,
        *termination_args,
        deterministic=True,
    ).shape == (batch, length)

    subgoal_head = GroundedSubgoalHead(
        geometry_dim=9,
        embedding_dim=width,
    )
    modality_mask = jnp.asarray([[True, True, True], [False, True, True]])
    variables = subgoal_head.init(key, context, modality_mask)
    proposals = subgoal_head.apply(variables, context, modality_mask)
    assert proposals.embedding.shape == (batch, 3, width)

    decoder = MVPFlowDecoder(
        action_dim=7,
        condition_dim=4 + 16 + width,
        width=width,
        depth=2,
        num_heads=4,
        max_steps=length,
        dropout_rate=0.0,
    )
    decoder_args = (
        jnp.ones((batch, length, 7)),
        jnp.full((batch,), 0.5),
        jnp.ones((batch, 8)),
        jnp.ones((batch, 4 + 16 + width)),
        time_mask,
    )
    variables = decoder.init(key, *decoder_args, deterministic=True)
    assert decoder.apply(
        variables,
        *decoder_args,
        deterministic=True,
    ).shape == (batch, length, 7)
    print("cask-core-smoke: PASS")


if __name__ == "__main__":
    main()
