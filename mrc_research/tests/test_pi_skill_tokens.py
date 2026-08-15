from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("openpi")

from cask.models.pi0_skill_adapter import CaskPi0Config
from cask.pi_layout import cask_suffix_block_mask
from cask.types import SkillCondition
from openpi.models.pi0_config import Pi0Config


def make_condition(config: CaskPi0Config, batch_size: int) -> SkillCondition:
    return SkillCondition(
        z_coordinates=jnp.zeros((batch_size, config.skill_code_dim)),
        residual=jnp.zeros((batch_size, config.skill_residual_dim)),
        subgoal_embedding=jnp.zeros((batch_size, config.skill_subgoal_dim)),
        observed_event=jnp.zeros((batch_size, config.skill_observed_event_dim)),
        validity=jnp.ones((batch_size, config.skill_validity_dim)),
        token_mask=jnp.ones((batch_size, 2), dtype=jnp.bool_),
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_skill_suffix_layout(pi05: bool) -> None:
    batch_size, horizon = 2, 4
    config = CaskPi0Config(
        pi05=pi05,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=horizon,
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=batch_size)
    actions = config.fake_act(batch_size=batch_size)
    tokens, input_mask, ar_mask, _ = model.embed_cask_suffix(
        observation,
        actions,
        jnp.full((batch_size,), 0.5),
        make_condition(config, batch_size),
    )
    expected_length = horizon + 2 + (0 if pi05 else 1)
    assert tokens.shape[1] == expected_length
    assert input_mask.shape == (batch_size, expected_length)
    np.testing.assert_array_equal(
        np.asarray(ar_mask),
        cask_suffix_block_mask(pi05=pi05, action_horizon=horizon),
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_standard_pi_path_is_unchanged(pi05: bool) -> None:
    horizon = 4
    base_config = Pi0Config(
        pi05=pi05,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=horizon,
    )
    cask_config = CaskPi0Config(
        pi05=pi05,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=horizon,
    )
    init_key = jax.random.key(0)
    base_model = base_config.create(init_key)
    cask_model = cask_config.create(init_key)
    observation = base_config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(1), (1, horizon, base_config.action_dim))
    base_actions = base_model.sample_actions(
        jax.random.key(2),
        observation,
        noise=noise,
        num_steps=2,
    )
    cask_actions = cask_model.sample_actions(
        jax.random.key(2),
        observation,
        noise=noise,
        num_steps=2,
    )
    np.testing.assert_allclose(base_actions, cask_actions, rtol=1e-5, atol=1e-5)
