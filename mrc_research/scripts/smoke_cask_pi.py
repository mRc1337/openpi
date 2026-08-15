"""Dummy-model smoke test for the CASK Pi0.5 integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cask.models.pi0_skill_adapter import CaskPi0Config
from cask.types import SkillCondition


def main() -> None:
    batch_size = 2
    print("cask-pi-smoke: building config", flush=True)
    config = CaskPi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
    )
    print("cask-pi-smoke: creating dummy model", flush=True)
    model = config.create(jax.random.key(0))
    print("cask-pi-smoke: creating fake batch", flush=True)
    observation = config.fake_obs(batch_size=batch_size)
    actions = config.fake_act(batch_size=batch_size)
    condition = SkillCondition(
        z_coordinates=jnp.zeros((batch_size, config.skill_code_dim)),
        residual=jnp.zeros((batch_size, config.skill_residual_dim)),
        subgoal_embedding=jnp.zeros((batch_size, config.skill_subgoal_dim)),
        observed_event=jnp.zeros((batch_size, config.skill_observed_event_dim)),
        validity=jnp.ones((batch_size, config.skill_validity_dim)),
        token_mask=jnp.ones((batch_size, 2), dtype=jnp.bool_),
    )
    action_mask = jnp.zeros_like(actions, dtype=jnp.bool_)
    action_mask = action_mask.at[..., :7].set(True)
    print("cask-pi-smoke: computing masked flow loss", flush=True)
    per_timestep, time_valid = model.compute_cask_loss(
        jax.random.key(1),
        observation,
        actions,
        condition,
        action_mask,
        train=False,
    )
    assert per_timestep.shape == (batch_size, config.action_horizon)
    assert time_valid.shape == per_timestep.shape

    print("cask-pi-smoke: sampling two ODE steps", flush=True)
    sampled = model.sample_cask_actions(
        jax.random.key(2),
        observation,
        condition,
        action_mask,
        num_steps=2,
    )
    assert sampled.shape == actions.shape
    np.testing.assert_array_equal(np.asarray(sampled[..., 7:]), 0)
    print("cask-pi-smoke: PASS")


if __name__ == "__main__":
    main()
