from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from cask.models.termination import CausalTerminationHead


def test_future_changes_do_not_change_past_hazards() -> None:
    batch, length, width, cutoff = 2, 6, 16, 3
    model = CausalTerminationHead(
        width=width,
        depth=2,
        num_heads=4,
        dropout_rate=0.0,
        max_steps=length,
    )
    context = jax.random.normal(jax.random.key(1), (batch, length, width))
    events = jax.random.normal(jax.random.key(2), (batch, length, 8))
    z = jnp.zeros((batch, 4))
    residual = jnp.zeros((batch, 16))
    subgoal = jnp.zeros((batch, width))
    elapsed = jnp.arange(length, dtype=jnp.float32)[None, :, None].repeat(batch, axis=0)
    time_mask = jnp.ones((batch, length), dtype=jnp.bool_)
    args = (context, events, z, residual, subgoal, elapsed, time_mask)
    variables = model.init(jax.random.key(0), *args, deterministic=True)
    original = model.apply(variables, *args, deterministic=True)

    changed_context = context.at[:, cutoff + 1 :].set(10_000)
    changed_events = events.at[:, cutoff + 1 :].set(-10_000)
    changed_args = (
        changed_context,
        changed_events,
        z,
        residual,
        subgoal,
        elapsed,
        time_mask,
    )
    changed = model.apply(variables, *changed_args, deterministic=True)
    np.testing.assert_allclose(
        np.asarray(original[:, : cutoff + 1]),
        np.asarray(changed[:, : cutoff + 1]),
        rtol=1e-5,
        atol=1e-5,
    )
