from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from cask.masking import ActionSlotMap
from cask.masking import masked_flow_mse


def test_action_slot_round_trip() -> None:
    slot_map = ActionSlotMap((0, 3, 7))
    native = jnp.asarray([[[1.0, 2.0, 3.0]]])
    packed = slot_map.pack(native)
    np.testing.assert_allclose(slot_map.unpack(packed), native)
    assert packed.shape == (1, 1, 32)


def test_invalid_pi_slots_do_not_change_masked_loss() -> None:
    slot_map = ActionSlotMap((0, 3))
    native_validity = jnp.ones((1, 2, 2), dtype=jnp.bool_)
    mask = slot_map.pack_validity(native_validity)
    target = jnp.zeros((1, 2, 32))
    prediction_a = jnp.zeros_like(target).at[..., 0].set(1.0)
    prediction_b = prediction_a.at[..., 17].set(10_000.0)
    loss_a, valid_a = masked_flow_mse(prediction_a, target, mask)
    loss_b, valid_b = masked_flow_mse(prediction_b, target, mask)
    np.testing.assert_allclose(loss_a, loss_b)
    np.testing.assert_array_equal(valid_a, valid_b)
