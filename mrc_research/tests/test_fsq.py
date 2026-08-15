from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from cask.models.fsq import codes_to_index
from cask.models.fsq import index_to_codes


def test_all_default_codes_round_trip() -> None:
    levels = (3, 3, 3, 3)
    indexes = jnp.arange(81)
    codes = index_to_codes(indexes, levels)
    recovered = codes_to_index(codes, levels)
    np.testing.assert_array_equal(recovered, indexes)
    assert set(np.asarray(codes).reshape(-1).tolist()) == {-1, 0, 1}
