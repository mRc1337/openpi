"""Finite Scalar Quantization for the shared discrete skill mode."""

from __future__ import annotations

import math

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class FSQOutput:
    continuous: jax.Array
    quantized: jax.Array
    integer_codes: jax.Array
    index: jax.Array


def _validate_levels(levels: tuple[int, ...]) -> None:
    if not levels:
        raise ValueError("FSQ needs at least one scalar coordinate.")
    if any(level < 3 or level % 2 == 0 for level in levels):
        raise ValueError("This implementation supports odd levels >= 3.")


def codes_to_index(integer_codes, levels: tuple[int, ...]):
    _validate_levels(levels)
    half = jnp.asarray([(level - 1) // 2 for level in levels], dtype=jnp.int32)
    digits = integer_codes.astype(jnp.int32) + half
    basis = jnp.asarray(
        [math.prod(levels[:coordinate]) for coordinate in range(len(levels))],
        dtype=jnp.int32,
    )
    return jnp.sum(digits * basis, axis=-1)


def index_to_codes(index, levels: tuple[int, ...]):
    _validate_levels(levels)
    index = jnp.asarray(index, dtype=jnp.int32)
    digits = []
    remaining = index
    for level in levels:
        digits.append(remaining % level)
        remaining = remaining // level
    digits = jnp.stack(digits, axis=-1)
    half = jnp.asarray([(level - 1) // 2 for level in levels], dtype=jnp.int32)
    return digits - half


class FiniteScalarQuantizer(nn.Module):
    """Odd-level FSQ with a straight-through estimator.

    The default levels (3, 3, 3, 3) produce four coordinates in {-1, 0, 1}
    and 81 possible joint codes.
    """

    levels: tuple[int, ...] = (3, 3, 3, 3)

    @nn.compact
    def __call__(self, inputs) -> FSQOutput:
        _validate_levels(self.levels)
        if inputs.shape[-1] != len(self.levels):
            raise ValueError(
                f"Expected {len(self.levels)} FSQ coordinates, got {inputs.shape[-1]}."
            )
        half = jnp.asarray([(level - 1) // 2 for level in self.levels], dtype=inputs.dtype)
        continuous = jnp.tanh(inputs)
        integer_codes = jnp.round(continuous * half)
        quantized = integer_codes / half
        straight_through = continuous + jax.lax.stop_gradient(quantized - continuous)
        return FSQOutput(
            continuous=continuous,
            quantized=straight_through,
            integer_codes=integer_codes.astype(jnp.int32),
            index=codes_to_index(integer_codes, self.levels),
        )
