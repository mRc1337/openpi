from __future__ import annotations

from cask.pi_layout import cask_suffix_block_mask


def test_pi0_suffix_block_mask() -> None:
    assert cask_suffix_block_mask(pi05=False, action_horizon=3) == (
        True,
        True,
        False,
        True,
        False,
        False,
    )


def test_pi05_suffix_block_mask() -> None:
    assert cask_suffix_block_mask(pi05=True, action_horizon=3) == (
        True,
        False,
        True,
        False,
        False,
    )
