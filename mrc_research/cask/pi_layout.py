"""Pure, dependency-free Pi suffix layout helpers."""

from __future__ import annotations


def skill_insert_index(*, pi05: bool) -> int:
    return 0 if pi05 else 1


def cask_suffix_block_mask(*, pi05: bool, action_horizon: int) -> tuple[bool, ...]:
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    state_prefix = () if pi05 else (True,)
    skill_block = (True, False)
    action_block = (True,) + (False,) * (action_horizon - 1)
    return state_prefix + skill_block + action_block
