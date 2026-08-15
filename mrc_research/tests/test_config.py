from __future__ import annotations

import pytest

from cask.config import CaskConfig
from cask.config import PiExecutorConfig


def test_default_config_is_structurally_valid() -> None:
    config = CaskConfig()
    config.validate()
    assert config.skill.num_combinations == 81


def test_m0_validation_rejects_unfrozen_robot_fields() -> None:
    with pytest.raises(ValueError, match="M0 has not frozen"):
        CaskConfig().validate(require_m0_fields=True)


def test_action_slots_are_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        PiExecutorConfig(native_action_slots=(0, 1, 1)).validate()
