from pathlib import Path

import pytest
import torch

from rift_v4.config import V4Config
from rift_v4.vocoder import official_generator_config, official_generator_state


def test_official_pc_nsf_config_matches_v4_contract() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    generator = official_generator_config(config)

    assert generator["mini_nsf"] is True
    assert generator["noise_sigma"] == 0.0
    assert generator["upsample_rates"] == [8, 8, 2, 2, 2]
    assert generator["sampling_rate"] == 44_100
    assert generator["num_mels"] == 128
    assert generator["hop_size"] == 512


def test_official_pc_nsf_state_extracts_only_generator() -> None:
    weight = torch.ones(2)
    state = official_generator_state(
        {
            "state_dict": {
                "generator.layer.weight": weight,
                "discriminator.layer.weight": torch.zeros(2),
            }
        }
    )

    assert set(state) == {"layer.weight"}
    assert state["layer.weight"] is weight


def test_official_pc_nsf_state_rejects_exported_or_empty_payload() -> None:
    with pytest.raises(ValueError, match="state_dict"):
        official_generator_state({"generator": {}})
    with pytest.raises(ValueError, match="no generator"):
        official_generator_state({"state_dict": {"other.weight": torch.ones(1)}})
