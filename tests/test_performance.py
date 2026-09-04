from dataclasses import replace
from pathlib import Path

import torch

from rift_v4.config import V4Config
from rift_v4.model import RIFTV4
from rift_v4.performance import (
    _is_float8_linear,
    apply_selective_float8_training,
)


def test_selective_float8_targets_only_attention_and_ffn_matrices() -> None:
    assert _is_float8_linear("blocks.0.attention.qkv")
    assert _is_float8_linear("blocks.0.attention.output")
    assert _is_float8_linear("blocks.0.feed_forward.input")
    assert _is_float8_linear("blocks.0.feed_forward.output")
    assert not _is_float8_linear("content_input")
    assert not _is_float8_linear("blocks.0.modulation.1")
    assert not _is_float8_linear("output")


def test_disabled_float8_keeps_canonical_model() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    performance = replace(config.performance, float8_training=False)
    model = RIFTV4(8, 16, 2, 32, 2, 8, 256, 5)
    state_keys = tuple(model.state_dict())

    summary = apply_selective_float8_training(
        model, performance, torch.device("cpu")
    )

    assert not summary["enabled"]
    assert tuple(model.state_dict()) == state_keys
