import pytest
import torch

from rift_v4.checkpoint_audit import (
    _balanced_panel_indices,
    _structure_at_t,
    _validate_t_values,
    _velocity_metrics,
)
from rift_v4.manifest import ManifestEntry
from rift_v4.model import RIFTV4


def _entry(identifier: str, dataset: str) -> ManifestEntry:
    return ManifestEntry(
        id=identifier,
        dataset=dataset,
        speaker="speaker",
        song=identifier,
        audio_path=f"/{identifier}.wav",
        feature_prefix=f"/{identifier}",
        frames=8,
        duration_seconds=1,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        quality_status="accepted",
    )


def test_velocity_metrics_identify_exact_direction_and_scale() -> None:
    target = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    metrics = _velocity_metrics(target, target, torch.ones(1, 2, dtype=torch.bool))
    assert metrics["mse"] == 0
    assert metrics["cosine"] == pytest.approx(1)
    assert metrics["prediction_rms"] == pytest.approx(metrics["target_rms"])


def test_balanced_panel_is_deterministic_and_dataset_balanced() -> None:
    entries = [_entry(f"a-{index}", "A") for index in range(9)] + [
        _entry(f"b-{index}", "B") for index in range(3)
    ]
    selected = _balanced_panel_indices(entries, 2)
    assert selected == _balanced_panel_indices(entries, 2)
    assert [entries[index].dataset for index in selected] == ["A", "A", "B", "B"]


def test_audit_timesteps_must_be_inside_open_unit_interval() -> None:
    _validate_t_values([0.05, 0.5, 0.95])
    with pytest.raises(ValueError, match="inside"):
        _validate_t_values([0.0, 0.5])


def test_structure_audit_returns_finite_per_layer_metrics() -> None:
    model = RIFTV4(8, 16, 2, 32, 2, 8, 64, 5).eval()
    mask = torch.tensor([[True] * 7 + [False]])
    batch = {
        "mel": torch.randn(1, 8, 8),
        "content": torch.randn(1, 8, 16),
        "f0": torch.rand(1, 8, 1) * 400,
        "rms": torch.rand(1, 8, 1),
        "speaker": torch.tensor([0]),
        "mask": mask,
    }
    result = _structure_at_t(
        model, batch, torch.randn_like(batch["mel"]), 0.5, use_bf16=False
    )
    assert len(result["layers"]) == 2
    for layer in result["layers"]:
        assert 0 <= layer["attention_entropy_median"] <= 1
        assert 0 <= layer["attention_max_probability_median"] <= 1
        assert layer["attn_residual_rms"] >= 0
        assert layer["ffn_residual_rms"] >= 0
        assert layer["attn_residual_ratio"] == pytest.approx(
            layer["attn_residual_rms"] / max(layer["stream_pre_attn_rms"], 1e-12)
        )
        assert layer["ffn_residual_ratio"] == pytest.approx(
            layer["ffn_residual_rms"] / max(layer["stream_pre_ffn_rms"], 1e-12)
        )
        assert all(
            torch.isfinite(torch.tensor(value))
            for key, value in layer.items()
            if key != "layer"
        )
