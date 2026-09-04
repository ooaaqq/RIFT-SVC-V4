import pytest
import torch

from rift_v4.shadow_identity_diagnostic import (
    clustered_mean_interval,
    correlation,
    orthonormal_dct,
    paired_model_delta,
    voice_group,
)


def test_dct_is_orthonormal() -> None:
    matrix = orthonormal_dct(16)
    assert torch.allclose(matrix @ matrix.T, torch.eye(16), atol=1e-5)


def test_correlation_and_voice_group() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert correlation(values, values) == pytest.approx(1.0)
    assert voice_group("M4Singer:Soprano-1") == "female"
    assert voice_group("GTSinger:German-DE-Tenor-1") == "male"


def test_clustered_interval_and_paired_delta() -> None:
    metadata = [
        {"song_key": "song-a"},
        {"song_key": "song-a"},
        {"song_key": "song-b"},
    ]
    interval = clustered_mean_interval([1.0, 3.0, 5.0], metadata, 100, 7)
    assert interval[0] <= 3.0 <= interval[1]

    def model(active_mse: float, similarity: float) -> dict[str, float]:
        return {
            "active_raw_mse": active_mse,
            "source_similarity": similarity,
            "unrelated_similarity_mean": similarity - 0.1,
            "source_margin_mean": 0.1,
            "wrong_similarity": similarity - 0.2,
            "wrong_margin": -0.2,
        }

    rows = [
        {"models": {"before": model(2.0, 0.5), "after": model(1.0, 0.6)}},
        {"models": {"before": model(3.0, 0.6), "after": model(2.0, 0.7)}},
        {"models": {"before": model(4.0, 0.7), "after": model(3.0, 0.8)}},
    ]
    delta = paired_model_delta(rows, metadata, "before", "after", 100, 9)
    assert delta["active_raw_mse"]["mean"] == pytest.approx(-1.0)
    assert delta["active_raw_mse"]["win_rate_after"] == 1.0
    assert delta["source_similarity"]["mean"] == pytest.approx(0.1)
