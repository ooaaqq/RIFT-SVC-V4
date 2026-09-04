import pytest
import torch

from rift_v4.output_head_audit import projected_matrix_report
from rift_v4.spectral_detail_audit import (
    band_name,
    coefficient_metrics,
    target_raw_to_v3,
    velocity_metrics,
)
from rift_v4.spectral_pitch_audit import build_strata


def test_v3_normalization_round_trip() -> None:
    raw = torch.tensor([-12.0, -5.0, 2.0])
    normalized = target_raw_to_v3(raw)
    assert torch.allclose((normalized + 1.0) * 7.0 - 12.0, raw)


def test_coefficient_and_velocity_metrics() -> None:
    target = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    prediction = target * 0.5
    residual = prediction - target
    selected = torch.tensor([True, True])
    endpoint = coefficient_metrics(residual, target, prediction, selected, 4)
    assert endpoint is not None
    assert endpoint["rms_ratio"] == pytest.approx(0.5)
    assert endpoint["cosine"] == pytest.approx(1.0)

    velocity = velocity_metrics(residual, target, prediction, selected)
    assert velocity is not None
    assert velocity["nmse"] == pytest.approx(0.25)
    assert velocity["rms_ratio"] == pytest.approx(0.5)
    assert band_name((16, 32)) == "dct_16_31"


def test_pitch_strata_partition_voiced_frames() -> None:
    target = torch.randn(2, 4, 128)
    tensors = {
        "f0": torch.tensor(
            [[[100.0], [101.0], [120.0], [0.0]], [[200.0], [205.0], [300.0], [0.0]]]
        ),
        "rms": torch.tensor(
            [[[0.1], [0.1], [0.1], [0.0]], [[0.1], [0.1], [0.1], [0.0]]]
        ),
    }
    strata, thresholds = build_strata(target, tensors, 4)
    assert sum(int(mask.sum()) for mask in strata["f0"].values()) == 6
    assert sum(int(mask.sum()) for mask in strata["pitch_motion"].values()) == 4
    assert len(thresholds["f0_hz_quartiles"]) == 3


def test_output_projection_report_covers_dct_bands() -> None:
    report = projected_matrix_report(torch.eye(128), torch.ones(128))
    assert set(report) == {"dct_0_15", "dct_16_31", "dct_32_127"}
    assert report["dct_32_127"]["relative_to_dct_0_15_mean"] == pytest.approx(
        1.0
    )
