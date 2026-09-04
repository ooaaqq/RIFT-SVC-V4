import pytest

from rift_v4.speaker_metric_calibration import normalized_transfer_progress


def test_normalized_transfer_progress() -> None:
    assert normalized_transfer_progress(-0.2, -0.4, 0.4) == pytest.approx(0.25)
    assert normalized_transfer_progress(0.4, -0.4, 0.4) == pytest.approx(1.0)
    assert normalized_transfer_progress(0.0, 0.1, 0.1) is None
    assert normalized_transfer_progress(0.0, 0.2, 0.1) is None
