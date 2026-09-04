from __future__ import annotations

import math

import pytest
import torch

from scripts.audit_silence_conditioning import erode_mask, rms_dbfs
from scripts.compare_fixed_endpoints import paired_summary


def test_erode_mask_removes_boundaries() -> None:
    mask = torch.tensor([False, True, True, True, True, True, False])
    assert erode_mask(mask, 1).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_rms_dbfs_uses_power_mean() -> None:
    values = torch.tensor([0.1, 0.01, 1.0])
    metrics = rms_dbfs(values, torch.tensor([True, True, False]))
    expected = 20 * math.log10(math.sqrt((0.1**2 + 0.01**2) / 2))
    assert metrics["frames"] == 2
    assert metrics["rms_dbfs"] == pytest.approx(expected)


def test_paired_summary_matches_entries_by_id() -> None:
    before = [
        {"entry_id": "a", "full_raw_mse": 2.0},
        {"entry_id": "b", "full_raw_mse": 1.0},
    ]
    after = [
        {"entry_id": "b", "full_raw_mse": 0.5},
        {"entry_id": "a", "full_raw_mse": 3.0},
    ]
    result = paired_summary(before, after, "full_raw_mse")
    assert result["paired_median_delta"] == pytest.approx(0.25)
    assert result["win_rate"] == 0.5
