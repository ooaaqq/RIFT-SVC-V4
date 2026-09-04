import pytest

from rift_v4.shadow_v3_compare import paired_statistics, summarize_rows


def _metadata() -> list[dict[str, object]]:
    return [
        {"song_key": "D:s1", "speaker": "D:A", "dataset": "D"},
        {"song_key": "D:s2", "speaker": "D:B", "dataset": "D"},
    ]


def test_shadow_v3_summary_includes_tail_macro_and_catastrophe() -> None:
    rows = [
        {
            "full_raw_mse": 1.0,
            "active_raw_mse": 2.0,
            "silence_raw_mse": None,
            "raw_l1": 0.5,
            "raw_cosine": 0.9,
        },
        {
            "full_raw_mse": 9.0,
            "active_raw_mse": 4.0,
            "silence_raw_mse": 10.0,
            "raw_l1": 1.5,
            "raw_cosine": 0.7,
        },
    ]
    summary = summarize_rows(rows, _metadata(), 5.0)

    assert summary["full_raw_mse"]["mean"] == pytest.approx(5.0)
    assert summary["full_raw_mse"]["p95"] == pytest.approx(8.6)
    assert summary["full_raw_mse"]["catastrophe_rate"] == pytest.approx(0.5)
    assert summary["silence_raw_mse"]["samples"] == 1
    assert summary["raw_l1"]["speaker_macro"] == pytest.approx(1.0)


def test_shadow_v3_paired_statistics_use_after_minus_before() -> None:
    before = [{"full_raw_mse": 2.0}, {"full_raw_mse": 4.0}]
    after = [{"full_raw_mse": 1.0}, {"full_raw_mse": 5.0}]
    result = paired_statistics(
        before, after, _metadata(), "full_raw_mse", 100, 7
    )

    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["median_delta"] == pytest.approx(0.0)
    assert result["win_rate_after"] == pytest.approx(0.5)
