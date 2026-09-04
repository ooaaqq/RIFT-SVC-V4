import pytest

from rift_v4.manifest import ManifestEntry
from rift_v4.shadow_panel import (
    deterministic_start,
    paired_statistics,
    select_shadow_entries,
)


def _entry(speaker: str, song: str, index: int) -> ManifestEntry:
    return ManifestEntry(
        id=f"{speaker}-{song}-{index}",
        dataset="D",
        speaker=speaker,
        song=song,
        audio_path=f"/{speaker}/{song}/{index}.wav",
        feature_prefix=f"/{speaker}/{song}/{index}",
        frames=1000,
        duration_seconds=10.0,
        sample_rate=44_100,
        channels=1,
        audio_sha256=f"{index + 1:064x}",
        split="validation",
        quality_status="accepted",
    )


def test_shadow_selection_prioritizes_unique_speaker_songs() -> None:
    entries = [
        _entry(speaker, f"song-{song}", index)
        for index, (speaker, song) in enumerate(
            (speaker, song)
            for speaker in ("A", "B", "C")
            for song in range(3)
        )
    ]
    entries.extend((_entry("A", "song-0", 20), _entry("B", "song-1", 21)))
    speakers = {f"D:{speaker}": index for index, speaker in enumerate(("A", "B", "C"))}

    selected = select_shadow_entries(entries, speakers, set(), 10, 768, 7)

    assert len(selected) == 10
    assert len({entry.id for _, entry in selected}) == 10
    assert len({(entry.speaker_key, entry.song) for _, entry in selected[:9]}) == 9
    assert {entry.speaker for _, entry in selected} == {"A", "B", "C"}


def test_deterministic_start_is_bounded() -> None:
    assert deterministic_start(7, "entry", 0, 0) == 0
    assert deterministic_start(7, "entry", 0, 99) == deterministic_start(
        7, "entry", 0, 99
    )
    assert 0 <= deterministic_start(7, "entry", 0, 99) <= 99
    with pytest.raises(ValueError):
        deterministic_start(7, "entry", 0, -1)


def test_paired_statistics_are_grouped_by_song() -> None:
    metadata = [
        {"song_key": "D:song-a", "speaker": "D:A"},
        {"song_key": "D:song-a", "speaker": "D:A"},
        {"song_key": "D:song-b", "speaker": "D:B"},
    ]
    result = paired_statistics(
        [1.0, 2.0, 3.0],
        [0.5, 2.5, 2.0],
        metadata,
        bootstrap_samples=100,
        seed=9,
    )

    assert result["samples"] == 3
    assert result["song_units"] == 2
    assert result["win_rate_after"] == pytest.approx(2 / 3)
    assert result["median_delta"] == pytest.approx(-0.5)
