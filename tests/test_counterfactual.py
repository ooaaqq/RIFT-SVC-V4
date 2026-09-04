from pathlib import Path

import pytest
import torch

from rift_v4.counterfactual import (
    aggregate_rows,
    centroid_cache_key,
    load_or_create_pairs,
    manifest_fingerprint,
    speaker_target_margin,
)
from rift_v4.manifest import ManifestEntry


def _entry(tmp_path: Path, speaker: str, song: str, index: int) -> ManifestEntry:
    prefix = tmp_path / f"{speaker}-{song}"
    torch.save(torch.full((8,), 100.0), f"{prefix}.f0.pt")
    torch.save(torch.ones(8), f"{prefix}.rms.pt")
    torch.save(torch.ones(8, 3), f"{prefix}.content.pt")
    return ManifestEntry(
        id=f"{speaker}-{song}",
        dataset="D",
        speaker=speaker,
        song=song,
        audio_path=str(prefix.with_suffix(".wav")),
        feature_prefix=str(prefix),
        frames=8,
        duration_seconds=1.0,
        sample_rate=44_100,
        channels=1,
        audio_sha256=f"{index:064x}",
        content_feature_path=f"{prefix}.content.pt",
        split="validation",
        quality_status="accepted",
    )


def test_pair_selection_is_deterministic_and_song_disjoint(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, speaker, f"song-{song}", index + 1)
        for index, (speaker, song) in enumerate(
            (speaker, song) for speaker in ("A", "B") for song in range(3)
        )
    ]
    speakers = {"D:A": 0, "D:B": 1}
    path = tmp_path / "pairs.json"
    first = load_or_create_pairs(path, entries, speakers, 2, 4, 2, 7)
    second = load_or_create_pairs(path, list(reversed(entries)), speakers, 1, 2, 1, 9)

    assert second == first
    assert len(first["pairs"]) == 2
    for pair in first["pairs"]:
        assert pair["source_speaker"] != pair["target_speaker"]
        source_songs = {item["song"] for item in pair["source_references"]}
        target_songs = {item["song"] for item in pair["target_references"]}
        assert pair["source"]["song"] not in source_songs
        assert pair["source"]["song"] not in target_songs

    assert first["schema_version"] == 2
    assert first["manifest_fingerprint_sha256"] == manifest_fingerprint(entries)


def test_pair_selection_excludes_locked_song_units(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, speaker, f"song-{song}", index + 1)
        for index, (speaker, song) in enumerate(
            (speaker, song) for speaker in ("A", "B") for song in range(4)
        )
    ]
    speakers = {"D:A": 0, "D:B": 1}
    spec = load_or_create_pairs(
        tmp_path / "excluded-pairs.json",
        entries,
        speakers,
        2,
        4,
        2,
        7,
        excluded_song_keys={"D:song-0"},
    )

    assert spec["excluded_song_keys"] == ["D:song-0"]
    for pair in spec["pairs"]:
        assert pair["source"]["song"] != "song-0"
        assert all(item["song"] != "song-0" for item in pair["source_references"])
        assert all(item["song"] != "song-0" for item in pair["target_references"])


def test_margin_and_condition_aggregation() -> None:
    rows = [
        {
            "condition": "target",
            "similarity_to_target": 0.8,
            "similarity_to_source": 0.3,
            "target_margin": 0.5,
            "content_cosine": 0.9,
            "voicing_f1": 0.95,
            "f0_cents_mae": 20.0,
        },
        {
            "condition": "target",
            "similarity_to_target": 0.6,
            "similarity_to_source": 0.4,
            "target_margin": 0.2,
            "content_cosine": 0.8,
            "voicing_f1": 0.85,
            "f0_cents_mae": None,
        },
    ]

    assert speaker_target_margin(0.8, 0.3) == pytest.approx(0.5)
    aggregate = aggregate_rows(rows)["target"]
    assert aggregate["target_margin"] == pytest.approx(0.35)
    assert aggregate["f0_cents_mae"] == pytest.approx(20.0)


def test_centroid_cache_key_includes_crop_coordinates() -> None:
    first = [{"id": "same", "start_frame": 10, "frames": 512}]
    second = [{"id": "same", "start_frame": 11, "frames": 512}]

    assert centroid_cache_key(first) != centroid_cache_key(second)
