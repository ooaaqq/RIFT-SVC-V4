from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rift_v4.config import V4Config
from rift_v4.datasets import ace_opencpop_identity
from rift_v4.features import (
    _feature_triplet_is_valid,
    compute_mel_stats,
    extract_auxiliary_features,
    merge_mel_stats,
)
from rift_v4.manifest import ManifestEntry
from rift_v4.split_cli import explicit_validation_split, song_disjoint_split


class FakePitch:
    def infer(self, audio, **kwargs):
        del audio
        return torch.full((1, int(kwargs["output_interp_target_length"]), 1), 220.0)


def _config() -> V4Config:
    return V4Config.load(Path(__file__).parents[1] / "config/v4.json")


def _entry(tmp_path: Path, index: int, song: str) -> ManifestEntry:
    prefix = tmp_path / f"feature-{index}"
    torch.save(torch.randn(20, 128), f"{prefix}.mel.pt")
    return ManifestEntry(
        id=f"id-{index}",
        dataset="A",
        speaker=f"speaker-{index % 2}",
        song=song,
        audio_path=str(tmp_path / f"{index}.wav"),
        feature_prefix=str(prefix),
        frames=20,
        duration_seconds=1,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        quality_status="accepted",
    )


def test_exact_features_align_and_stats_are_usable(tmp_path: Path) -> None:
    mel, f0, rms = extract_auxiliary_features(
        torch.randn(44_100), _config(), FakePitch()
    )
    assert mel.shape == (86, 128)
    assert f0.shape == rms.shape == (86,)
    entries = [_entry(tmp_path, index, f"song-{index}") for index in range(3)]
    stats = compute_mel_stats(entries, 128)
    value = torch.randn(5, 128)
    normalized = stats.normalize(value)
    torch.testing.assert_close(stats.denormalize(normalized), value)
    assert stats.frames == 60


def test_feature_cache_accepts_extractor_native_vectors(tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ("mel.pt", "f0.pt", "rms.pt"))
    torch.save(torch.randn(10, 128), paths[0])
    torch.save(torch.rand(10), paths[1])
    torch.save(torch.rand(10), paths[2])
    assert _feature_triplet_is_valid(paths, 128)


def test_merge_mel_stats_matches_weighted_population_statistics(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"mean": [1.0, 2.0], "std": [1.0, 1.0], "frames": 2}')
    second.write_text('{"mean": [3.0, 4.0], "std": [1.0, 1.0], "frames": 2}')
    result = merge_mel_stats([first, second])
    assert result.frames == 4
    assert result.mean == (2.0, 3.0)
    assert result.std == pytest.approx((2**0.5, 2**0.5))


def test_split_is_song_disjoint_and_deterministic(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "shared"),
        _entry(tmp_path, 1, "shared"),
        _entry(tmp_path, 2, "second"),
        _entry(tmp_path, 3, "third"),
        _entry(tmp_path, 4, "fourth"),
    ]
    first = song_disjoint_split(entries, 0.25, 0.25, 9)
    second = song_disjoint_split(entries, 0.25, 0.25, 9)
    assert [entry.split for entry in first] == [entry.split for entry in second]
    song_splits = {}
    for entry in first:
        song_splits.setdefault(entry.song, set()).add(entry.split)
    assert all(len(splits) == 1 for splits in song_splits.values())
    assert {entry.split for entry in first} == {"train", "validation", "test"}


def test_explicit_validation_split_uses_only_requested_songs(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, index, f"song-{index}") for index in range(6)]
    split = explicit_validation_split(entries, {"song-1", "song-4"})
    assert {entry.song for entry in split if entry.split == "validation"} == {
        "song-1",
        "song-4",
    }
    assert sum(entry.split == "train" for entry in split) == 4
    assert not any(entry.split == "test" for entry in split)


def test_explicit_validation_split_rejects_unknown_song(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absent from manifest"):
        explicit_validation_split([_entry(tmp_path, 0, "known")], {"unknown"})


def test_ace_opencpop_identity_uses_source_song() -> None:
    assert ace_opencpop_identity("acesinger_1#2001000001", "1") == (
        "1",
        "2001",
        "2001000001",
    )
    with pytest.raises(ValueError, match="invalid ACE-Opencpop identity"):
        ace_opencpop_identity("acesinger_2#2001000001", "1")


def test_opencpop_family_preserves_official_test_and_shared_split(
    tmp_path: Path,
) -> None:
    songs = ["2044", "2086", "2092", "2093", "2100", "2001", "2002", "2003"]
    entries = []
    for index, song in enumerate(songs):
        original = replace(
            _entry(tmp_path, index, song),
            dataset="Opencpop",
            speaker="official",
        )
        synthetic = replace(
            _entry(tmp_path, index + len(songs), song),
            dataset="ACE-Opencpop",
            speaker=f"ace-{index % 2}",
        )
        entries.extend((original, synthetic))
    split = song_disjoint_split(entries, 0.2, 0.2, 2026)
    assignments: dict[str, set[str]] = {}
    for entry in split:
        assignments.setdefault(entry.song, set()).add(entry.split)
    assert all(len(values) == 1 for values in assignments.values())
    assert all(assignments[song] == {"test"} for song in songs[:5])


def test_opencpop_family_rejects_incomplete_official_test(tmp_path: Path) -> None:
    entry = replace(_entry(tmp_path, 0, "2001"), dataset="ACE-Opencpop", speaker="ace")
    with pytest.raises(ValueError, match="missing official test songs"):
        song_disjoint_split([entry], 0.05, 0.05, 2026)
