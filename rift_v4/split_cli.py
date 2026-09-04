from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .manifest import ManifestEntry, load_manifest, write_manifest

OPENCPOP_FAMILY = frozenset({"ACE-Opencpop", "Opencpop"})
OPENCPOP_TEST_SONGS = frozenset({"2044", "2086", "2092", "2093", "2100"})


def song_disjoint_split(
    entries: list[ManifestEntry], validation_ratio: float, test_ratio: float, seed: int
) -> list[ManifestEntry]:
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("invalid split ratios")
    groups: dict[tuple[str, str], list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        if entry.quality_status != "accepted":
            continue
        groups[split_group(entry)].append(entry)
    if not groups:
        raise ValueError("manifest has no accepted recordings to split")
    by_dataset: dict[str, list[tuple[str, list[ManifestEntry]]]] = defaultdict(list)
    for (dataset, song), recordings in groups.items():
        by_dataset[dataset].append((song, recordings))
    assignments: dict[tuple[str, str], str] = {}
    for dataset, songs in by_dataset.items():
        songs.sort(key=lambda item: _stable_score(seed, dataset, item[0]))
        count = len(songs)
        validation_count = round(count * validation_ratio)
        official_test = OPENCPOP_TEST_SONGS if dataset == "Opencpop-family" else set()
        observed_songs = {song for song, _ in songs}
        missing_official = official_test - observed_songs
        if missing_official:
            raise ValueError(
                "Opencpop-family is missing official test songs: "
                f"{sorted(missing_official)}"
            )
        test_count = len(official_test) if official_test else round(count * test_ratio)
        if count >= 3 and validation_ratio > 0:
            validation_count = max(1, validation_count)
        if count >= 3 and test_ratio > 0:
            test_count = max(1, test_count)
        while validation_count + test_count >= count:
            if test_count > validation_count:
                test_count -= 1
            else:
                validation_count -= 1
        validation_candidates = [song for song, _ in songs if song not in official_test]
        validation_songs = set(validation_candidates[:validation_count])
        for index, (song, _) in enumerate(songs):
            if official_test:
                split = (
                    "test"
                    if song in official_test
                    else "validation"
                    if song in validation_songs
                    else "train"
                )
            else:
                split = (
                    "test"
                    if index < test_count
                    else "validation"
                    if index < test_count + validation_count
                    else "train"
                )
            assignments[(dataset, song)] = split
    # A closed-set speaker model cannot validate or test a speaker that has no
    # training song. Move the least-preferred held-out song back to train.
    speaker_songs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in entries:
        if entry.quality_status != "accepted":
            continue
        speaker_songs[(entry.dataset, entry.speaker)].add(entry.song)
    for (dataset, _speaker), songs in sorted(speaker_songs.items()):
        keys = [
            ("Opencpop-family", song) if dataset in OPENCPOP_FAMILY else (dataset, song)
            for song in songs
        ]
        if any(assignments[key] == "train" for key in keys):
            continue
        eligible = [
            key
            for key in keys
            if not (key[0] == "Opencpop-family" and key[1] in OPENCPOP_TEST_SONGS)
        ]
        if not eligible:
            raise ValueError(
                f"{dataset}:{_speaker} has no non-test song available for training"
            )
        selected = max(eligible, key=lambda key: _stable_score(seed, key[0], key[1]))
        assignments[selected] = "train"
    return [
        replace(entry, split=assignments[split_group(entry)])
        if entry.quality_status == "accepted"
        else entry
        for entry in entries
    ]


def explicit_validation_split(
    entries: list[ManifestEntry], validation_songs: set[str]
) -> list[ManifestEntry]:
    if not validation_songs:
        raise ValueError("explicit validation split requires at least one song")
    accepted_songs = {
        entry.song for entry in entries if entry.quality_status == "accepted"
    }
    missing = validation_songs - accepted_songs
    if missing:
        raise ValueError(
            f"validation songs are absent from manifest: {sorted(missing)}"
        )
    result = [
        replace(
            entry,
            split="validation" if entry.song in validation_songs else "train",
        )
        if entry.quality_status == "accepted"
        else entry
        for entry in entries
    ]
    training_speakers = {
        entry.speaker_key
        for entry in result
        if entry.quality_status == "accepted" and entry.split == "train"
    }
    accepted_speakers = {
        entry.speaker_key for entry in result if entry.quality_status == "accepted"
    }
    missing_training = accepted_speakers - training_speakers
    if missing_training:
        raise ValueError(
            "explicit validation split leaves speakers without training songs: "
            f"{sorted(missing_training)}"
        )
    return result


def split_group(entry: ManifestEntry) -> tuple[str, str]:
    family = "Opencpop-family" if entry.dataset in OPENCPOP_FAMILY else entry.dataset
    return family, entry.song


def _stable_score(seed: int, dataset: str, song: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{dataset}\0{song}".encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic song-disjoint splits"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-song", action="append", default=[])
    args = parser.parse_args()
    source = load_manifest(args.manifest)
    entries = (
        explicit_validation_split(source, set(args.validation_song))
        if args.validation_song
        else song_disjoint_split(
            source, args.validation_ratio, args.test_ratio, args.seed
        )
    )
    write_manifest(entries, args.output)
    counts = {
        name: sum(entry.split == name for entry in entries)
        for name in ("train", "validation", "test")
    }
    print(f"wrote song-disjoint split: {counts}")


if __name__ == "__main__":
    main()
