from pathlib import Path

import pytest
import torch

from rift_v4.data import (
    FeatureDataset,
    HierarchicalBatchSampler,
    bounded_normalize,
    collate_features,
)
from rift_v4.manifest import ManifestEntry, load_manifest, write_manifest


def _entry(
    tmp_path: Path, index: int, dataset: str, speaker: str, song: str
) -> ManifestEntry:
    prefix = tmp_path / f"features-{index}"
    torch.save(torch.randn(128, 20 + index), f"{prefix}.mel.pt")
    torch.save(torch.randn(10 + index, 16), f"{prefix}.content.pt")
    torch.save(torch.rand(20 + index), f"{prefix}.f0.pt")
    torch.save(torch.rand(20 + index), f"{prefix}.rms.pt")
    return ManifestEntry(
        id=f"id-{index}",
        dataset=dataset,
        speaker=speaker,
        song=song,
        audio_path=str(tmp_path / f"{index}.wav"),
        feature_prefix=str(prefix),
        frames=20 + index,
        duration_seconds=1,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        quality_status="accepted",
    )


def test_manifest_round_trip_and_hierarchical_batch(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice", "one"),
        _entry(tmp_path, 1, "A", "bob", "two"),
        _entry(tmp_path, 2, "B", "carol", "three"),
    ]
    path = tmp_path / "manifest.jsonl"
    assert write_manifest(entries, path) == 3
    loaded = load_manifest(path)
    sampler = HierarchicalBatchSampler(
        loaded,
        2,
        24,
        3,
        [8, 12],
        [0.5, 0.5],
        dataset_probabilities={"A": 0.5, "B": 0.5},
        seed=4,
    )
    first = next(iter(sampler))
    assert len(first) == 2
    dataset = FeatureDataset(loaded, mel_channels=128, content_dim=16)
    batch = collate_features([dataset[request] for request in first])
    assert batch["mel"].shape[0] == 2
    assert batch["mel"].shape[-1] == 128
    assert batch["content"].shape[-1] == 16
    assert batch["mask"].dtype == torch.bool
    assert torch.all(batch["requested_length"] == first[0].frames)


def test_speaker_ids_are_namespaced_by_dataset(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "001", "one"),
        _entry(tmp_path, 1, "B", "001", "two"),
    ]
    dataset = FeatureDataset(entries, mel_channels=128, content_dim=16)
    assert dataset.speaker_to_id.keys() == {"A:001", "B:001"}


def test_frame_budget_reduces_long_context_batch(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, 0, "A", "alice", "one")]
    sampler = HierarchicalBatchSampler(
        entries,
        batch_size=64,
        batch_frame_budget=16_384,
        steps_per_epoch=1,
        frame_buckets=[512],
        bucket_probabilities=[1.0],
    )
    batch = next(iter(sampler))
    assert len(batch) == 32
    assert all(request.frames == 512 for request in batch)


def test_dataset_probabilities_are_explicit(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice", "one"),
        _entry(tmp_path, 1, "A", "alice", "two"),
        _entry(tmp_path, 2, "B", "bob", "three"),
    ]
    sampler = HierarchicalBatchSampler(
        entries,
        batch_size=1,
        batch_frame_budget=8,
        steps_per_epoch=1,
        frame_buckets=[8],
        bucket_probabilities=[1.0],
        dataset_probabilities={"A": 0.25, "B": 0.75},
    )
    assert sampler.datasets == ["A", "B"]
    assert sampler.dataset_probabilities == pytest.approx([0.25, 0.75])


def test_bounded_normalize_uses_capped_simplex_projection() -> None:
    probabilities = bounded_normalize([1e-9, 1, 100], 1 / 6, 2 / 3)

    assert probabilities == pytest.approx([1 / 6, 1 / 6, 2 / 3])
    assert sum(probabilities) == pytest.approx(1.0)


def test_duration_tempered_speakers_stay_inside_bounds(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "short", "one"),
        _entry(tmp_path, 80, "A", "long", "two"),
    ]
    sampler = HierarchicalBatchSampler(
        entries,
        batch_size=2,
        batch_frame_budget=16,
        steps_per_epoch=1,
        frame_buckets=[8],
        bucket_probabilities=[1.0],
        speaker_probability_floor_ratio=0.5,
        speaker_probability_ceiling_ratio=1.5,
    )

    probabilities = sampler.speaker_probabilities["A"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert 0.25 <= probabilities["short"] < probabilities["long"] <= 0.75


def test_duration_tempered_songs_stay_inside_bounds(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice", "short"),
        _entry(tmp_path, 80, "A", "alice", "long"),
    ]
    sampler = HierarchicalBatchSampler(
        entries,
        batch_size=2,
        batch_frame_budget=16,
        steps_per_epoch=1,
        frame_buckets=[8],
        bucket_probabilities=[1.0],
        song_probability_floor_ratio=0.5,
        song_probability_ceiling_ratio=1.5,
    )

    probabilities = sampler.song_probabilities["A"]["alice"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert 0.25 <= probabilities["short"] < probabilities["long"] <= 0.75


def test_sampling_audit_extends_through_song_exposure(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "A", "alice", "one"),
        _entry(tmp_path, 1, "A", "alice", "two"),
    ]
    sampler = HierarchicalBatchSampler(
        entries,
        batch_size=2,
        batch_frame_budget=16,
        steps_per_epoch=1,
        frame_buckets=[8],
        bucket_probabilities=[1.0],
        dataset_probabilities={"A": 1.0},
    )

    audit = sampler.sampling_audit(
        max_steps=100,
        sample_rate=44_100,
        hop_length=512,
        speaker_drop_probability=0.2,
    )

    assert audit["expected_total_crops"] == pytest.approx(200)
    assert len(audit["speakers"]) == 1
    assert len(audit["songs"]) == 2
    assert audit["song_repeat_equivalent_quantiles"].keys() == {
        "p50",
        "p90",
        "p95",
        "p99",
        "p100",
    }
    assert sum(row["expected_crops"] for row in audit["songs"]) == pytest.approx(200)


def test_singleton_real_speaker_cap_is_enforced(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, 0, "singleton", "solo", "one"),
        _entry(tmp_path, 1, "multi", "a", "two"),
        _entry(tmp_path, 2, "multi", "b", "three"),
    ]

    with pytest.raises(ValueError, match="singleton real speaker probability"):
        HierarchicalBatchSampler(
            entries,
            batch_size=1,
            batch_frame_budget=8,
            steps_per_epoch=1,
            frame_buckets=[8],
            bucket_probabilities=[1.0],
            dataset_probabilities={"multi": 0.1, "singleton": 0.9},
            max_singleton_real_speaker_median_ratio=3.0,
        )


def test_sampler_rejects_invalid_frame_budget(tmp_path: Path) -> None:
    entry = _entry(tmp_path, 0, "A", "alice", "one")
    with pytest.raises(ValueError, match="batch_frame_budget"):
        HierarchicalBatchSampler([entry], 1, 7, 1, [8], [1.0])
