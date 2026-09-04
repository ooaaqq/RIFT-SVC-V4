from pathlib import Path

import pytest
import torch

from rift_v4.evaluate import load_or_create_panel, pitch_metrics
from rift_v4.manifest import ManifestEntry


def _entry(tmp_path: Path, dataset: str, index: int) -> ManifestEntry:
    prefix = tmp_path / f"{dataset}-{index}"
    torch.save(torch.tensor([0.0, 100.0, 110.0, 0.0]), f"{prefix}.f0.pt")
    torch.save(torch.ones(4), f"{prefix}.rms.pt")
    torch.save(torch.ones(4, 3), f"{prefix}.content.pt")
    return ManifestEntry(
        id=f"{dataset}-{index}",
        dataset=dataset,
        speaker=f"speaker-{index}",
        song=f"song-{index}",
        audio_path=str(prefix.with_suffix(".wav")),
        feature_prefix=str(prefix),
        frames=4,
        duration_seconds=1.0,
        sample_rate=44_100,
        channels=1,
        audio_sha256=f"{index + 1:064x}",
        split="validation",
        quality_status="accepted",
    )


def test_panel_selection_is_persisted_and_dataset_balanced(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, dataset, index)
        for index, dataset in enumerate(("A", "A", "B", "B"))
    ]
    speakers = {entry.speaker_key: index for index, entry in enumerate(entries)}
    path = tmp_path / "panel.json"

    first = load_or_create_panel(path, entries, speakers, 4, 4, 2026)
    second = load_or_create_panel(path, list(reversed(entries)), speakers, 2, 2, 9)

    assert second == first
    assert [item["dataset"] for item in first["samples"]] == ["A", "B", "A", "B"]

    torch.save(torch.zeros(4), f"{entries[0].feature_prefix}.f0.pt")
    with pytest.raises(ValueError, match="feature changed"):
        load_or_create_panel(path, entries, speakers, 4, 4, 2026)


def test_pitch_metrics_separate_pitch_and_voicing_errors() -> None:
    target = torch.tensor([0.0, 100.0, 200.0, 0.0])
    exact = pitch_metrics(target, target)
    missed_voicing = pitch_metrics(target, torch.tensor([0.0, 100.0, 0.0, 0.0]))

    assert exact["voicing_f1"] == pytest.approx(1.0)
    assert exact["f0_cents_mae"] == pytest.approx(0.0)
    assert missed_voicing["voicing_recall"] == pytest.approx(0.5)
    assert missed_voicing["f0_cents_mae"] == pytest.approx(0.0)
