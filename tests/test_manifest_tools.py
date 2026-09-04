import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rift_v4.manifest import ManifestEntry, load_manifest, write_manifest
from rift_v4.manifest_tools import (
    _gtsinger_group,
    audit_feature_cache,
    audit_manifest,
    merge_manifests,
    reconcile_frames,
)


def _entry(identifier: str, *, song: str = "song", split: str = "train"):
    return ManifestEntry(
        id=identifier,
        dataset="test",
        speaker="singer",
        song=song,
        audio_path=f"/{identifier}.wav",
        feature_prefix=f"/{identifier}",
        frames=100,
        duration_seconds=1.0,
        sample_rate=44100,
        channels=1,
        audio_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
        split=split,
        quality_status="accepted",
    )


def test_merge_manifests_rejects_conflicting_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest([_entry("a")], first)
    write_manifest([replace(_entry("a"), song="different")], second)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge_manifests([first, second], tmp_path / "merged.jsonl")


def test_audit_rejects_song_split_leak(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [_entry("a", split="train"), _entry("b", split="validation")],
        manifest,
    )

    with pytest.raises(ValueError, match="songs cross splits"):
        audit_manifest(manifest)


def test_gtsinger_group_uses_staged_feature_name() -> None:
    entry = replace(
        _entry("gtsinger"),
        audio_path="/official/song/Control_Group/0001.wav",
        feature_prefix="/features/technique__Vibrato_Group__0001",
    )
    assert _gtsinger_group(entry) == "Vibrato_Group"


def test_reconcile_frames_uses_actual_mel_length(tmp_path: Path) -> None:
    prefix = tmp_path / "features" / "sample"
    prefix.parent.mkdir()
    torch.save(torch.zeros(37, 128), f"{prefix}.mel.pt")
    torch.save(torch.zeros(37), f"{prefix}.f0.pt")
    torch.save(torch.zeros(37), f"{prefix}.rms.pt")
    content = tmp_path / "sample.content.pt"
    torch.save(torch.zeros(74, 768), content)
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    write_manifest(
        [
            replace(
                _entry("sample"),
                feature_prefix=str(prefix),
                content_feature_path=str(content),
                content_encoder_id="contentvec-dualphase10ms-v1:test",
                content_encoder_sha256="a" * 64,
            )
        ],
        source,
    )

    count, changed = reconcile_frames(source, output, 128, require_content=True)

    assert (count, changed) == (1, 1)
    assert load_manifest(output)[0].frames == 37


def test_feature_cache_audit_checks_values_and_dual_phase_ratio(tmp_path: Path) -> None:
    prefix = tmp_path / "features" / "sample"
    prefix.parent.mkdir()
    torch.save(torch.zeros(37, 128), f"{prefix}.mel.pt")
    torch.save(torch.zeros(37), f"{prefix}.f0.pt")
    torch.save(torch.ones(37), f"{prefix}.rms.pt")
    content = tmp_path / "sample.content.pt"
    torch.save(torch.zeros(74, 768), content)
    entry = replace(
        _entry("sample"),
        feature_prefix=str(prefix),
        frames=37,
        content_feature_path=str(content),
    )

    summary = audit_feature_cache([entry], 128, 768)

    assert summary == {"checked": 1, "mel_frames": 37}
    torch.save(torch.full((37,), float("nan")), f"{prefix}.f0.pt")
    with pytest.raises(ValueError, match="non-finite"):
        audit_feature_cache([entry], 128, 768)
