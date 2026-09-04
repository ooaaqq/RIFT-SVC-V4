import hashlib
from pathlib import Path

import torch

from rift_v4.release import export_release


def test_export_release_keeps_one_ema_storage_and_writes_sidecars(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step.pt"
    ema = {"model.weight": torch.arange(16_384, dtype=torch.float32)}
    torch.save(
        {
            "schema_version": 4,
            "checkpoint_kind": "audit",
            "validation_protocol": 6,
            "step": 130_000,
            "best_validation": 0.1,
            "model": {"model.weight": torch.zeros(16_384)},
            "ema": ema,
            "config": {"model": {"dim": 1024, "depth": 16}},
            "software": {
                "torch": "2.12.1+cu130",
                "cuda": "13.0",
                "torchao": "0.18.0",
            },
            "speaker_to_id": {"A:singer": 0},
            "manifest_sha256": "a" * 64,
            "mel_stats": {"mean": [0.0], "std": [1.0], "frames": 10},
        },
        checkpoint,
    )

    summary = export_release(checkpoint, tmp_path / "release")

    release_path = tmp_path / "release" / str(summary["release_file"])
    release = torch.load(release_path, map_location="cpu", weights_only=False)
    assert release["checkpoint_kind"] == "release"
    assert release["step"] == 130_000
    assert release["model"] is release["ema"]
    assert torch.equal(release["ema"]["model.weight"], ema["model.weight"])
    assert release["software"]["torchao"] == "0.18.0"
    assert release_path.stat().st_size < checkpoint.stat().st_size
    for name in (
        "config.json",
        "speaker_map.json",
        "mel_stats.json",
        "training_summary.json",
        "SHA256SUMS",
    ):
        assert (tmp_path / "release" / name).is_file()
    assert (
        summary["release_sha256"]
        == hashlib.sha256(release_path.read_bytes()).hexdigest()
    )
