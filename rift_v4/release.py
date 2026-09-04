from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


def export_release(checkpoint_path: Path, output_dir: Path) -> dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    required = {"ema", "config", "speaker_to_id", "mel_stats", "step"}
    missing = sorted(required - set(checkpoint))
    if checkpoint.get("schema_version") != 4 or missing:
        raise ValueError(f"invalid V4 checkpoint; missing={missing}")
    ema = checkpoint["ema"]
    if not isinstance(ema, dict) or not ema:
        raise ValueError("checkpoint EMA state is empty")
    for name, value in ema.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"EMA value is not a tensor: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"EMA tensor is non-finite: {name}")

    config = checkpoint["config"]
    model = config["model"]
    step = int(checkpoint["step"])
    filename = f"rift-v4-foundation-{model['dim']}x{model['depth']}-step{step}.pt"
    source_sha256 = _sha256(checkpoint_path)
    payload = {
        "schema_version": 4,
        "checkpoint_kind": "release",
        "release_schema_version": 1,
        "validation_protocol": checkpoint.get("validation_protocol"),
        "step": step,
        "best_validation": checkpoint.get("best_validation"),
        "model": ema,
        "ema": ema,
        "config": config,
        "software": checkpoint.get("software"),
        "speaker_to_id": checkpoint["speaker_to_id"],
        "manifest_sha256": checkpoint.get("manifest_sha256"),
        "mel_stats": checkpoint["mel_stats"],
        "source_checkpoint_sha256": source_sha256,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / filename
    _atomic_torch_save(payload, weights_path)
    release_sha256 = _sha256(weights_path)
    summary = {
        "schema_version": 1,
        "release_file": filename,
        "release_sha256": release_sha256,
        "release_bytes": weights_path.stat().st_size,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": source_sha256,
        "step": step,
        "checkpoint_state": "ema",
        "model": model,
        "software": checkpoint.get("software"),
        "speakers": len(checkpoint["speaker_to_id"]),
        "validation_protocol": checkpoint.get("validation_protocol"),
        "best_validation": checkpoint.get("best_validation"),
        "manifest_sha256": checkpoint.get("manifest_sha256"),
    }
    _atomic_json(output_dir / "config.json", config)
    _atomic_json(output_dir / "speaker_map.json", checkpoint["speaker_to_id"])
    _atomic_json(output_dir / "mel_stats.json", checkpoint["mel_stats"])
    _atomic_json(output_dir / "training_summary.json", summary)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    _atomic_text(
        output_dir / "SHA256SUMS",
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
    )
    return summary


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an EMA-only V4 release")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_release(args.checkpoint, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
