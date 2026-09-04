from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .config import V4Config
from .features import MelStats
from .manifest import ManifestEntry, load_manifest
from .shadow_panel import (
    _atomic_json,
    _feature_hashes,
    _load_checkpoint,
    deterministic_start,
    file_sha256,
    load_locked_tensors,
    stable_key,
)
from .shadow_v3_compare import load_v3
from .spectral_detail_audit import (
    BANDS,
    TIMES,
    band_name,
    compare_local_velocity,
    local_velocity_report,
)
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V3/V4 local spectral velocity on matched training crops"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--shadow-lock", type=Path, required=True)
    parser.add_argument("--train-lock", type=Path, required=True)
    parser.add_argument("--shadow-audit", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint = _load_checkpoint(args.v4_checkpoint, config)
    shadow_lock = json.loads(args.shadow_lock.read_text(encoding="utf-8"))
    train_lock = load_or_create_train_lock(
        args.train_lock,
        entries,
        args.manifest,
        args.mel_stats,
        args.shadow_lock,
        shadow_lock,
        args.frames,
        args.seed,
    )
    tensors = load_locked_tensors(
        train_lock, entries, config, stats, checkpoint["speaker_to_id"]
    )
    noise = torch.randn(
        len(train_lock["samples"]),
        args.frames,
        config.mel.channels,
        generator=torch.Generator().manual_seed(
            int(shadow_lock["protocol"]["seed"])
        ),
    )
    device = torch.device(args.device)
    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    v3_result = local_velocity_report(
        v3,
        "v3",
        tensors,
        noise,
        train_lock["samples"],
        stats,
        args.frames,
        args.batch_size,
        device,
    )
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(checkpoint["speaker_to_id"])).to(device).eval()
    system.load_state_dict(checkpoint["ema"], strict=True)
    v4_result = local_velocity_report(
        system.model,
        "v4",
        tensors,
        noise,
        train_lock["samples"],
        stats,
        args.frames,
        args.batch_size,
        device,
    )
    train_comparison = compare_local_velocity(
        v3_result,
        v4_result,
        train_lock["samples"],
        args.bootstrap_samples,
    )
    shadow = json.loads(args.shadow_audit.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "protocol": {
            "train_lock": str(args.train_lock),
            "train_lock_sha256": file_sha256(args.train_lock),
            "shadow_lock": str(args.shadow_lock),
            "shadow_lock_sha256": file_sha256(args.shadow_lock),
            "frames": args.frames,
            "times": list(TIMES),
            "bands": [band_name(band) for band in BANDS],
            "speaker_mix": "exactly matched to Shadow-128 speaker counts",
            "selection": "different train songs first, then secondary segments",
            "noise": "same numerical Gaussian seed as Shadow in native spaces",
            "v4_state": "220k EMA correct speaker",
            "v3_state": "official 300k null speaker",
        },
        "coverage": train_lock["coverage"],
        "train": {
            "v3_null": v3_result,
            "v4_220_ema_correct": v4_result,
            "v4_minus_v3": train_comparison,
        },
        "train_minus_shadow_gap": train_minus_shadow(
            train_comparison,
            shadow["comparisons"]["v4_minus_v3_local_velocity"],
        ),
    }
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def load_or_create_train_lock(
    path: Path,
    entries: list[ManifestEntry],
    manifest_path: Path,
    mel_stats_path: Path,
    shadow_path: Path,
    shadow: dict[str, object],
    frames: int,
    seed: int,
) -> dict[str, object]:
    expected = {
        "manifest_sha256": file_sha256(manifest_path),
        "mel_stats_sha256": file_sha256(mel_stats_path),
        "shadow_lock_sha256": file_sha256(shadow_path),
    }
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock.get("schema_version") != 1 or lock.get("source") != expected:
            raise ValueError("training panel lock source changed")
        return lock

    required = Counter(row["speaker"] for row in shadow["samples"])
    grouped: dict[str, dict[str, list[tuple[int, ManifestEntry]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, entry in enumerate(entries):
        if (
            entry.quality_status == "accepted"
            and entry.split == "train"
            and entry.speaker_key in required
            and entry.frames >= frames
        ):
            grouped[entry.speaker_key][entry.song].append((index, entry))
    selected = []
    for speaker, count in sorted(required.items()):
        songs = grouped[speaker]
        primary = [
            min(values, key=lambda item: stable_key(seed, "primary", item[1].id))
            for values in songs.values()
        ]
        primary.sort(key=lambda item: stable_key(seed, "song", item[1].song))
        primary_ids = {entry.id for _, entry in primary}
        secondary = sorted(
            [
                item
                for values in songs.values()
                for item in values
                if item[1].id not in primary_ids
            ],
            key=lambda item: stable_key(seed, "secondary", item[1].id),
        )
        candidates = primary + secondary
        if len(candidates) < count:
            raise RuntimeError(
                f"{speaker}: need {count} training crops, found {len(candidates)}"
            )
        selected.extend(candidates[:count])
    selected.sort(key=lambda item: stable_key(seed, "order", item[1].id))
    samples = []
    for ordinal, (manifest_index, entry) in enumerate(selected):
        start = deterministic_start(seed, entry.id, ordinal, entry.frames - frames)
        samples.append(
            {
                "ordinal": ordinal,
                "manifest_index": manifest_index,
                "entry_id": entry.id,
                "audio_sha256": entry.audio_sha256,
                "dataset": entry.dataset,
                "speaker": entry.speaker_key,
                "song": entry.song,
                "song_key": f"{entry.dataset}:{entry.song}",
                "split": entry.split,
                "start_frame": start,
                "maximum_frames": frames,
                "feature_sha256": _feature_hashes(entry),
            }
        )
    lock = {
        "schema_version": 1,
        "source": expected,
        "protocol": {"frames": [frames], "seed": seed},
        "coverage": {
            "crops": len(samples),
            "unique_song_units": len({row["song_key"] for row in samples}),
            "physical_speakers": len({row["speaker"] for row in samples}),
            "dataset_counts": dict(
                sorted(Counter(row["dataset"] for row in samples).items())
            ),
            "speaker_counts": dict(
                sorted(Counter(row["speaker"] for row in samples).items())
            ),
        },
        "samples": samples,
    }
    _atomic_json(path, lock)
    return lock


def train_minus_shadow(train, shadow):
    result = {}
    for time_value in TIMES:
        key = str(time_value)
        result[key] = {}
        for band in BANDS:
            name = band_name(band)
            result[key][name] = {}
            for region in ("active", "voiced", "unvoiced"):
                result[key][name][region] = {
                    metric: train[key][name][region][metric]["mean"]
                    - shadow[key][name][region][metric]["mean"]
                    for metric in (
                        "mse_contribution",
                        "nmse",
                        "rms_ratio",
                        "cosine",
                        "correlation",
                    )
                }
    return result


if __name__ == "__main__":
    main()
