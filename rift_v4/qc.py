from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import V4Config
from .manifest import ManifestEntry, load_manifest, write_manifest


def evaluate(entry: ManifestEntry, expected_sample_rate: int) -> ManifestEntry:
    audio, sample_rate = sf.read(entry.audio_path, dtype="float32", always_2d=True)
    mono = audio[:, 0]
    peak = float(np.max(np.abs(mono), initial=0))
    clipping_ratio = float(np.mean(np.abs(mono) >= 0.999))
    block = max(1, sample_rate // 50)
    usable = len(mono) // block * block
    if usable:
        rms = np.sqrt(np.mean(mono[:usable].reshape(-1, block) ** 2, axis=1) + 1e-12)
        active_ratio = float(np.mean(rms > 10 ** (-50 / 20)))
    else:
        active_ratio = 0.0
    reasons: list[str] = []
    if sample_rate < expected_sample_rate:
        reasons.append(f"sample_rate_below_target:{sample_rate}")
    if audio.shape[1] != 1:
        reasons.append(f"channels:{audio.shape[1]}")
    if not 1.0 <= entry.duration_seconds <= 20 * 60:
        reasons.append("duration")
    if peak < 1e-4:
        reasons.append("near_silent")
    if clipping_ratio > 1e-4:
        reasons.append("clipping")
    if active_ratio < 0.2:
        reasons.append("mostly_silent")
    score = max(
        0.0,
        1.0
        - min(1.0, clipping_ratio * 1000) * 0.5
        - max(0.0, 0.5 - active_ratio) * 0.5,
    )
    return replace(
        entry,
        quality_status="rejected" if reasons else "accepted",
        quality_score=score,
        peak=peak,
        clipping_ratio=clipping_ratio,
        exclusion_reasons=tuple(reasons),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic waveform gates")
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = V4Config.load(args.config)
    entries = [
        evaluate(entry, config.sample_rate) for entry in load_manifest(args.manifest)
    ]
    entries = reject_duplicate_audio(entries)
    write_manifest(entries, args.output)
    accepted = sum(entry.quality_status == "accepted" for entry in entries)
    print(f"accepted {accepted}/{len(entries)} recordings; wrote {args.output}")


def reject_duplicate_audio(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    seen: dict[str, str] = {}
    result: list[ManifestEntry] = []
    for entry in sorted(entries, key=lambda item: item.id):
        original = seen.get(entry.audio_sha256)
        if original is None or entry.quality_status != "accepted":
            if entry.quality_status == "accepted":
                seen[entry.audio_sha256] = entry.id
            result.append(entry)
            continue
        result.append(
            replace(
                entry,
                quality_status="rejected",
                quality_score=0.0,
                exclusion_reasons=(f"duplicate_audio:{original}",),
            )
        )
    return result


if __name__ == "__main__":
    main()
