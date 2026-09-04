from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch

from .config import V4Config
from .datasets import GTSINGER_GROUPS
from .manifest import ManifestEntry, load_manifest, write_manifest
from .split_cli import split_group


def merge_manifests(inputs: list[Path], output: Path) -> int:
    by_id: dict[str, ManifestEntry] = {}
    for path in inputs:
        for entry in load_manifest(path):
            previous = by_id.get(entry.id)
            if previous is not None and previous != entry:
                raise ValueError(f"conflicting duplicate manifest id: {entry.id}")
            by_id[entry.id] = entry
    return write_manifest(by_id.values(), output)


def reconcile_frames(
    path: Path,
    output: Path,
    mel_channels: int,
    require_content: bool = False,
) -> tuple[int, int]:
    """Use extracted tensor lengths as the canonical sampler frame counts."""

    entries = load_manifest(path)
    updated: list[ManifestEntry] = []
    changed = 0
    for entry in entries:
        if entry.quality_status != "accepted":
            updated.append(entry)
            continue
        prefix = Path(entry.feature_prefix)
        mel = torch.as_tensor(
            torch.load(f"{prefix}.mel.pt", map_location="cpu", weights_only=True)
        ).squeeze()
        if mel.ndim != 2:
            raise ValueError(f"{entry.id}: invalid mel shape {tuple(mel.shape)}")
        if mel.shape[-1] == mel_channels:
            actual_frames = mel.shape[0]
        elif mel.shape[0] == mel_channels:
            actual_frames = mel.shape[1]
        else:
            raise ValueError(
                f"{entry.id}: mel has no {mel_channels}-wide axis: {tuple(mel.shape)}"
            )
        if actual_frames <= 0:
            raise ValueError(f"{entry.id}: mel has no frames")
        for suffix in (".f0.pt", ".rms.pt"):
            if not Path(f"{prefix}{suffix}").is_file():
                raise FileNotFoundError(f"{entry.id}: missing {suffix[1:]} feature")
        if require_content and (
            not entry.content_feature_path
            or not Path(entry.content_feature_path).is_file()
        ):
            raise FileNotFoundError(f"{entry.id}: missing content feature")
        if entry.frames != actual_frames:
            changed += 1
            entry = replace(entry, frames=actual_frames)
        updated.append(entry)
    write_manifest(updated, output)
    return len(updated), changed


def load_dataset_catalog(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported dataset catalog schema")
    corpora = payload.get("corpora")
    if not isinstance(corpora, dict) or not corpora:
        raise ValueError("dataset catalog has no corpora")
    return corpora


def catalog_manifests(catalog: Path, manifest_dir: Path) -> list[Path]:
    paths = [
        manifest_dir / str(specification["manifest"])
        for specification in load_dataset_catalog(catalog).values()
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"catalog manifests are missing: {missing}")
    return paths


def audit_manifest(
    path: Path,
    config: V4Config | None = None,
    require_features: bool = False,
    require_content: bool = False,
    catalog: Path | None = None,
    verify_audio: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("audit workers must be positive")
    entries = load_manifest(path)
    accepted = [entry for entry in entries if entry.quality_status == "accepted"]
    if not accepted:
        raise ValueError("manifest contains no accepted recordings")

    split_by_song: dict[tuple[str, str], set[str]] = defaultdict(set)
    audio_ids: dict[str, str] = {}
    duplicate_audio: list[tuple[str, str]] = []
    missing: list[str] = []
    for entry in accepted:
        split_by_song[split_group(entry)].add(entry.split)
        previous = audio_ids.setdefault(entry.audio_sha256, entry.id)
        if previous != entry.id:
            duplicate_audio.append((previous, entry.id))
        paths: list[Path] = []
        if require_features:
            paths.extend(
                Path(f"{entry.feature_prefix}{suffix}")
                for suffix in (".mel.pt", ".f0.pt", ".rms.pt")
            )
        if require_content:
            if not entry.content_feature_path:
                missing.append(f"{entry.id}:content-provenance")
            else:
                paths.append(Path(entry.content_feature_path))
        missing.extend(
            f"{entry.id}:{item.name}" for item in paths if not item.is_file()
        )

    split_leaks = [key for key, splits in split_by_song.items() if len(splits) != 1]
    if split_leaks:
        raise ValueError(f"songs cross splits: {split_leaks[:5]}")
    if duplicate_audio:
        raise ValueError(f"accepted duplicate audio: {duplicate_audio[:5]}")
    if missing:
        raise FileNotFoundError(f"missing derived features: {missing[:5]}")

    feature_summary: dict[str, object] | None = None
    if require_features or require_content:
        feature_summary = audit_feature_cache(
            accepted,
            mel_channels=config.mel.channels if config else 128,
            content_dim=config.model.content_dim if config else 768,
            require_content=require_content,
            workers=workers,
        )
    if verify_audio:
        _audit_audio_files(accepted, workers=workers)

    encoder_hashes = {
        entry.content_encoder_sha256
        for entry in accepted
        if entry.content_encoder_sha256 is not None
    }
    if require_content and len(encoder_hashes) != 1:
        raise ValueError(f"mixed ContentVec provenance: {encoder_hashes}")

    datasets = Counter(entry.dataset for entry in accepted)
    if config is not None:
        configured = set(config.sampling.dataset_probabilities)
        observed = set(datasets)
        if configured != observed:
            raise ValueError(
                f"dataset weight mismatch: configured={sorted(configured)}, "
                f"observed={sorted(observed)}"
            )

    catalog_entries = load_dataset_catalog(catalog) if catalog else None
    if catalog_entries is not None:
        unknown = {entry.dataset for entry in entries} - set(catalog_entries)
        if unknown:
            raise ValueError(
                f"manifest contains uncatalogued datasets: {sorted(unknown)}"
            )
        for dataset, count in Counter(entry.dataset for entry in entries).items():
            specification = catalog_entries[dataset]
            expected = int(specification["expected_entries"])
            if count != expected:
                raise ValueError(
                    f"{dataset}: expected {expected} entries, found {count}"
                )
            raw_hours = (
                sum(
                    entry.duration_seconds
                    for entry in entries
                    if entry.dataset == dataset
                )
                / 3600
            )
            lower, upper = specification["raw_hours"]
            if not float(lower) <= raw_hours <= float(upper):
                raise ValueError(
                    f"{dataset}: unexpected raw duration {raw_hours:.3f} h"
                )

    if "GTSinger" in {entry.dataset for entry in entries}:
        gtsinger = [entry for entry in entries if entry.dataset == "GTSinger"]
        groups = Counter(_gtsinger_group(entry) for entry in gtsinger)
        if set(groups) != GTSINGER_GROUPS:
            raise ValueError(f"incomplete GTSinger groups: {dict(groups)}")

    return {
        "entries": len(entries),
        "accepted": len(accepted),
        "accepted_hours": round(
            sum(entry.duration_seconds for entry in accepted) / 3600, 3
        ),
        "speakers": len({entry.speaker_key for entry in accepted}),
        "datasets": dict(sorted(datasets.items())),
        "splits": dict(sorted(Counter(entry.split for entry in accepted).items())),
        "content_encoder_sha256": sorted(encoder_hashes),
        "feature_cache": feature_summary,
        "audio_verified": verify_audio,
    }


def audit_feature_cache(
    entries: list[ManifestEntry],
    mel_channels: int,
    content_dim: int,
    require_content: bool = True,
    workers: int = 1,
) -> dict[str, object]:
    """Validate derived tensors before they can enter a training run."""

    issues: list[str] = []
    checked = 0
    mel_frames = 0

    def inspect(entry: ManifestEntry) -> tuple[str | None, int]:
        prefix = Path(entry.feature_prefix)
        paths = {
            "mel": Path(f"{prefix}.mel.pt"),
            "f0": Path(f"{prefix}.f0.pt"),
            "rms": Path(f"{prefix}.rms.pt"),
        }
        if require_content:
            if not entry.content_feature_path:
                return f"{entry.id}: missing content provenance", 0
            paths["content"] = Path(entry.content_feature_path)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            return f"{entry.id}: missing {','.join(missing)}", 0
        try:
            mel = _orient_matrix(
                _load_feature_tensor(paths["mel"]), mel_channels, "mel"
            )
            f0 = _orient_vector(_load_feature_tensor(paths["f0"]), "f0")
            rms = _orient_vector(_load_feature_tensor(paths["rms"]), "rms")
            if not _is_finite(mel):
                raise ValueError("mel contains NaN or Inf")
            if not _is_finite(f0) or (f0 < 0).any() or (f0 > 2000).any():
                raise ValueError("f0 is non-finite or outside [0, 2000] Hz")
            if not _is_finite(rms) or (rms < 0).any():
                raise ValueError("rms is non-finite or negative")
            if mel.shape[0] != f0.shape[0] or mel.shape[0] != rms.shape[0]:
                raise ValueError(
                    "mel, f0, and rms frame counts differ: "
                    f"{mel.shape[0]}/{f0.shape[0]}/{rms.shape[0]}"
                )
            if entry.frames != mel.shape[0]:
                raise ValueError(
                    f"manifest frames {entry.frames} differ from mel {mel.shape[0]}"
                )
            if require_content:
                content = _orient_matrix(
                    _load_feature_tensor(paths["content"]), content_dim, "content"
                )
                if not _is_finite(content):
                    raise ValueError("content contains NaN or Inf")
                ratio = content.shape[0] / max(1, mel.shape[0])
                if not math.isfinite(ratio) or not 0.5 <= ratio <= 2.5:
                    raise ValueError(
                        f"content/mel frame ratio is implausible: {ratio:.3f}"
                    )
        except Exception as error:
            return f"{entry.id}: {error}", 0
        return None, int(mel.shape[0])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for issue, frames in executor.map(inspect, entries):
            if issue is not None:
                issues.append(issue)
            else:
                checked += 1
                mel_frames += frames
    if issues:
        preview = "; ".join(issues[:8])
        suffix = " ..." if len(issues) > 8 else ""
        raise ValueError(
            f"feature cache audit failed for {len(issues)} entries: {preview}{suffix}"
        )
    return {"checked": checked, "mel_frames": mel_frames}


def _load_feature_tensor(path: Path) -> torch.Tensor:
    return torch.as_tensor(
        torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    )


def _orient_matrix(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
    value = value.squeeze()
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got {tuple(value.shape)}")
    if value.shape[1] == width:
        return value
    if value.shape[0] == width:
        return value.transpose(0, 1)
    raise ValueError(f"{name} has no {width}-wide axis: {tuple(value.shape)}")


def _orient_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim != 1:
        value = value.squeeze()
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got {tuple(value.shape)}")
    return value


def _is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _audit_audio_files(entries: list[ManifestEntry], workers: int = 1) -> None:
    issues: list[str] = []

    def inspect(entry: ManifestEntry) -> str | None:
        path = Path(entry.audio_path)
        try:
            if not path.is_file():
                raise FileNotFoundError("audio file is missing")
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError("audio stream is empty or has an invalid rate")
            if info.samplerate != entry.sample_rate or info.channels != entry.channels:
                raise ValueError(
                    "audio metadata differs from manifest: "
                    f"{info.samplerate}Hz/{info.channels}ch"
                )
            duration = info.frames / info.samplerate
            if abs(duration - entry.duration_seconds) > 1.0 / info.samplerate:
                raise ValueError("audio duration differs from manifest")
            if _sha256_file(path) != entry.audio_sha256:
                raise ValueError("audio SHA-256 differs from manifest")
        except Exception as error:
            return f"{entry.id}: {error}"
        return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        issues.extend(
            issue for issue in executor.map(inspect, entries) if issue is not None
        )
    if issues:
        preview = "; ".join(issues[:8])
        suffix = " ..." if len(issues) > 8 else ""
        raise ValueError(
            f"audio audit failed for {len(issues)} entries: {preview}{suffix}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gtsinger_group(entry: ManifestEntry) -> str:
    # audio_path resolves staging symlinks back into the official source tree;
    # feature_prefix retains the collision-safe staged filename and group tag.
    fields = Path(entry.feature_prefix).name.split("__")
    groups = GTSINGER_GROUPS.intersection(fields)
    if len(groups) != 1:
        raise ValueError(f"cannot identify GTSinger group: {entry.feature_prefix}")
    return next(iter(groups))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and audit V4 manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge")
    merge_source = merge.add_mutually_exclusive_group(required=True)
    merge_source.add_argument("--input", type=Path, action="append")
    merge_source.add_argument("--catalog", type=Path)
    merge.add_argument("--manifest-dir", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    reconcile = subparsers.add_parser("reconcile-frames")
    reconcile.add_argument("--manifest", type=Path, required=True)
    reconcile.add_argument("--output", type=Path, required=True)
    reconcile.add_argument("--mel-channels", type=int, default=128)
    reconcile.add_argument("--require-content", action="store_true")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--config", type=Path)
    audit.add_argument("--require-features", action="store_true")
    audit.add_argument("--require-content", action="store_true")
    audit.add_argument(
        "--verify-audio",
        action="store_true",
        help="re-open and hash every accepted source recording",
    )
    audit.add_argument("--catalog", type=Path)
    audit.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "merge":
        if args.catalog:
            if args.manifest_dir is None:
                parser.error("merge --catalog requires --manifest-dir")
            inputs = catalog_manifests(args.catalog, args.manifest_dir)
        else:
            inputs = args.input
        count = merge_manifests(inputs, args.output)
        print(f"merged {count} recordings from {len(inputs)} manifests")
        return
    if args.command == "reconcile-frames":
        count, changed = reconcile_frames(
            args.manifest, args.output, args.mel_channels, args.require_content
        )
        print(f"reconciled {changed} frame counts across {count} recordings")
        return
    config = V4Config.load(args.config) if args.config else None
    summary = audit_manifest(
        args.manifest,
        config,
        args.require_features,
        args.require_content,
        args.catalog,
        args.verify_audio,
        args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
