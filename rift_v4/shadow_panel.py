from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import V4Config
from .data import FeatureDataset
from .features import MelStats
from .manifest import ManifestEntry, load_manifest
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a locked song-disjoint endpoint shadow panel"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--min-songs", type=int, default=100)
    parser.add_argument("--frames", type=int, nargs="+", default=(512, 768))
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if not 128 <= args.panel_size <= 256:
        parser.error("--panel-size must be between 128 and 256")
    frames = tuple(sorted(set(args.frames)))
    if not frames or frames[0] <= 0:
        parser.error("--frames must contain positive lengths")
    if args.min_songs <= 0 or args.intervals <= 0 or args.batch_size <= 0:
        parser.error("panel dimensions must be positive")

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    first = _load_checkpoint(args.checkpoint[0], config)
    speaker_to_id = first["speaker_to_id"]
    del first

    lock = load_or_create_panel_lock(
        args.panel_lock,
        entries,
        args.manifest,
        args.mel_stats,
        speaker_to_id,
        set(config.sampling.synthetic_datasets),
        args.panel_size,
        args.min_songs,
        frames,
        args.seed,
    )
    if args.prepare_only:
        print(json.dumps(lock["coverage"], ensure_ascii=False, indent=2))
        print(f"wrote {args.panel_lock}")
        return
    tensors = load_locked_tensors(
        lock,
        entries,
        config,
        stats,
        speaker_to_id,
    )
    device = torch.device(args.device)
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    results: dict[str, object] = {}
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "panel_lock": str(args.panel_lock),
            "panel_lock_sha256": file_sha256(args.panel_lock),
            "checkpoint_state": "ema",
            "speaker_conditioning": "correct physical speaker",
            "frames": list(frames),
            "noise": "one locked Gaussian tensor; shorter lengths use its prefix",
            "noise_seed": args.seed,
            "solver": "linear Euler",
            "intervals": args.intervals,
            "silence_threshold_rms": 1e-3,
            "bootstrap_unit": "dataset plus song",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "coverage": lock["coverage"],
        "checkpoints": results,
        "comparisons": {},
    }
    for path in args.checkpoint:
        checkpoint = _load_checkpoint(path, config)
        if speaker_map_sha256(checkpoint["speaker_to_id"]) != lock["source"][
            "speaker_to_id_sha256"
        ]:
            raise ValueError(f"speaker mapping changed in {path}")
        step = int(checkpoint["step"])
        print(json.dumps({"loading_ema_step": step}), flush=True)
        system.load_state_dict(checkpoint["ema"], strict=True)
        del checkpoint
        per_frame = evaluate_checkpoint(
            system.model,
            tensors,
            lock["samples"],
            stats,
            frames,
            args.intervals,
            args.batch_size,
            args.seed,
            device,
        )
        results[str(step)] = {
            "step": step,
            "checkpoint": str(path),
            "frames": per_frame,
        }
        _atomic_json(args.output, payload)

    ordered_steps = sorted(int(step) for step in results)
    comparisons = {}
    for left_index, left in enumerate(ordered_steps):
        for right in ordered_steps[left_index + 1 :]:
            name = f"{left}_to_{right}"
            comparisons[name] = compare_checkpoints(
                results[str(left)]["frames"],
                results[str(right)]["frames"],
                lock["samples"],
                args.bootstrap_samples,
                args.seed + left + right,
            )
    payload["comparisons"] = comparisons
    payload["status"] = "complete"
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def load_or_create_panel_lock(
    path: Path,
    entries: list[ManifestEntry],
    manifest_path: Path,
    mel_stats_path: Path,
    speaker_to_id: dict[str, int],
    synthetic_datasets: set[str],
    panel_size: int,
    min_songs: int,
    frames: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    manifest_digest = file_sha256(manifest_path)
    stats_digest = file_sha256(mel_stats_path)
    mapping_digest = speaker_map_sha256(speaker_to_id)
    by_id = {entry.id: (index, entry) for index, entry in enumerate(entries)}
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock.get("schema_version") != 1:
            raise ValueError("unsupported shadow-panel lock schema")
        expected = {
            "manifest_sha256": manifest_digest,
            "mel_stats_sha256": stats_digest,
            "speaker_to_id_sha256": mapping_digest,
        }
        for name, value in expected.items():
            if lock["source"].get(name) != value:
                raise ValueError(f"shadow-panel source changed: {name}")
        protocol = lock["protocol"]
        requested = {
            "panel_size": panel_size,
            "minimum_unique_songs": min_songs,
            "frames": list(frames),
            "seed": seed,
        }
        for name, value in requested.items():
            if protocol.get(name) != value:
                raise ValueError(f"shadow-panel protocol changed: {name}")
        for row in lock["samples"]:
            current = by_id.get(row["entry_id"])
            if current is None or current[0] != row["manifest_index"]:
                raise ValueError(f"shadow-panel entry moved: {row['entry_id']}")
            if current[1].audio_sha256 != row["audio_sha256"]:
                raise ValueError(f"shadow-panel audio changed: {row['entry_id']}")
            _verify_feature_hashes(current[1], row["feature_sha256"])
        return lock

    selected = select_shadow_entries(
        entries,
        speaker_to_id,
        synthetic_datasets,
        panel_size,
        max(frames),
        seed,
    )
    samples = []
    for ordinal, (manifest_index, entry) in enumerate(selected):
        maximum_start = entry.frames - max(frames)
        start = deterministic_start(seed, entry.id, ordinal, maximum_start)
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
                "maximum_frames": max(frames),
                "feature_sha256": _feature_hashes(entry),
            }
        )
    unique_songs = len({row["song_key"] for row in samples})
    if unique_songs < min_songs:
        raise RuntimeError(
            f"shadow panel has only {unique_songs} unique songs; need {min_songs}"
        )
    coverage = {
        "crops": len(samples),
        "unique_song_units": unique_songs,
        "physical_speakers": len({row["speaker"] for row in samples}),
        "dataset_counts": dict(
            sorted(Counter(row["dataset"] for row in samples).items())
        ),
        "speaker_crop_min": min(Counter(row["speaker"] for row in samples).values()),
        "speaker_crop_max": max(Counter(row["speaker"] for row in samples).values()),
    }
    lock = {
        "schema_version": 1,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "mel_stats": str(mel_stats_path),
            "mel_stats_sha256": stats_digest,
            "speaker_to_id_sha256": mapping_digest,
        },
        "protocol": {
            "selection": (
                "automatic deterministic speaker round-robin; first one entry per "
                "speaker-song, then deterministic secondary entries"
            ),
            "panel_size": panel_size,
            "minimum_unique_songs": min_songs,
            "frames": list(frames),
            "nested_crop": "each shorter crop is a prefix of the locked maximum crop",
            "seed": seed,
            "accepted_only": True,
            "splits": ["validation", "test"],
            "synthetic_datasets_excluded": sorted(synthetic_datasets),
            "human_selection": False,
        },
        "coverage": coverage,
        "samples": samples,
    }
    _atomic_json(path, lock)
    return lock


def select_shadow_entries(
    entries: list[ManifestEntry],
    speaker_to_id: dict[str, int],
    synthetic_datasets: set[str],
    panel_size: int,
    maximum_frames: int,
    seed: int,
) -> list[tuple[int, ManifestEntry]]:
    grouped: dict[str, dict[str, list[tuple[int, ManifestEntry]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_by_speaker: dict[str, list[tuple[int, ManifestEntry]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if (
            entry.quality_status == "accepted"
            and entry.split in {"validation", "test"}
            and entry.dataset not in synthetic_datasets
            and entry.speaker_key in speaker_to_id
            and entry.frames >= maximum_frames
        ):
            grouped[entry.speaker_key][entry.song].append((index, entry))
            all_by_speaker[entry.speaker_key].append((index, entry))
    primary: dict[str, list[tuple[int, ManifestEntry]]] = {}
    for speaker, songs in grouped.items():
        values = [
            min(records, key=lambda item: stable_key(seed, "entry", item[1].id))
            for records in songs.values()
        ]
        primary[speaker] = sorted(
            values,
            key=lambda item: stable_key(seed, "song", speaker, item[1].song),
        )
    selected = _round_robin(primary, panel_size)
    selected_ids = {entry.id for _, entry in selected}
    if len(selected) < panel_size:
        secondary = {
            speaker: sorted(
                [item for item in values if item[1].id not in selected_ids],
                key=lambda item: stable_key(seed, "secondary", item[1].id),
            )
            for speaker, values in all_by_speaker.items()
        }
        selected.extend(_round_robin(secondary, panel_size - len(selected)))
    if len(selected) < panel_size:
        raise RuntimeError(
            f"only {len(selected)} eligible locked crops for requested {panel_size}"
        )
    return selected


def _round_robin(
    by_speaker: dict[str, list[tuple[int, ManifestEntry]]], limit: int
) -> list[tuple[int, ManifestEntry]]:
    speakers = sorted(by_speaker, key=lambda value: stable_key(0, "speaker", value))
    result = []
    round_index = 0
    while len(result) < limit:
        added = 0
        for speaker in speakers:
            values = by_speaker[speaker]
            if round_index < len(values):
                result.append(values[round_index])
                added += 1
                if len(result) == limit:
                    break
        if not added:
            break
        round_index += 1
    return result


def deterministic_start(
    seed: int, entry_id: str, ordinal: int, maximum_start: int
) -> int:
    if maximum_start < 0:
        raise ValueError(f"{entry_id}: crop is longer than recording")
    value = int.from_bytes(stable_key(seed, "crop", entry_id, str(ordinal))[:8], "big")
    return value % (maximum_start + 1)


def load_locked_tensors(
    lock: dict[str, object],
    entries: list[ManifestEntry],
    config: V4Config,
    stats: MelStats,
    speaker_to_id: dict[str, int],
) -> dict[str, torch.Tensor]:
    dataset = FeatureDataset(
        entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        speaker_to_id,
        voiced_crop_probability=0.0,
    )
    loaded: dict[str, list[torch.Tensor]] = defaultdict(list)
    maximum_frames = max(lock["protocol"]["frames"])
    for row in lock["samples"]:
        entry = entries[int(row["manifest_index"])]
        if entry.id != row["entry_id"]:
            raise ValueError(f"locked manifest index changed: {row['entry_id']}")
        features = dataset._load_features(entry)
        start = int(row["start_frame"])
        stop = start + maximum_frames
        if min(value.shape[0] for value in features.values()) < stop:
            raise ValueError(f"locked crop exceeds features: {entry.id}")
        for name, value in features.items():
            loaded[name].append(value[start:stop])
        loaded["speaker"].append(
            torch.tensor(speaker_to_id[entry.speaker_key], dtype=torch.long)
        )
    return {name: torch.stack(values) for name, values in loaded.items()}


def evaluate_checkpoint(
    model,
    tensors: dict[str, torch.Tensor],
    metadata: list[dict[str, object]],
    stats: MelStats,
    frames_values: tuple[int, ...],
    intervals: int,
    batch_size: int,
    noise_seed: int,
    device: torch.device,
) -> dict[str, object]:
    maximum_frames = max(frames_values)
    generator = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(
        len(metadata), maximum_frames, tensors["mel"].shape[-1], generator=generator
    )
    output = {}
    for frames in frames_values:
        rows = []
        for begin in range(0, len(metadata), batch_size):
            end = min(begin + batch_size, len(metadata))
            mel = tensors["mel"][begin:end, :frames].to(device)
            content = tensors["content"][begin:end, :frames].to(device)
            f0 = tensors["f0"][begin:end, :frames].to(device)
            rms = tensors["rms"][begin:end, :frames].to(device)
            speaker = tensors["speaker"][begin:end].to(device)
            generated = noise[begin:end, :frames].to(device)
            mask = torch.ones(end - begin, frames, dtype=torch.bool, device=device)
            times = torch.linspace(0.0, 1.0, intervals + 1, device=device)
            with torch.inference_mode(), torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                for index in range(intervals):
                    timestep = times[index].expand(end - begin)
                    generated += (times[index + 1] - times[index]) * model(
                        generated, content, f0, rms, speaker, timestep, mask
                    ).float()
                raw_prediction = stats.denormalize(generated.float())
                raw_target = stats.denormalize(mel.float())
                residual = raw_prediction - raw_target
                frame_mse = residual.square().mean(-1)
                active = rms.squeeze(-1) > 1e-3
                full_mse = frame_mse.mean(-1)
                active_mse = _per_sample_masked_mean(frame_mse, active)
                silence_mse = _per_sample_masked_mean(frame_mse, ~active)
                l1 = residual.abs().mean((1, 2))
                cosine = F.cosine_similarity(
                    raw_prediction.flatten(1), raw_target.flatten(1), dim=1
                )
            for offset, sample_index in enumerate(range(begin, end)):
                rows.append(
                    {
                        "ordinal": int(metadata[sample_index]["ordinal"]),
                        "entry_id": metadata[sample_index]["entry_id"],
                        "full_raw_mse": float(full_mse[offset]),
                        "active_raw_mse": _optional_float(active_mse[offset]),
                        "silence_raw_mse": _optional_float(silence_mse[offset]),
                        "raw_l1": float(l1[offset]),
                        "raw_cosine": float(cosine[offset]),
                        "active_frames": int(active[offset].sum()),
                        "silence_frames": int((~active[offset]).sum()),
                    }
                )
            print(
                json.dumps(
                    {"frames": frames, "completed": end, "total": len(metadata)}
                ),
                flush=True,
            )
        output[str(frames)] = {
            "aggregate": {
                metric: summarize_samples(rows, metric)
                for metric in (
                    "full_raw_mse",
                    "active_raw_mse",
                    "silence_raw_mse",
                    "raw_l1",
                    "raw_cosine",
                )
            },
            "samples": rows,
        }
    return output


def compare_checkpoints(
    left: dict[str, object],
    right: dict[str, object],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    result = {}
    for frame_index, frames in enumerate(sorted(left, key=int)):
        left_rows = left[frames]["samples"]
        right_rows = right[frames]["samples"]
        result[frames] = {}
        for metric_index, metric in enumerate(
            (
                "full_raw_mse",
                "active_raw_mse",
                "silence_raw_mse",
                "raw_l1",
                "raw_cosine",
            )
        ):
            result[frames][metric] = paired_statistics(
                [row[metric] for row in left_rows],
                [row[metric] for row in right_rows],
                metadata,
                bootstrap_samples,
                seed + frame_index * 100 + metric_index,
                higher_is_better=metric == "raw_cosine",
            )
    return result


def paired_statistics(
    before: list[float | None],
    after: list[float | None],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
    *,
    higher_is_better: bool = False,
) -> dict[str, object]:
    paired = [
        (float(right) - float(left), row)
        for left, right, row in zip(before, after, metadata, strict=True)
        if left is not None and right is not None
    ]
    if not paired:
        return {"samples": 0}
    deltas = torch.tensor([value for value, _ in paired], dtype=torch.float64)
    wins = deltas > 0 if higher_is_better else deltas < 0
    by_song: dict[str, list[float]] = defaultdict(list)
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for value, row in paired:
        by_song[str(row["song_key"])].append(value)
        by_speaker[str(row["speaker"])].append(value)
    song_values = torch.tensor(
        [sum(values) / len(values) for values in by_song.values()],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        len(song_values),
        (bootstrap_samples, len(song_values)),
        generator=generator,
    )
    bootstrap = song_values[draws].mean(1)
    interval = torch.quantile(
        bootstrap, torch.tensor((0.025, 0.975), dtype=torch.float64)
    )
    speaker_means = [sum(values) / len(values) for values in by_speaker.values()]
    return {
        "definition": "after minus before",
        "samples": len(paired),
        "song_units": len(song_values),
        "win_rate_after": float(wins.double().mean()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(deltas.median()),
        "song_bootstrap_mean_delta_95_ci": [float(interval[0]), float(interval[1])],
        "speaker_macro_delta": sum(speaker_means) / len(speaker_means),
    }


def summarize_samples(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "samples": len(values),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
    }


def stable_key(seed: int, *parts: str) -> bytes:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).digest()


def speaker_map_sha256(speaker_to_id: dict[str, int]) -> str:
    payload = json.dumps(speaker_to_id, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_paths(entry: ManifestEntry) -> dict[str, Path]:
    return {
        "mel": Path(f"{entry.feature_prefix}.mel.pt"),
        "f0": Path(f"{entry.feature_prefix}.f0.pt"),
        "rms": Path(f"{entry.feature_prefix}.rms.pt"),
        "content": Path(
            entry.content_feature_path or f"{entry.feature_prefix}.content.pt"
        ),
    }


def _feature_hashes(entry: ManifestEntry) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in _feature_paths(entry).items()}


def _verify_feature_hashes(entry: ManifestEntry, expected: dict[str, str]) -> None:
    actual = _feature_hashes(entry)
    if actual != expected:
        raise ValueError(f"shadow-panel features changed: {entry.id}")


def _load_checkpoint(path: Path, config: V4Config) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("schema_version") != 4:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    if checkpoint.get("config") != asdict(config):
        raise ValueError(f"checkpoint configuration differs: {path}")
    return checkpoint


def _per_sample_masked_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> list[torch.Tensor | None]:
    result = []
    for row, selected in zip(values, mask, strict=True):
        result.append(row[selected].mean() if bool(selected.any()) else None)
    return result


def _optional_float(value: torch.Tensor | None) -> float | None:
    return float(value) if value is not None else None


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
