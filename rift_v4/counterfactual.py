from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from transformers import AutoFeatureExtractor, WavLMForXVector

from .config import V4Config
from .content_encoder import FrozenContentEncoder
from .evaluate import (
    _file_sha256,
    _load_f0,
    _load_panel_features,
    _panel_crop_start,
    _panel_feature_paths,
    _reference_crop,
    pitch_metrics,
)
from .extract_content import encode_chunked
from .features import MelStats, extract_auxiliary_features
from .infer import sample_chunked
from .manifest import ManifestEntry, load_manifest
from .third_party import PCNSFLock, verify_contentvec_snapshot
from .train import build_system
from .vocoder import load_pc_nsf_generator, synthesize_pc_nsf_tensors

SYNTHETIC_DATASETS = {"ACE-Opencpop"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fixed A-to-B speaker conversion pairs"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path)
    parser.add_argument("--pc-nsf-lock", type=Path)
    parser.add_argument("--vocoder-checkpoint", type=Path)
    parser.add_argument("--contentvec-model", type=Path)
    parser.add_argument(
        "--contentvec-lock",
        type=Path,
        default=Path("third_party/contentvec.lock.json"),
    )
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--references-per-speaker", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state", choices=("raw", "ema"), action="append")
    parser.add_argument(
        "--exclude-panel-lock",
        type=Path,
        help="exclude every dataset/song unit used by another locked panel",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("checkpoint does not use schema 4")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint configuration differs from audit config")
    entries = load_manifest(args.manifest)
    excluded_song_keys = (
        load_excluded_song_keys(args.exclude_panel_lock)
        if args.exclude_panel_lock
        else set()
    )
    pairs = args.pairs or config.evaluation.counterfactual_pairs
    frames = args.frames or config.evaluation.endpoint_panel_frames[0]
    spec = load_or_create_pairs(
        args.pair_spec,
        entries,
        checkpoint["speaker_to_id"],
        pairs,
        frames,
        args.references_per_speaker,
        args.seed,
        excluded_song_keys=excluded_song_keys,
    )
    if args.prepare_only:
        print(json.dumps(spec["coverage"], ensure_ascii=False, indent=2))
        print(f"wrote {args.pair_spec}")
        return
    runtime_assets = {
        "pc-nsf checkout": args.pc_nsf_checkout,
        "pc-nsf lock": args.pc_nsf_lock,
        "vocoder checkpoint": args.vocoder_checkpoint,
        "ContentVec model": args.contentvec_model,
    }
    missing = [name for name, path in runtime_assets.items() if path is None]
    if missing:
        parser.error(f"counterfactual audit needs: {', '.join(missing)}")
    payload = audit_counterfactual(
        config=config,
        entries=entries,
        checkpoint=checkpoint,
        spec=spec,
        pc_nsf_checkout=args.pc_nsf_checkout,
        pc_nsf_lock=args.pc_nsf_lock,
        vocoder_checkpoint=args.vocoder_checkpoint,
        contentvec_model=args.contentvec_model,
        contentvec_lock=args.contentvec_lock,
        device=torch.device(args.device),
        steps=args.steps or config.evaluation.inference_steps,
        guidance=(
            args.guidance
            if args.guidance is not None
            else config.evaluation.guidance
        ),
        states=tuple(args.state or ("ema",)),
    )
    _atomic_json(args.output, payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def load_or_create_pairs(
    path: Path,
    entries: list[ManifestEntry],
    speaker_to_id: dict[str, int],
    pairs: int,
    frames: int,
        references_per_speaker: int,
        seed: int,
    *,
    excluded_song_keys: set[str] | None = None,
) -> dict[str, object]:
    if min(pairs, frames, references_per_speaker) <= 0:
        raise ValueError("pairs, frames, and references must be positive")
    excluded_song_keys = excluded_song_keys or set()
    fingerprint = manifest_fingerprint(entries)
    by_id = {entry.id: entry for entry in entries}
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("schema_version") != 2:
            raise ValueError("unsupported counterfactual pair schema")
        if spec.get("manifest_fingerprint_sha256") != fingerprint:
            raise ValueError("counterfactual manifest changed")
        if spec.get("excluded_song_keys") != sorted(excluded_song_keys):
            raise ValueError("counterfactual exclusion panel changed")
        for pair in spec.get("pairs", []):
            items = (
                pair["source"],
                *pair["source_references"],
                *pair["target_references"],
            )
            for item in items:
                entry = by_id.get(item["id"])
                if entry is None or entry.audio_sha256 != item["audio_sha256"]:
                    raise ValueError(f"counterfactual source changed: {item['id']}")
                if item.get("feature_sha256"):
                    actual = {
                        name: _file_sha256(value)
                        for name, value in _panel_feature_paths(entry).items()
                    }
                    if actual != item["feature_sha256"]:
                        raise ValueError(f"counterfactual features changed: {entry.id}")
        return spec

    grouped: dict[str, dict[str, list[ManifestEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        if (
            entry.quality_status == "accepted"
            and entry.split in {"validation", "test"}
            and entry.dataset not in SYNTHETIC_DATASETS
            and entry.speaker_key in speaker_to_id
            and entry.frames >= frames
            and f"{entry.dataset}:{entry.song}" not in excluded_song_keys
        ):
            grouped[entry.speaker_key][entry.song].append(entry)
    eligible = {
        speaker: songs
        for speaker, songs in grouped.items()
        if len(songs) >= references_per_speaker + 1
    }
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for speaker in eligible:
        by_dataset[speaker.split(":", 1)[0]].append(speaker)
    by_dataset = {
        dataset: sorted(speakers, key=lambda value: _stable_key(seed, value))
        for dataset, speakers in by_dataset.items()
        if len(speakers) >= 2
    }
    candidates: list[tuple[str, str, int]] = []
    offsets = {dataset: 0 for dataset in by_dataset}
    while len(candidates) < pairs:
        progressed = False
        for dataset in sorted(by_dataset):
            speakers = by_dataset[dataset]
            offset = offsets[dataset]
            maximum = len(speakers) * (len(speakers) - 1)
            if offset < maximum:
                source_index = offset % len(speakers)
                rotation = offset // len(speakers)
                candidates.append(
                    (
                        speakers[source_index],
                        speakers[
                            (source_index + rotation + 1) % len(speakers)
                        ],
                        rotation,
                    )
                )
                offsets[dataset] += 1
                progressed = True
                if len(candidates) == pairs:
                    break
        if not progressed:
            break
    if len(candidates) < pairs:
        raise ValueError(
            f"only {len(candidates)} speakers have enough held-out songs "
            f"for {pairs} pairs"
        )

    pair_rows = []
    for index, (source_speaker, target_speaker, rotation) in enumerate(candidates):
        source_songs = _ordered_songs(eligible[source_speaker], seed, source_speaker)
        target_songs = _ordered_songs(eligible[target_speaker], seed, target_speaker)
        source_songs = source_songs[rotation:] + source_songs[:rotation]
        source = _entry_item(
            source_songs[0], frames, seed + index, include_features=True
        )
        target_songs = [
            entry for entry in target_songs if entry.song != source_songs[0].song
        ]
        source_refs = [
            _entry_item(entry, frames, seed + 10_000 + index * 10 + ref)
            for ref, entry in enumerate(source_songs[1 : references_per_speaker + 1])
        ]
        target_refs = [
            _entry_item(entry, frames, seed + 20_000 + index * 10 + ref)
            for ref, entry in enumerate(target_songs[:references_per_speaker])
        ]
        pair_rows.append(
            {
                "source_speaker": source_speaker,
                "target_speaker": target_speaker,
                "seed": seed + index,
                "source": source,
                "source_references": source_refs,
                "target_references": target_refs,
            }
        )
    spec = {
        "schema_version": 2,
        "manifest_fingerprint_sha256": fingerprint,
        "excluded_song_keys": sorted(excluded_song_keys),
        "selection": (
            "real same-dataset speaker rotation; held-out song-disjoint references"
        ),
        "seed": seed,
        "frames": frames,
        "references_per_speaker": references_per_speaker,
        "coverage": {
            "pairs": len(pair_rows),
            "source_speakers": len({row["source_speaker"] for row in pair_rows}),
            "target_speakers": len({row["target_speaker"] for row in pair_rows}),
            "source_song_units": len(
                {
                    (row["source_speaker"].split(":", 1)[0], row["source"]["song"])
                    for row in pair_rows
                }
            ),
            "excluded_song_units": len(excluded_song_keys),
        },
        "pairs": pair_rows,
    }
    _atomic_json(path, spec)
    return spec


def _ordered_songs(
    songs: dict[str, list[ManifestEntry]], seed: int, speaker: str
) -> list[ManifestEntry]:
    ordered = sorted(songs, key=lambda song: _stable_key(seed, speaker, song))
    return [max(songs[song], key=lambda entry: entry.frames) for song in ordered]


def _entry_item(
    entry: ManifestEntry, frames: int, seed: int, *, include_features: bool = False
) -> dict[str, object]:
    start = _panel_crop_start(_load_f0(entry), frames, seed, prefer_voiced=True)
    item: dict[str, object] = {
        "id": entry.id,
        "audio_sha256": entry.audio_sha256,
        "song": entry.song,
        "start_frame": start,
        "frames": frames,
    }
    if include_features:
        item["feature_sha256"] = {
            name: _file_sha256(value)
            for name, value in _panel_feature_paths(entry).items()
        }
    return item


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def manifest_fingerprint(entries: list[ManifestEntry]) -> str:
    payload = [asdict(entry) for entry in sorted(entries, key=lambda item: item.id)]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_excluded_song_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("samples"), list
    ):
        raise ValueError("unsupported exclusion panel lock")
    return {str(row["song_key"]) for row in payload["samples"]}


def speaker_target_margin(
    similarity_to_target: float, similarity_to_source: float
) -> float:
    return similarity_to_target - similarity_to_source


def centroid_cache_key(
    items: list[dict[str, object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item["id"], int(item["start_frame"]), int(item["frames"]))
        for item in items
    )


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for condition in ("target", "source", "null"):
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        names = (
            "similarity_to_target",
            "similarity_to_source",
            "target_margin",
            "content_cosine",
            "voicing_f1",
            "f0_cents_mae",
        )
        result[condition] = {
            name: float(
                sum(float(row[name]) for row in selected if row[name] is not None)
                / sum(row[name] is not None for row in selected)
            )
            for name in names
            if any(row[name] is not None for row in selected)
        }
    return result


def audit_counterfactual(
    *,
    config: V4Config,
    entries: list[ManifestEntry],
    checkpoint: dict[str, object],
    spec: dict[str, object],
    pc_nsf_checkout: Path,
    pc_nsf_lock: Path,
    vocoder_checkpoint: Path,
    contentvec_model: Path,
    contentvec_lock: Path,
    device: torch.device,
    steps: int,
    guidance: float,
    states: tuple[str, ...] = ("ema",),
) -> dict[str, object]:
    by_id = {entry.id: entry for entry in entries}
    speaker_to_id = checkpoint["speaker_to_id"]
    stats_payload = checkpoint["mel_stats"]
    stats = MelStats(
        tuple(stats_payload["mean"]),
        tuple(stats_payload["std"]),
        int(stats_payload["frames"]),
    )
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    lock = PCNSFLock.load(pc_nsf_lock)
    lock.validate_contract(config)
    lock.verify_checkout(pc_nsf_checkout)
    lock.verify_installed_checkpoint(vocoder_checkpoint)
    vocoder = load_pc_nsf_generator(
        pc_nsf_checkout, vocoder_checkpoint, device, config
    )
    verify_contentvec_snapshot(contentvec_model, contentvec_lock, config)
    content_encoder = FrozenContentEncoder.from_local_pretrained(
        contentvec_model, config.model.content_dim
    ).to(device)
    speaker_processor = AutoFeatureExtractor.from_pretrained(
        config.evaluation.speaker_encoder_repository,
        revision=config.evaluation.speaker_encoder_revision,
    )
    speaker_encoder = WavLMForXVector.from_pretrained(
        config.evaluation.speaker_encoder_repository,
        revision=config.evaluation.speaker_encoder_revision,
    ).to(device).eval()
    try:
        from torchfcpe import spawn_bundled_infer_model
    except ImportError as error:
        raise RuntimeError("install the 'features' extra for F0 audit") from error
    pitch_model = spawn_bundled_infer_model(device=str(device))

    centroids: dict[tuple[tuple[object, ...], ...], torch.Tensor] = {}

    def centroid(items: list[dict[str, object]]) -> torch.Tensor:
        key = centroid_cache_key(items)
        if key not in centroids:
            waves = [
                _item_waveform(by_id[item["id"]], item, config) for item in items
            ]
            embeddings = [
                _speaker_embedding(
                    wave,
                    config.sample_rate,
                    speaker_processor,
                    speaker_encoder,
                    device,
                )
                for wave in waves
            ]
            centroids[key] = F.normalize(torch.stack(embeddings).mean(0), dim=0)
        return centroids[key]

    all_rows: dict[str, list[dict[str, object]]] = {}
    for state_name in states:
        state = checkpoint["model" if state_name == "raw" else "ema"]
        system.load_state_dict(state, strict=True)
        rows = []
        for index, pair in enumerate(spec["pairs"]):
            source_centroid = centroid(pair["source_references"])
            target_centroid = centroid(pair["target_references"])
            source_entry = by_id[pair["source"]["id"]]
            source_item = pair["source"]
            frames = int(source_item["frames"])
            content, source_f0, rms = _load_panel_features(
                source_entry, config, int(source_item["start_frame"]), frames
            )
            condition_ids = {
                "target": speaker_to_id[pair["target_speaker"]],
                "source": speaker_to_id[pair["source_speaker"]],
                "null": system.model.null_speaker_id,
            }
            for condition, speaker_id in condition_ids.items():
                generated = sample_chunked(
                    system,
                    content[None].to(device),
                    source_f0[None, :, None].to(device),
                    rms[None, :, None].to(device),
                    torch.tensor([speaker_id], device=device),
                    torch.ones(1, frames, dtype=torch.bool, device=device),
                    config.mel.channels,
                    steps,
                    guidance,
                    "heun",
                    torch.Generator(device=device).manual_seed(int(pair["seed"])),
                    config.inference.time_schedule,
                    config.sampling.frame_buckets[-1],
                    64,
                )
                waveform = synthesize_pc_nsf_tensors(
                    vocoder, stats.denormalize(generated), source_f0, device, config
                )
                output_embedding = _speaker_embedding(
                    waveform,
                    config.sample_rate,
                    speaker_processor,
                    speaker_encoder,
                    device,
                )
                source_similarity = float(output_embedding @ source_centroid)
                target_similarity = float(output_embedding @ target_centroid)
                _, generated_f0, _ = extract_auxiliary_features(
                    waveform.to(device), config, pitch_model
                )
                pitch = pitch_metrics(source_f0, generated_f0)
                generated_content = encode_chunked(
                    content_encoder,
                    AF.resample(
                        waveform,
                        config.sample_rate,
                        config.content_encoder.sample_rate,
                    ),
                    config.content_encoder.sample_rate,
                    30.0,
                    1.0,
                    device,
                    config.content_encoder.phase_shift_seconds,
                )
                generated_content = F.interpolate(
                    generated_content.T[None],
                    size=content.shape[0],
                    mode="linear",
                    align_corners=False,
                )[0].T
                rows.append(
                    {
                        "pair": index,
                        "condition": condition,
                        "source_speaker": pair["source_speaker"],
                        "target_speaker": pair["target_speaker"],
                        "source_song": source_item["song"],
                        "similarity_to_target": target_similarity,
                        "similarity_to_source": source_similarity,
                        "target_margin": speaker_target_margin(
                            target_similarity, source_similarity
                        ),
                        "content_cosine": float(
                            F.cosine_similarity(
                                generated_content.flatten(), content.flatten(), dim=0
                            )
                        ),
                        "voicing_f1": pitch["voicing_f1"],
                        "f0_cents_mae": pitch["f0_cents_mae"],
                    }
                )
            print(
                json.dumps(
                    {
                        "state": state_name,
                        "completed_pairs": index + 1,
                        "total_pairs": len(spec["pairs"]),
                    }
                ),
                flush=True,
            )
        all_rows[state_name] = rows
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint.get("step")),
        "states": list(states),
        "protocol": {
            "pairs": len(spec["pairs"]),
            "frames": spec["frames"],
            "steps": steps,
            "guidance": guidance,
            "conditions": ["target", "source", "null"],
            "speaker_encoder": {
                "repository": config.evaluation.speaker_encoder_repository,
                "revision": config.evaluation.speaker_encoder_revision,
            },
        },
        "aggregate": {state: aggregate_rows(rows) for state, rows in all_rows.items()},
        "samples": all_rows,
    }


@torch.inference_mode()
def _speaker_embedding(
    waveform,
    sample_rate: int,
    processor,
    model,
    device: torch.device,
) -> torch.Tensor:
    waveform = AF.resample(
        torch.as_tensor(waveform).float(), sample_rate, 16_000
    ).clamp(-1, 1)
    inputs = processor(
        waveform.numpy(), sampling_rate=16_000, return_tensors="pt"
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    return F.normalize(model(**inputs).embeddings[0].float(), dim=0).cpu()


def _item_waveform(
    entry: ManifestEntry, item: dict[str, object], config: V4Config
) -> torch.Tensor:
    return _reference_crop(
        entry, config, int(item["start_frame"]), int(item["frames"])
    )


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
