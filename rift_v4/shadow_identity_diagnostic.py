from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

from .config import V4Config
from .counterfactual import (
    _entry_item,
    _item_waveform,
    _speaker_embedding,
    centroid_cache_key,
    manifest_fingerprint,
)
from .features import MelStats
from .manifest import ManifestEntry, load_manifest
from .shadow_panel import (
    _load_checkpoint,
    file_sha256,
    load_locked_tensors,
    load_or_create_panel_lock,
)
from .shadow_v3_compare import load_v3
from .third_party import PCNSFLock
from .train import build_system
from .vocoder import load_pc_nsf_generator, synthesize_pc_nsf_tensors

MODELS = ("v3_null", "v4_null", "v4_correct", "v4_wrong")
CUTOFFS = (8, 16, 32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose V3/V4 precision, identity, and spectral-envelope error"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--references", type=int, default=2)
    parser.add_argument("--impostors", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    v4_checkpoint = _load_checkpoint(args.v4_checkpoint, config)
    speaker_to_id = v4_checkpoint["speaker_to_id"]
    raw_lock = json.loads(args.panel_lock.read_text(encoding="utf-8"))
    panel_lock = load_or_create_panel_lock(
        args.panel_lock,
        entries,
        args.manifest,
        args.mel_stats,
        speaker_to_id,
        set(config.sampling.synthetic_datasets),
        int(raw_lock["protocol"]["panel_size"]),
        int(raw_lock["protocol"]["minimum_unique_songs"]),
        tuple(int(value) for value in raw_lock["protocol"]["frames"]),
        int(raw_lock["protocol"]["seed"]),
    )
    identity_lock = load_or_create_identity_lock(
        args.identity_lock,
        entries,
        args.manifest,
        panel_lock,
        args.panel_lock,
        speaker_to_id,
        args.frames,
        args.references,
        args.impostors,
        args.seed,
    )
    tensors = load_locked_tensors(panel_lock, entries, config, stats, speaker_to_id)
    wrong_ids = torch.tensor(
        [speaker_to_id[row["wrong_speaker"]] for row in identity_lock["samples"]],
        dtype=torch.long,
    )
    maximum_frames = max(int(value) for value in panel_lock["protocol"]["frames"])
    noise = torch.randn(
        len(panel_lock["samples"]),
        maximum_frames,
        config.mel.channels,
        generator=torch.Generator().manual_seed(int(panel_lock["protocol"]["seed"])),
    )[:, : args.frames]
    device = torch.device(args.device)

    predictions: dict[str, torch.Tensor] = {}
    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    predictions["v3_null"] = integrate_model(
        v3,
        "v3_null",
        tensors,
        wrong_ids,
        noise,
        args.frames,
        args.intervals,
        args.batch_size,
        device,
    )
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(speaker_to_id)).to(device).eval()
    system.load_state_dict(v4_checkpoint["ema"], strict=True)
    for name in ("v4_null", "v4_correct", "v4_wrong"):
        predictions[name] = integrate_model(
            system.model,
            name,
            tensors,
            wrong_ids,
            noise,
            args.frames,
            args.intervals,
            args.batch_size,
            device,
        )
    del system, v4_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_target = stats.denormalize(tensors["mel"][:, : args.frames].float())
    raw_predictions = {
        name: (
            (value + 1.0) * 7.0 - 12.0
            if name == "v3_null"
            else stats.denormalize(value)
        )
        for name, value in predictions.items()
    }
    spectral_rows = spectral_diagnostics(
        raw_predictions,
        raw_target,
        tensors["f0"][:, : args.frames],
        tensors["rms"][:, : args.frames],
        panel_lock["samples"],
        CUTOFFS,
    )

    pc_lock = PCNSFLock.load(args.pc_nsf_lock)
    pc_lock.validate_contract(config)
    pc_lock.verify_checkout(args.pc_nsf_checkout)
    pc_lock.verify_installed_checkpoint(args.vocoder_checkpoint)
    vocoder = load_pc_nsf_generator(
        args.pc_nsf_checkout, args.vocoder_checkpoint, device, config
    )
    processor = AutoFeatureExtractor.from_pretrained(
        config.evaluation.speaker_encoder_repository,
        revision=config.evaluation.speaker_encoder_revision,
    )
    encoder = (
        WavLMForXVector.from_pretrained(
            config.evaluation.speaker_encoder_repository,
            revision=config.evaluation.speaker_encoder_revision,
        )
        .to(device)
        .eval()
    )
    identity_rows = speaker_diagnostics(
        raw_predictions,
        tensors["f0"][:, : args.frames],
        entries,
        identity_lock,
        config,
        vocoder,
        processor,
        encoder,
        device,
    )
    rows = merge_rows(spectral_rows, identity_rows)
    payload = {
        "schema_version": 1,
        "protocol": {
            "panel_lock": str(args.panel_lock),
            "panel_lock_sha256": file_sha256(args.panel_lock),
            "identity_lock": str(args.identity_lock),
            "identity_lock_sha256": file_sha256(args.identity_lock),
            "frames": args.frames,
            "intervals": args.intervals,
            "solver": "linear Euler",
            "noise": "same locked 768-frame Gaussian tensor prefix",
            "models": list(MODELS),
            "dct": "orthonormal DCT-II along 128-bin log-mel frequency axis",
            "dct_cutoffs": list(CUTOFFS),
            "speaker_encoder": {
                "repository": config.evaluation.speaker_encoder_repository,
                "revision": config.evaluation.speaker_encoder_revision,
            },
            "speaker_references": args.references,
            "unrelated_impostors": args.impostors,
        },
        "checkpoints": {
            "v3": str(args.v3_checkpoint),
            "v4": str(args.v4_checkpoint),
        },
        "coverage": panel_lock["coverage"],
        "summary": summarize_diagnostic(
            rows, panel_lock["samples"], args.bootstrap_samples, args.seed
        ),
        "samples": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


def load_or_create_identity_lock(
    path: Path,
    entries: list[ManifestEntry],
    manifest_path: Path,
    panel_lock: dict[str, object],
    panel_lock_path: Path,
    speaker_to_id: dict[str, int],
    frames: int,
    references: int,
    impostors: int,
    seed: int,
) -> dict[str, object]:
    manifest_sha256 = manifest_fingerprint(entries)
    panel_sha256 = file_sha256(panel_lock_path)
    by_id = {entry.id: entry for entry in entries}
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "manifest_fingerprint_sha256": manifest_sha256,
            "panel_lock_sha256": panel_sha256,
            "frames": frames,
            "references": references,
            "impostors": impostors,
            "seed": seed,
        }
        if lock.get("schema_version") != 1:
            raise ValueError("unsupported identity lock schema")
        for name, value in expected.items():
            if lock["protocol"].get(name) != value:
                raise ValueError(f"identity lock changed: {name}")
        for row in lock["samples"]:
            for item in (
                *row["source_references"],
                *row["wrong_references"],
                *(ref for group in row["impostor_references"] for ref in group),
            ):
                entry = by_id.get(item["id"])
                if entry is None or entry.audio_sha256 != item["audio_sha256"]:
                    raise ValueError(f"identity reference changed: {item['id']}")
        return lock

    grouped: dict[str, dict[str, list[ManifestEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        if (
            entry.quality_status == "accepted"
            and entry.frames >= frames
            and entry.speaker_key in speaker_to_id
        ):
            grouped[entry.speaker_key][entry.song].append(entry)
    speakers = sorted(grouped)
    rows = []
    for index, panel in enumerate(panel_lock["samples"]):
        source = str(panel["speaker"])
        source_song = str(panel["song"])
        source_refs = select_references(
            grouped[source], source_song, references, frames, seed, index, "source"
        )
        candidates = [speaker for speaker in speakers if speaker != source]
        preferred = [
            speaker
            for speaker in candidates
            if speaker.split(":", 1)[0] == source.split(":", 1)[0]
            and voice_group(speaker) == voice_group(source)
        ]
        if not preferred:
            preferred = [
                speaker
                for speaker in candidates
                if voice_group(speaker) == voice_group(source)
            ]
        ordered = sorted(
            preferred or candidates,
            key=lambda speaker: stable_key(seed, str(index), source, speaker),
        )
        wrong = ordered[0]
        wrong_refs = select_references(
            grouped[wrong], source_song, references, frames, seed, index, "wrong"
        )
        impostor_speakers = ordered[:impostors]
        if len(impostor_speakers) < impostors:
            fallback = sorted(
                (speaker for speaker in candidates if speaker not in impostor_speakers),
                key=lambda speaker: stable_key(seed, "fallback", str(index), speaker),
            )
            impostor_speakers.extend(fallback[: impostors - len(impostor_speakers)])
        impostor_refs = [
            select_references(
                grouped[speaker],
                source_song,
                references,
                frames,
                seed,
                index,
                f"impostor-{position}",
            )
            for position, speaker in enumerate(impostor_speakers)
        ]
        rows.append(
            {
                "ordinal": int(panel["ordinal"]),
                "entry_id": panel["entry_id"],
                "source_speaker": source,
                "wrong_speaker": wrong,
                "impostor_speakers": impostor_speakers,
                "source_references": source_refs,
                "wrong_references": wrong_refs,
                "impostor_references": impostor_refs,
            }
        )
    lock = {
        "schema_version": 1,
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_fingerprint_sha256": manifest_sha256,
            "panel_lock_sha256": panel_sha256,
            "frames": frames,
            "references": references,
            "impostors": impostors,
            "seed": seed,
            "selection": (
                "different-song references; same-dataset voice-group impostors "
                "when available"
            ),
        },
        "samples": rows,
    }
    _atomic_json(path, lock)
    return lock


def select_references(
    songs: dict[str, list[ManifestEntry]],
    excluded_song: str,
    count: int,
    frames: int,
    seed: int,
    ordinal: int,
    role: str,
) -> list[dict[str, object]]:
    candidates = [
        max(entries, key=lambda entry: entry.frames)
        for song, entries in songs.items()
        if song != excluded_song
    ]
    candidates.sort(
        key=lambda entry: stable_key(seed, role, str(ordinal), entry.song, entry.id)
    )
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} reference songs for {role}")
    return [
        _entry_item(entry, frames, seed + ordinal * 100 + position)
        for position, entry in enumerate(candidates[:count])
    ]


def voice_group(speaker: str) -> str:
    value = speaker.lower()
    if any(token in value for token in ("female", "soprano", "alto", "kiritan")):
        return "female"
    if any(token in value for token in ("male", "tenor", "bass", "baritone")):
        return "male"
    return "unknown"


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


@torch.inference_mode()
def integrate_model(
    model,
    model_name: str,
    tensors: dict[str, torch.Tensor],
    wrong_ids: torch.Tensor,
    noise: torch.Tensor,
    frames: int,
    intervals: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for begin in range(0, len(noise), batch_size):
        end = min(begin + batch_size, len(noise))
        content = tensors["content"][begin:end, :frames].to(device)
        f0 = tensors["f0"][begin:end, :frames].to(device)
        rms = tensors["rms"][begin:end, :frames].to(device)
        state = noise[begin:end].to(device).clone()
        mask = torch.ones(end - begin, frames, dtype=torch.bool, device=device)
        correct = tensors["speaker"][begin:end].to(device)
        times = torch.linspace(0, 1, intervals + 1, device=device)
        for index in range(intervals):
            timestep = times[index].expand(end - begin)
            if model_name == "v3_null":
                velocity = model(
                    x=state,
                    spk=torch.zeros(end - begin, dtype=torch.long, device=device),
                    f0=f0.squeeze(-1),
                    rms=rms.squeeze(-1),
                    cvec=content,
                    time=timestep,
                    mask=mask,
                    drop_speaker=True,
                )
            else:
                if model_name == "v4_null":
                    speaker = torch.full_like(correct, model.null_speaker_id)
                elif model_name == "v4_wrong":
                    speaker = wrong_ids[begin:end].to(device)
                else:
                    speaker = correct
                velocity = model(state, content, f0, rms, speaker, timestep, mask)
            state += (times[index + 1] - times[index]) * velocity.float()
        outputs.append(state.float().cpu())
        print(
            json.dumps({"stage": "generate", "model": model_name, "completed": end}),
            flush=True,
        )
    return torch.cat(outputs)


def orthonormal_dct(size: int) -> torch.Tensor:
    frequency = torch.arange(size, dtype=torch.float64)[:, None]
    position = torch.arange(size, dtype=torch.float64)[None, :]
    matrix = torch.cos(math.pi / size * (position + 0.5) * frequency)
    matrix *= math.sqrt(2.0 / size)
    matrix[0] /= math.sqrt(2.0)
    return matrix.float()


def spectral_diagnostics(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    f0: torch.Tensor,
    rms: torch.Tensor,
    metadata: list[dict[str, object]],
    cutoffs: tuple[int, ...],
) -> list[dict[str, object]]:
    dct = orthonormal_dct(target.shape[-1])
    active = rms.squeeze(-1) > 1e-3
    voiced = active & (f0.squeeze(-1) > 0)
    unvoiced = active & ~voiced
    rows = []
    for sample, item in enumerate(metadata):
        models = {}
        for name, prediction in predictions.items():
            residual = prediction[sample] - target[sample]
            frame_mse = residual.square().mean(-1)
            selected = active[sample]
            coeff = residual[selected] @ dct.T if bool(selected.any()) else None
            total = float(coeff.square().mean()) if coeff is not None else None
            row: dict[str, object] = {
                "active_raw_mse": masked_mean(frame_mse, selected),
                "voiced_raw_mse": masked_mean(frame_mse, voiced[sample]),
                "unvoiced_active_raw_mse": masked_mean(frame_mse, unvoiced[sample]),
                "per_bin_active_raw_mse": (
                    residual[selected].square().mean(0).tolist()
                    if coeff is not None
                    else None
                ),
            }
            for cutoff in cutoffs:
                low = (
                    float(coeff[:, :cutoff].square().sum() / coeff.numel())
                    if coeff is not None
                    else None
                )
                high = (
                    float(coeff[:, cutoff:].square().sum() / coeff.numel())
                    if coeff is not None
                    else None
                )
                row[f"dct_{cutoff}"] = {
                    "low_contribution_mse": low,
                    "high_contribution_mse": high,
                    "low_fraction": (
                        low / total if total is not None and total > 0 else None
                    ),
                }
            models[name] = row
        rows.append(
            {
                "ordinal": int(item["ordinal"]),
                "entry_id": item["entry_id"],
                "dataset": item["dataset"],
                "speaker": item["speaker"],
                "song_key": item["song_key"],
                "models": models,
            }
        )
    return rows


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    return float(values[mask].mean()) if bool(mask.any()) else None


@torch.inference_mode()
def speaker_diagnostics(
    predictions: dict[str, torch.Tensor],
    f0: torch.Tensor,
    entries: list[ManifestEntry],
    identity_lock: dict[str, object],
    config: V4Config,
    vocoder,
    processor,
    encoder,
    device: torch.device,
) -> list[dict[str, object]]:
    by_id = {entry.id: entry for entry in entries}
    embeddings: dict[tuple[object, ...], torch.Tensor] = {}

    def embedding(item: dict[str, object]) -> torch.Tensor:
        key = (item["id"], int(item["start_frame"]), int(item["frames"]))
        if key not in embeddings:
            embeddings[key] = _speaker_embedding(
                _item_waveform(by_id[item["id"]], item, config),
                config.sample_rate,
                processor,
                encoder,
                device,
            )
        return embeddings[key]

    def centroid(items: list[dict[str, object]]) -> torch.Tensor:
        key = ("centroid", *centroid_cache_key(items))
        if key not in embeddings:
            embeddings[key] = F.normalize(
                torch.stack([embedding(item) for item in items]).mean(0), dim=0
            )
        return embeddings[key]

    rows = []
    for sample, item in enumerate(identity_lock["samples"]):
        source = centroid(item["source_references"])
        wrong = centroid(item["wrong_references"])
        impostors = [centroid(refs) for refs in item["impostor_references"]]
        models = {}
        for name in MODELS:
            waveform = synthesize_pc_nsf_tensors(
                vocoder, predictions[name][sample], f0[sample], device, config
            )
            value = _speaker_embedding(
                waveform,
                config.sample_rate,
                processor,
                encoder,
                device,
            )
            source_similarity = float(value @ source)
            wrong_similarity = float(value @ wrong)
            unrelated = torch.tensor([float(value @ other) for other in impostors])
            models[name] = {
                "source_similarity": source_similarity,
                "wrong_similarity": wrong_similarity,
                "unrelated_similarity_mean": float(unrelated.mean()),
                "unrelated_similarity_max": float(unrelated.max()),
                "source_margin_mean": source_similarity - float(unrelated.mean()),
                "source_margin_max": source_similarity - float(unrelated.max()),
                "wrong_margin": wrong_similarity - source_similarity,
            }
        rows.append({"ordinal": item["ordinal"], "models": models})
        print(
            json.dumps({"stage": "speaker", "completed": sample + 1, "total": len(f0)}),
            flush=True,
        )
    return rows


def merge_rows(
    spectral: list[dict[str, object]], identity: list[dict[str, object]]
) -> list[dict[str, object]]:
    identity_by_ordinal = {int(row["ordinal"]): row for row in identity}
    for row in spectral:
        other = identity_by_ordinal[int(row["ordinal"])]
        for name in MODELS:
            row["models"][name].update(other["models"][name])
    return spectral


def summarize_diagnostic(
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    summary: dict[str, object] = {"models": {}}
    metrics = (
        "active_raw_mse",
        "voiced_raw_mse",
        "unvoiced_active_raw_mse",
        "source_similarity",
        "unrelated_similarity_mean",
        "source_margin_mean",
        "wrong_similarity",
        "wrong_margin",
    )
    for name in MODELS:
        summary["models"][name] = {
            metric: distribution([row["models"][name][metric] for row in rows])
            for metric in metrics
        }
        summary["models"][name]["dct"] = {
            str(cutoff): {
                key: distribution(
                    [row["models"][name][f"dct_{cutoff}"][key] for row in rows]
                )
                for key in (
                    "low_contribution_mse",
                    "high_contribution_mse",
                    "low_fraction",
                )
            }
            for cutoff in CUTOFFS
        }

    correlation_rows = [
        (row, item)
        for row, item in zip(rows, metadata, strict=True)
        if row["models"]["v4_correct"]["active_raw_mse"] is not None
        and row["models"]["v3_null"]["active_raw_mse"] is not None
    ]
    mse_advantage = [
        row["models"]["v4_correct"]["active_raw_mse"]
        - row["models"]["v3_null"]["active_raw_mse"]
        for row, _ in correlation_rows
    ]
    similarity_advantage = [
        row["models"]["v3_null"]["source_similarity"]
        - row["models"]["v4_correct"]["source_similarity"]
        for row, _ in correlation_rows
    ]
    correlation_metadata = [item for _, item in correlation_rows]
    summary["v3_advantage_correlation"] = correlation_report(
        mse_advantage,
        similarity_advantage,
        correlation_metadata,
        bootstrap_samples,
        seed,
    )
    non_kiritan = [row["dataset"] != "Kiritan" for row, _ in correlation_rows]
    summary["v3_advantage_correlation_non_kiritan"] = correlation_report(
        [value for value, keep in zip(mse_advantage, non_kiritan, strict=True) if keep],
        [
            value
            for value, keep in zip(similarity_advantage, non_kiritan, strict=True)
            if keep
        ],
        [
            item
            for item, keep in zip(correlation_metadata, non_kiritan, strict=True)
            if keep
        ],
        bootstrap_samples,
        seed + 1,
    )
    margin_advantage = [
        row["models"]["v3_null"]["source_margin_mean"]
        - row["models"]["v4_correct"]["source_margin_mean"]
        for row, _ in correlation_rows
    ]
    voiced_advantage = [
        row["models"]["v4_correct"]["voiced_raw_mse"]
        - row["models"]["v3_null"]["voiced_raw_mse"]
        for row, _ in correlation_rows
    ]
    high_advantage = [
        row["models"]["v4_correct"]["dct_16"]["high_contribution_mse"]
        - row["models"]["v3_null"]["dct_16"]["high_contribution_mse"]
        for row, _ in correlation_rows
    ]
    low_advantage = [
        row["models"]["v4_correct"]["dct_16"]["low_contribution_mse"]
        - row["models"]["v3_null"]["dct_16"]["low_contribution_mse"]
        for row, _ in correlation_rows
    ]
    summary["speaker_explanation_correlations"] = {
        "active_mse_vs_source_margin": correlation_report(
            mse_advantage,
            margin_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 2,
        ),
        "voiced_mse_vs_source_similarity": correlation_report(
            voiced_advantage,
            similarity_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 3,
        ),
        "voiced_mse_vs_source_margin": correlation_report(
            voiced_advantage,
            margin_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 4,
        ),
        "dct16_high_vs_source_similarity": correlation_report(
            high_advantage,
            similarity_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 5,
        ),
        "dct16_high_vs_source_margin": correlation_report(
            high_advantage,
            margin_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 6,
        ),
    }
    summary["error_component_correlations"] = {
        "active_mse_vs_voiced_mse": correlation_report(
            mse_advantage,
            voiced_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 7,
        ),
        "active_mse_vs_dct16_high": correlation_report(
            mse_advantage,
            high_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 8,
        ),
        "active_mse_vs_dct16_low": correlation_report(
            mse_advantage,
            low_advantage,
            correlation_metadata,
            bootstrap_samples,
            seed + 9,
        ),
    }
    summary["v4_wrong_minus_correct"] = paired_model_delta(
        rows,
        metadata,
        "v4_correct",
        "v4_wrong",
        bootstrap_samples,
        seed + 10,
    )
    summary["v4_correct_minus_v3"] = paired_model_delta(
        rows,
        metadata,
        "v3_null",
        "v4_correct",
        bootstrap_samples,
        seed + 11,
    )
    summary["v4_correct_minus_v3"]["dct"] = {
        str(cutoff): {
            component: paired_nested_delta(
                rows,
                metadata,
                "v3_null",
                "v4_correct",
                f"dct_{cutoff}",
                component,
                bootstrap_samples,
                seed + 20 + cutoff + component.startswith("high"),
            )
            for component in ("low_contribution_mse", "high_contribution_mse")
        }
        for cutoff in CUTOFFS
    }
    return summary


def distribution(values: list[float | None]) -> dict[str, float | int]:
    selected = torch.tensor(
        [float(value) for value in values if value is not None], dtype=torch.float64
    )
    return {
        "samples": len(selected),
        "mean": float(selected.mean()),
        "median": float(torch.quantile(selected, 0.5)),
        "p10": float(torch.quantile(selected, 0.1)),
        "p90": float(torch.quantile(selected, 0.9)),
    }


def paired_model_delta(
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
    before: str,
    after: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    metrics = (
        "active_raw_mse",
        "source_similarity",
        "unrelated_similarity_mean",
        "source_margin_mean",
        "wrong_similarity",
        "wrong_margin",
    )
    result = {}
    for metric in metrics:
        selected = [
            (row["models"][after][metric] - row["models"][before][metric], item)
            for row, item in zip(rows, metadata, strict=True)
            if row["models"][after][metric] is not None
            and row["models"][before][metric] is not None
        ]
        values = [value for value, _ in selected]
        result[metric] = distribution(values)
        result[metric]["song_bootstrap_mean_95_ci"] = clustered_mean_interval(
            values,
            [item for _, item in selected],
            bootstrap_samples,
            seed,
        )
        result[metric]["win_rate_after"] = sum(
            value < 0 if metric == "active_raw_mse" else value > 0 for value in values
        ) / len(values)
    return result


def paired_nested_delta(
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
    before: str,
    after: str,
    group: str,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    selected = [
        (
            row["models"][after][group][metric]
            - row["models"][before][group][metric],
            item,
        )
        for row, item in zip(rows, metadata, strict=True)
        if row["models"][after][group][metric] is not None
        and row["models"][before][group][metric] is not None
    ]
    values = [value for value, _ in selected]
    result = distribution(values)
    result["song_bootstrap_mean_95_ci"] = clustered_mean_interval(
        values,
        [item for _, item in selected],
        bootstrap_samples,
        seed,
    )
    result["win_rate_after"] = sum(value < 0 for value in values) / len(values)
    return result


def clustered_mean_interval(
    values: list[float],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> list[float]:
    value_tensor = torch.tensor(values, dtype=torch.float64)
    by_song: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        by_song[str(item["song_key"])].append(index)
    groups = list(by_song.values())
    generator = torch.Generator().manual_seed(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        chosen = torch.randint(len(groups), (len(groups),), generator=generator)
        indices = [index for group in chosen.tolist() for index in groups[group]]
        bootstrap.append(float(value_tensor[indices].mean()))
    interval = torch.quantile(
        torch.tensor(bootstrap, dtype=torch.float64),
        torch.tensor((0.025, 0.975), dtype=torch.float64),
    )
    return [float(value) for value in interval]


def correlation_report(
    x: list[float],
    y: list[float],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    x_tensor = torch.tensor(x, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64)
    pearson = correlation(x_tensor, y_tensor)
    spearman = correlation(rank(x_tensor), rank(y_tensor))
    slope = float(
        ((x_tensor - x_tensor.mean()) * (y_tensor - y_tensor.mean())).sum()
        / (x_tensor - x_tensor.mean()).square().sum()
    )
    by_song: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        by_song[str(item["song_key"])].append(index)
    groups = list(by_song.values())
    generator = torch.Generator().manual_seed(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        chosen = torch.randint(len(groups), (len(groups),), generator=generator)
        indices = [index for group in chosen.tolist() for index in groups[group]]
        bootstrap.append(correlation(rank(x_tensor[indices]), rank(y_tensor[indices])))
    interval = torch.quantile(
        torch.tensor(bootstrap, dtype=torch.float64),
        torch.tensor((0.025, 0.975), dtype=torch.float64),
    )
    return {
        "samples": len(x),
        "pearson": pearson,
        "spearman": spearman,
        "spearman_song_bootstrap_95_ci": [float(value) for value in interval],
        "similarity_advantage_per_mse_slope": slope,
    }


def rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    result = torch.empty_like(values)
    result[order] = torch.arange(len(values), dtype=values.dtype)
    return result


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator)


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
