from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

from .config import V4Config
from .data import _matrix, _resize, _vector
from .features import MelStats, extract_auxiliary_features
from .infer import sample_chunked
from .manifest import ManifestEntry, load_manifest
from .third_party import PCNSFLock
from .train import build_system
from .vocoder import (
    _write_waveform,
    load_pc_nsf_generator,
    synthesize_pc_nsf_tensors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a fixed validation audio panel and pitch audit"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    samples = args.samples or config.evaluation.audio_panel_samples
    frames = args.frames or config.evaluation.audio_panel_frames
    inference_steps = args.steps or config.evaluation.inference_steps
    guidance = (
        args.guidance if args.guidance is not None else config.evaluation.guidance
    )
    if samples <= 0 or frames <= 0 or inference_steps <= 0:
        raise ValueError("samples, frames, and inference steps must be positive")
    entries = load_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("checkpoint does not use schema 4")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint configuration differs from panel configuration")
    speaker_to_id = checkpoint["speaker_to_id"]
    panel = load_or_create_panel(
        args.panel_spec,
        entries,
        speaker_to_id,
        samples,
        frames,
        args.seed,
    )
    render_panel(
        config,
        entries,
        checkpoint,
        panel,
        args.output,
        args.pc_nsf_checkout,
        args.pc_nsf_lock,
        args.vocoder_checkpoint,
        torch.device(args.device),
        inference_steps,
        guidance,
    )


def load_or_create_panel(
    path: Path,
    entries: list[ManifestEntry],
    speaker_to_id: dict[str, int],
    samples: int,
    frames: int,
    seed: int,
) -> dict[str, object]:
    by_id = {entry.id: entry for entry in entries}
    if path.exists():
        panel = json.loads(path.read_text(encoding="utf-8"))
        if panel.get("schema_version") != 1:
            raise ValueError("unsupported audio panel schema")
        for item in panel.get("samples", []):
            entry = by_id.get(item["id"])
            if entry is None or entry.audio_sha256 != item["audio_sha256"]:
                raise ValueError(f"panel source changed or is missing: {item['id']}")
            if entry.speaker_key != item["speaker"]:
                raise ValueError(f"panel speaker changed: {entry.id}")
            for name, feature_path in _panel_feature_paths(entry).items():
                if _file_sha256(feature_path) != item[f"{name}_sha256"]:
                    raise ValueError(f"panel {name} feature changed: {entry.id}")
        return panel

    candidates = [
        entry
        for entry in entries
        if entry.split == "validation"
        and entry.quality_status == "accepted"
        and entry.speaker_key in speaker_to_id
    ]
    if len(candidates) < samples:
        raise ValueError(f"audio panel needs {samples} validation entries")
    selected = _balanced_panel_entries(candidates, samples, seed)
    items = []
    for index, entry in enumerate(selected):
        f0 = _load_f0(entry)
        wanted = min(frames, f0.numel())
        crop_seed = seed + index
        crop_start = _panel_crop_start(
            f0, wanted, crop_seed, prefer_voiced=index < math.ceil(samples * 0.75)
        )
        items.append(
            {
                "id": entry.id,
                "dataset": entry.dataset,
                "speaker": entry.speaker_key,
                "song": entry.song,
                "audio_sha256": entry.audio_sha256,
                "crop_start_frame": crop_start,
                "frames": wanted,
                "seed": crop_seed,
                **{
                    f"{name}_sha256": _file_sha256(feature_path)
                    for name, feature_path in _panel_feature_paths(entry).items()
                },
            }
        )
    panel = {
        "schema_version": 1,
        "selection": "dataset-round-robin; 75% voiced-aware and 25% random crops",
        "seed": seed,
        "samples": items,
    }
    _atomic_json(path, panel)
    return panel


def _balanced_panel_entries(
    entries: list[ManifestEntry], samples: int, seed: int
) -> list[ManifestEntry]:
    grouped: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.dataset, []).append(entry)
    for dataset, values in grouped.items():
        values.sort(key=lambda entry: _stable_key(seed, dataset, entry.id))
    selected: list[ManifestEntry] = []
    offsets = {dataset: 0 for dataset in grouped}
    datasets = sorted(grouped)
    while len(selected) < samples:
        progressed = False
        for dataset in datasets:
            offset = offsets[dataset]
            if offset < len(grouped[dataset]):
                selected.append(grouped[dataset][offset])
                offsets[dataset] += 1
                progressed = True
                if len(selected) == samples:
                    break
        if not progressed:
            break
    return selected


def render_panel(
    config: V4Config,
    entries: list[ManifestEntry],
    checkpoint: dict[str, object],
    panel: dict[str, object],
    output_root: Path,
    pc_nsf_checkout: Path,
    pc_nsf_lock: Path,
    vocoder_checkpoint: Path,
    device: torch.device,
    inference_steps: int,
    guidance: float,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    step = int(checkpoint["step"])
    output_root.mkdir(parents=True, exist_ok=True)
    final_destination = output_root / f"step-{step:07d}"
    if final_destination.exists():
        raise FileExistsError(f"panel output already exists: {final_destination}")
    destination = Path(tempfile.mkdtemp(dir=output_root, prefix=f".step-{step:07d}."))

    speaker_to_id = checkpoint["speaker_to_id"]
    system = build_system(config, len(speaker_to_id))
    system.load_state_dict(checkpoint.get("ema", checkpoint["model"]), strict=True)
    system.to(device).eval()
    stats_payload = checkpoint["mel_stats"]
    stats = MelStats(
        tuple(stats_payload["mean"]),
        tuple(stats_payload["std"]),
        int(stats_payload["frames"]),
    )
    lock = PCNSFLock.load(pc_nsf_lock)
    lock.validate_contract(config)
    lock.verify_checkout(pc_nsf_checkout)
    lock.verify_installed_checkpoint(vocoder_checkpoint)
    vocoder = load_pc_nsf_generator(pc_nsf_checkout, vocoder_checkpoint, device, config)
    try:
        from torchfcpe import spawn_bundled_infer_model
    except ImportError as error:
        raise RuntimeError("install the 'features' extra for panel F0 audit") from error
    pitch_model = spawn_bundled_infer_model(device=str(device))

    by_id = {entry.id: entry for entry in entries}
    results = []
    for index, item in enumerate(panel["samples"]):
        entry = by_id[item["id"]]
        start = int(item["crop_start_frame"])
        frames = int(item["frames"])
        content, target_f0, rms = _load_panel_features(entry, config, start, frames)
        mask = torch.ones(1, frames, dtype=torch.bool, device=device)
        speaker = torch.tensor([speaker_to_id[entry.speaker_key]], device=device)
        generated = sample_chunked(
            system,
            content[None].to(device),
            target_f0[None, :, None].to(device),
            rms[None, :, None].to(device),
            speaker,
            mask,
            config.mel.channels,
            inference_steps,
            guidance,
            "heun",
            torch.Generator(device=device).manual_seed(int(item["seed"])),
            config.inference.time_schedule,
            config.sampling.frame_buckets[-1],
            64,
        )
        mel = stats.denormalize(generated)
        waveform = synthesize_pc_nsf_tensors(vocoder, mel, target_f0, device, config)
        _, generated_f0, _ = extract_auxiliary_features(
            waveform.to(device), config, pitch_model
        )
        name = f"{index:02d}-{_safe_name(entry.dataset)}-{_safe_name(entry.speaker)}"
        _write_waveform(
            destination / f"{name}-generated.wav", waveform, config.sample_rate
        )
        reference = _reference_crop(entry, config, start, frames)
        _write_waveform(
            destination / f"{name}-reference.wav", reference, config.sample_rate
        )
        metrics = pitch_metrics(target_f0, generated_f0)
        results.append(
            {
                **item,
                "generated": f"{name}-generated.wav",
                "reference": f"{name}-reference.wav",
                "pitch": metrics,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "checkpoint_step": step,
        "checkpoint": f"step-{step:07d}.pt",
        "inference": {
            "steps": inference_steps,
            "guidance": guidance,
            "method": "heun",
            "time_schedule": config.inference.time_schedule,
        },
        "aggregate_pitch": _aggregate_pitch(results),
        "samples": results,
    }
    _atomic_json(destination / "metrics.json", payload)
    os.replace(destination, final_destination)
    with (output_root / "panel_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def pitch_metrics(
    target_f0: torch.Tensor, generated_f0: torch.Tensor
) -> dict[str, float | int | None]:
    target = target_f0.float().flatten()
    generated = generated_f0.float().flatten()
    frames = min(target.numel(), generated.numel())
    target = target[:frames]
    generated = generated[:frames]
    target_voiced = target > 0
    generated_voiced = generated > 0
    both = target_voiced & generated_voiced
    true_positive = int(both.sum())
    precision = true_positive / max(1, int(generated_voiced.sum()))
    recall = true_positive / max(1, int(target_voiced.sum()))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    cents = 1200 * torch.log2(generated[both] / target[both]) if both.any() else None
    return {
        "frames": frames,
        "target_voiced_ratio": float(target_voiced.float().mean()),
        "generated_voiced_ratio": float(generated_voiced.float().mean()),
        "voicing_precision": precision,
        "voicing_recall": recall,
        "voicing_f1": f1,
        "f0_cents_mae": float(cents.abs().mean()) if cents is not None else None,
        "gross_pitch_error_ratio": (
            float((cents.abs() > 50).float().mean()) if cents is not None else None
        ),
    }


def _aggregate_pitch(results: list[dict[str, object]]) -> dict[str, float]:
    names = (
        "target_voiced_ratio",
        "generated_voiced_ratio",
        "voicing_precision",
        "voicing_recall",
        "voicing_f1",
        "f0_cents_mae",
        "gross_pitch_error_ratio",
    )
    aggregate = {}
    for name in names:
        values = [
            item["pitch"][name] for item in results if item["pitch"][name] is not None
        ]
        if values:
            aggregate[name] = float(sum(values) / len(values))
    return aggregate


def _load_panel_features(
    entry: ManifestEntry, config: V4Config, start: int, frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = Path(entry.feature_prefix)
    f0 = _load_f0(entry)
    rms = (
        _vector(
            torch.load(f"{prefix}.rms.pt", map_location="cpu", weights_only=True), "rms"
        )
        .squeeze(-1)
        .float()
    )
    content = _matrix(
        torch.load(
            entry.content_feature_path or f"{prefix}.content.pt",
            map_location="cpu",
            weights_only=True,
        ),
        config.model.content_dim,
        "content",
    ).float()
    content = _resize(content, f0.numel())
    stop = start + frames
    if stop > f0.numel() or rms.numel() != f0.numel():
        raise ValueError(f"invalid panel feature alignment: {entry.id}")
    return content[start:stop], f0[start:stop], rms[start:stop]


def _load_f0(entry: ManifestEntry) -> torch.Tensor:
    value = (
        _vector(
            torch.load(
                f"{entry.feature_prefix}.f0.pt", map_location="cpu", weights_only=True
            ),
            "f0",
        )
        .squeeze(-1)
        .float()
    )
    if not bool(torch.isfinite(value).all()) or (value < 0).any():
        raise ValueError(f"invalid F0 for panel entry: {entry.id}")
    return value


def _panel_crop_start(
    f0: torch.Tensor, frames: int, seed: int, prefer_voiced: bool
) -> int:
    maximum = f0.numel() - frames
    if maximum <= 0:
        return 0
    generator = random.Random(seed)
    starts = [generator.randrange(maximum + 1) for _ in range(16)]
    if not prefer_voiced:
        return starts[0]
    return max(
        starts,
        key=lambda start: float((f0[start : start + frames] > 0).float().mean()),
    )


def _reference_crop(
    entry: ManifestEntry, config: V4Config, start: int, frames: int
) -> torch.Tensor:
    audio, rate = sf.read(entry.audio_path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"panel reference must be mono: {entry.id}")
    waveform = torch.from_numpy(audio[:, 0])
    if rate != config.sample_rate:
        waveform = AF.resample(waveform, rate, config.sample_rate)
    start_sample = start * config.hop_length
    samples = frames * config.hop_length
    crop = waveform[start_sample : start_sample + samples]
    return torch.nn.functional.pad(crop, (0, max(0, samples - crop.numel())))


def _stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def _panel_feature_paths(entry: ManifestEntry) -> dict[str, Path]:
    prefix = Path(entry.feature_prefix)
    return {
        "f0": Path(f"{prefix}.f0.pt"),
        "rms": Path(f"{prefix}.rms.pt"),
        "content": Path(entry.content_feature_path or f"{prefix}.content.pt"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


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
