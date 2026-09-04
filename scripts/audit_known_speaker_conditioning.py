from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system

# The upstream package __init__ imports its entire training stack. This audit only
# needs the architecture, so expose the source directory as a package without
# importing Lightning, W&B, vocoder, or pitch-extraction dependencies.
_v3_source = Path(os.environ["RIFT_V3_SOURCE"]).resolve()
_v3_package = types.ModuleType("rift_svc")
_v3_package.__path__ = [str(_v3_source / "rift_svc")]
sys.modules["rift_svc"] = _v3_package
from rift_svc.dit import DiT as V3DiT


SPEAKERS = (
    "GTSinger:French-FR-Soprano-1",
    "GTSinger:Japanese-JA-Soprano-1",
    "GTSinger:Korean-KO-Soprano-2",
    "GTSinger:Spanish-ES-Bass-1",
    "M4Singer:Alto-5",
    "M4Singer:Alto-6",
    "M4Singer:Bass-1",
    "M4Singer:Tenor-5",
)
TIMES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_v3(path: Path, device: torch.device) -> tuple[V3DiT, dict[str, int]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["hyper_parameters"]["cfg"]
    model = V3DiT(num_speaker=1, **cfg["model"])
    prefix = "model.transformer."
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["spk_embed.weight"] or unexpected:
        raise RuntimeError(f"V3 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval(), cfg["spk2idx"]


def best_start(f0: torch.Tensor, frames: int) -> int:
    voiced = (f0.reshape(-1) > 0).float()
    if voiced.numel() <= frames:
        return 0
    scores = F.conv1d(
        voiced[None, None], torch.ones(1, 1, frames), stride=max(1, frames // 8)
    ).flatten()
    return int(scores.argmax()) * max(1, frames // 8)


def select_panel(entries, dataset: FeatureDataset, frames: int):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if (
            entry.quality_status == "accepted"
            and entry.split in {"validation", "test"}
            and entry.speaker_key in SPEAKERS
            and entry.frames >= frames
        ):
            grouped[entry.speaker_key][entry.song].append(entry)
    panel = []
    for speaker in SPEAKERS:
        songs = grouped[speaker]
        if len(songs) < 2:
            raise RuntimeError(f"{speaker}: fewer than two held-out songs")
        for song in sorted(songs)[:2]:
            entry = max(songs[song], key=lambda item: item.frames)
            features = dataset._load_features(entry)
            start = best_start(features["f0"], frames)
            stop = start + frames
            panel.append((entry, start, {key: value[start:stop] for key, value in features.items()}))
    return panel


def aggregate(prediction: torch.Tensor, target: torch.Tensor, speakers: list[str]):
    error = prediction - target
    mse = error.square().mean((1, 2))
    l1 = error.abs().mean((1, 2))
    target_energy = target.square().mean((1, 2)).clamp_min(1e-12)
    nmse = mse / target_energy
    cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    by_speaker = {}
    for speaker in SPEAKERS:
        indices = [index for index, value in enumerate(speakers) if value == speaker]
        by_speaker[speaker] = {
            "mse": float(mse[indices].mean()),
            "nmse": float(nmse[indices].mean()),
            "l1": float(l1[indices].mean()),
            "cosine": float(cosine[indices].mean()),
        }
    return {
        "mse": float(mse.mean()),
        "nmse": float(nmse.mean()),
        "l1": float(l1.mean()),
        "cosine": float(cosine.mean()),
        "sample_mse": [float(value) for value in mse],
        "sample_nmse": [float(value) for value in nmse],
        "sample_l1": [float(value) for value in l1],
        "sample_cosine": [float(value) for value in cosine],
        "by_speaker": by_speaker,
    }


def integrate_v3(model, state, content, f0, rms, speaker, mask, intervals):
    times = torch.linspace(0, 1, intervals + 1, device=state.device)
    for index in range(intervals):
        velocity = model(
            x=state, spk=speaker, f0=f0, rms=rms, cvec=content,
            time=times[index].expand(state.shape[0]), mask=mask,
            drop_speaker=True,
        )
        state = state + (times[index + 1] - times[index]) * velocity
    return state


def integrate_v4(model, state, content, f0, rms, speaker, mask, intervals):
    times = torch.linspace(0, 1, intervals + 1, device=state.device)
    for index in range(intervals):
        velocity = model(
            state, content, f0, rms, speaker,
            times[index].expand(state.shape[0]), mask,
        )
        state = state + (times[index + 1] - times[index]) * velocity
    return state


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = V4Config.load(args.v4_config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    v4_checkpoint = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    dataset = FeatureDataset(
        entries, config.mel.channels, config.model.content_dim, stats,
        v4_checkpoint["speaker_to_id"], voiced_crop_probability=0.0,
    )
    panel = select_panel(entries, dataset, args.frames)
    speakers = [entry.speaker_key for entry, _, _ in panel]
    raw_mel = torch.stack([
        stats.denormalize(features["mel"]) for _, _, features in panel
    ]).to(device)
    content = torch.stack([features["content"] for _, _, features in panel]).to(device)
    f0 = torch.stack([features["f0"] for _, _, features in panel]).to(device)
    rms = torch.stack([features["rms"] for _, _, features in panel]).to(device)
    mask = torch.ones((len(panel), args.frames), dtype=torch.bool, device=device)
    noise = torch.randn(raw_mel.shape, generator=torch.Generator().manual_seed(args.seed)).to(device)

    v3, v3_mapping = load_v3(args.v3_checkpoint, device)
    missing_names = [speaker for speaker in speakers if not v3_name(speaker) in v3_mapping]
    if missing_names:
        raise RuntimeError(f"speakers absent from V3 mapping: {missing_names}")
    v3_mel = (raw_mel + 12.0) / 7.0 - 1.0
    v3_speaker = torch.zeros(len(panel), dtype=torch.long, device=device)

    v4_system = build_system(config, len(v4_checkpoint["speaker_to_id"])).to(device).eval()
    v4_system.load_state_dict(v4_checkpoint["ema"], strict=True)
    v4_model = v4_system.model
    v4_correct = torch.tensor(
        [v4_checkpoint["speaker_to_id"][speaker] for speaker in speakers],
        dtype=torch.long, device=device,
    )
    v4_null = torch.full_like(v4_correct, v4_model.null_speaker_id)
    v4_mel = stats.normalize(raw_mel)
    sigma = raw_mel.new_tensor(stats.std).view(1, 1, -1)

    results = {"endpoint": {}, "timestep": {}}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        v3_endpoint = integrate_v3(
            v3, noise.clone(), content, f0.squeeze(-1), rms.squeeze(-1),
            v3_speaker, mask, args.intervals,
        ).float()
        results["endpoint"]["v3_null"] = aggregate(
            (v3_endpoint + 1.0) * 7.0 - 12.0, raw_mel, speakers
        )
        for name, speaker_ids in (("v4_correct", v4_correct), ("v4_null", v4_null)):
            endpoint = integrate_v4(
                v4_model, noise.clone(), content, f0, rms, speaker_ids, mask,
                args.intervals,
            ).float()
            results["endpoint"][name] = aggregate(
                stats.denormalize(endpoint), raw_mel, speakers
            )

        for value in TIMES:
            time = torch.full((len(panel),), value, device=device)
            v3_target = v3_mel - noise
            v3_prediction = v3(
                x=(1 - value) * noise + value * v3_mel,
                spk=v3_speaker, f0=f0.squeeze(-1), rms=rms.squeeze(-1),
                cvec=content, time=time, mask=mask, drop_speaker=True,
            ).float()
            row = {"v3_null": aggregate(7.0 * v3_prediction, 7.0 * v3_target, speakers)}
            v4_target = v4_mel - noise
            for name, speaker_ids in (("v4_correct", v4_correct), ("v4_null", v4_null)):
                prediction = v4_model(
                    (1 - value) * noise + value * v4_mel,
                    content, f0, rms, speaker_ids, time, mask,
                ).float()
                row[name] = aggregate(sigma * prediction, sigma * v4_target, speakers)
            results["timestep"][str(value)] = row

    payload = {
        "schema_version": 1,
        "protocol": {
            "frames": args.frames,
            "noise_seed": args.seed,
            "integration": f"linear Euler, {args.intervals} intervals",
            "v4_state": "EMA",
            "shared_features": "V4 raw mel and aligned ContentVec/FCPE-F0/RMS",
            "v3_physical_embedding": "unavailable in published checkpoint",
            "v3_split_provenance": "unavailable; V4 validation/test songs used",
        },
        "checkpoints": {"v3": str(args.v3_checkpoint), "v4": str(args.v4_checkpoint)},
        "panel": [
            {
                "dataset": entry.dataset, "speaker": entry.speaker_key,
                "song": entry.song, "split": entry.split, "entry_id": entry.id,
                "start_frame": start,
            }
            for entry, start, _ in panel
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"endpoint": results["endpoint"]}, indent=2), flush=True)
    print(f"wrote {args.output}")


def v3_name(v4_name: str) -> str:
    dataset, speaker = v4_name.split(":", 1)
    if dataset == "M4Singer":
        return f"M4Singer-{speaker}"
    if dataset == "GTSinger":
        language_code, voice = speaker.split("-", 1)
        del language_code
        return f"gtsinger-{voice}"
    raise ValueError(f"unsupported speaker mapping: {v4_name}")


if __name__ == "__main__":
    main()
