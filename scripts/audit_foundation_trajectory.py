from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system


PANELS = (
    ("acapella-baobei--original--01", 1175),
    ("acapella-shuixingji--dereverb--01", 12016),
    ("acapella-xiaoxingyun--dereverb--01", 842),
    ("ggd-365--dereverb--01", 6961),
    ("ggd-370--dereverb--01", 1136),
    ("ggd-47--dereverb--01", 1182),
    ("local-2023-503--dereverb--01", 384),
    ("local-2023-563--original--01", 323),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def integrate(model, state, content, f0, rms, speaker, mask, intervals):
    times = torch.linspace(0.0, 1.0, intervals + 1, device=state.device)
    for index in range(intervals):
        time = times[index].expand(state.shape[0])
        velocity = model(state, content, f0, rms, speaker, time, mask)
        state = state + (times[index + 1] - times[index]) * velocity
    return state


def metrics(prediction, target, sigma):
    residual = prediction - target
    squared = residual.square()
    sample_mse = squared.mean((1, 2))
    sample_l1 = residual.abs().mean((1, 2))
    sample_cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    per_bin_by_sample = squared.mean(1)
    # alpha=0 is the V4 standardized objective; alpha=2 is raw-space MSE.
    normalized_squared = squared / sigma.square().view(1, 1, -1)
    weighted = {
        str(alpha): float(
            (normalized_squared * sigma.pow(alpha).view(1, 1, -1)).mean()
        )
        for alpha in (0, 1, 2)
    }
    return {
        "raw_mse": float(sample_mse.mean()),
        "raw_l1": float(sample_l1.mean()),
        "raw_cosine": float(sample_cosine.mean()),
        "common_standardized_mse": weighted["0"],
        "sigma_alpha_weighted_mse": weighted,
        "sample_raw_mse": [float(value) for value in sample_mse],
        "sample_raw_l1": [float(value) for value in sample_l1],
        "sample_raw_cosine": [float(value) for value in sample_cosine],
        "raw_mse_per_bin": [float(value) for value in per_bin_by_sample.mean(0)],
        "raw_mse_per_bin_by_sample": [
            [float(value) for value in row] for row in per_bin_by_sample
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    sigma = torch.tensor(stats.std, device=device)
    entries = {entry.song: entry for entry in load_manifest(args.manifest)}
    dataset = FeatureDataset(
        list(entries.values()), config.mel.channels, config.model.content_dim,
        stats, {"Target:target": 0}, voiced_crop_probability=0.0,
    )
    loaded = []
    for song, start in PANELS:
        features = dataset._load_features(entries[song])
        end = start + args.frames
        if end > min(value.shape[0] for value in features.values()):
            raise ValueError(f"panel exceeds feature length: {song}@{start}")
        loaded.append(tuple(features[name][start:end] for name in ("mel", "content", "f0", "rms")))
    mel, content, f0, rms = (
        torch.stack(values).to(device) for values in zip(*loaded, strict=True)
    )
    raw_target = stats.denormalize(mel)
    mask = torch.ones((len(PANELS), args.frames), dtype=torch.bool, device=device)
    noise = torch.randn(mel.shape, generator=torch.Generator().manual_seed(args.seed)).to(device)

    results = []
    system = None
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for checkpoint_path in args.checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if system is None:
                system = build_system(config, len(checkpoint["speaker_to_id"])).to(device).eval()
            system.load_state_dict(checkpoint["ema"], strict=True)
            speaker = torch.full(
                (len(PANELS),), system.model.null_speaker_id,
                dtype=torch.long, device=device,
            )
            generated = integrate(
                system.model, noise.clone(), content, f0, rms, speaker, mask,
                args.intervals,
            ).float()
            row = {
                "step": int(checkpoint["step"]),
                "checkpoint": str(checkpoint_path),
                **metrics(stats.denormalize(generated), raw_target, sigma),
            }
            results.append(row)
            print(json.dumps({key: value for key, value in row.items() if "per_bin" not in key}), flush=True)

    payload = {
        "schema_version": 1,
        "protocol": {
            "checkpoint_state": "ema",
            "speaker_conditioning": "trained null speaker embedding",
            "solver": f"linear Euler, {args.intervals} intervals",
            "noise_seed": args.seed,
            "frames": args.frames,
            "panels": [{"song": song, "start_frame": start} for song, start in PANELS],
            "common_standardization": "V4 per-bin sigma",
        },
        "v4_sigma": [float(value) for value in sigma.cpu()],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
