from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system

SPEAKERS = (
    "GTSinger:French-FR-Soprano-1",
    "GTSinger:Japanese-JA-Soprano-1",
    "GTSinger:Korean-KO-Soprano-2",
    "GTSinger:Spanish-ES-Bass-1",
    "M4Singer:Alto-1",
    "M4Singer:Alto-3",
    "M4Singer:Bass-2",
    "M4Singer:Tenor-5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def best_start(f0: torch.Tensor, frames: int) -> int:
    voiced = (f0.reshape(-1) > 0).float()
    if voiced.numel() <= frames:
        return 0
    stride = max(1, frames // 8)
    scores = F.conv1d(voiced[None, None], torch.ones(1, 1, frames), stride=stride)
    return int(scores.argmax()) * stride


def select_panel(entries, dataset, frames: int, splits: set[str]):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if (
            entry.quality_status == "accepted"
            and entry.split in splits
            and entry.speaker_key in SPEAKERS
            and entry.frames >= frames
        ):
            grouped[entry.speaker_key][entry.song].append(entry)
    panel = []
    for speaker in SPEAKERS:
        songs = grouped[speaker]
        if len(songs) < 2:
            raise RuntimeError(f"{speaker}: fewer than two songs in {sorted(splits)}")
        for song in sorted(songs)[:2]:
            entry = max(songs[song], key=lambda item: item.frames)
            features = dataset._load_features(entry)
            start = best_start(features["f0"], frames)
            panel.append(
                (
                    entry,
                    start,
                    {
                        key: value[start : start + frames]
                        for key, value in features.items()
                    },
                )
            )
    return panel


def audit_panel(model, panel, speaker_to_id, stats, frames, times, args, device):
    normalized_mel = torch.stack([features["mel"] for _, _, features in panel]).to(
        device
    )
    raw_mel = stats.denormalize(normalized_mel)
    content = torch.stack([features["content"] for _, _, features in panel]).to(device)
    f0 = torch.stack([features["f0"] for _, _, features in panel]).to(device)
    rms = torch.stack([features["rms"] for _, _, features in panel]).to(device)
    speaker = torch.tensor(
        [speaker_to_id[entry.speaker_key] for entry, _, _ in panel],
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones((len(panel), frames), dtype=torch.bool, device=device)
    noise = torch.randn(
        normalized_mel.shape,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)
    sigma = normalized_mel.new_tensor(stats.std).view(1, 1, -1)

    by_t = {}
    flow_losses = []
    velocity_nmse = []
    with torch.inference_mode(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for value in times:
            time = torch.full((len(panel),), value, device=device)
            target = normalized_mel - noise
            state = (1.0 - value) * noise + value * normalized_mel
            prediction = model(state, content, f0, rms, speaker, time, mask).float()
            normalized_mse = prediction.sub(target).square().mean()
            raw_prediction = sigma * prediction
            raw_target = sigma * target
            raw_mse = raw_prediction.sub(raw_target).square().mean()
            raw_nmse = raw_mse / raw_target.square().mean().clamp_min(1e-12)
            cosine = F.cosine_similarity(
                raw_prediction.flatten(1), raw_target.flatten(1), dim=1
            ).mean()
            flow_losses.append(float(normalized_mse))
            velocity_nmse.append(float(raw_nmse))
            by_t[str(value)] = {
                "normalized_flow_mse": float(normalized_mse),
                "raw_velocity_mse": float(raw_mse),
                "raw_velocity_nmse": float(raw_nmse),
                "raw_velocity_cosine": float(cosine),
            }

        generated = noise.clone()
        times = torch.linspace(0.0, 1.0, args.intervals + 1, device=device)
        for index in range(args.intervals):
            time = times[index].expand(len(panel))
            generated += (times[index + 1] - times[index]) * model(
                generated, content, f0, rms, speaker, time, mask
            ).float()
        raw_prediction = stats.denormalize(generated)
        residual = raw_prediction - raw_mel
        per_sample_frame_mse = residual.square().mean(dim=-1)
        target_rms = rms.squeeze(-1)
        silence_mask = target_rms <= 1e-3
        active_mask = ~silence_mask
        endpoint_mse = per_sample_frame_mse.mean()
        endpoint_l1 = residual.abs().mean()
        endpoint_cosine = F.cosine_similarity(
            raw_prediction.flatten(1), raw_mel.flatten(1), dim=1
        ).mean()

    sample_endpoint = []
    for index, (entry, start, _) in enumerate(panel):
        sample_endpoint.append(
            {
                "speaker": entry.speaker_key,
                "song": entry.song,
                "split": entry.split,
                "entry_id": entry.id,
                "start_frame": start,
                "full_raw_mse": float(per_sample_frame_mse[index].mean()),
                "active_raw_mse": _masked_mean(
                    per_sample_frame_mse[index], active_mask[index]
                ),
                "silence_raw_mse": _masked_mean(
                    per_sample_frame_mse[index], silence_mask[index]
                ),
                "active_frames": int(active_mask[index].sum()),
                "silence_frames": int(silence_mask[index].sum()),
            }
        )

    return {
        "fixed_t_mean_normalized_flow_mse": sum(flow_losses) / len(flow_losses),
        "fixed_t_mean_raw_velocity_nmse": sum(velocity_nmse) / len(velocity_nmse),
        "endpoint_raw_mse": float(endpoint_mse),
        "endpoint_raw_l1": float(endpoint_l1),
        "endpoint_raw_cosine": float(endpoint_cosine),
        "endpoint_partition": {
            "silence_threshold_rms": 1e-3,
            "silence_threshold_dbfs": -60.0,
            "full_raw_mse": float(endpoint_mse),
            "active_raw_mse": _masked_mean(per_sample_frame_mse, active_mask),
            "silence_raw_mse": _masked_mean(per_sample_frame_mse, silence_mask),
            "active_frames": int(active_mask.sum()),
            "silence_frames": int(silence_mask.sum()),
        },
        "by_t": by_t,
        "samples": sample_endpoint,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    dataset = FeatureDataset(
        entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        checkpoint["speaker_to_id"],
        voiced_crop_probability=0.0,
    )
    frames_values = tuple(args.frames or config.evaluation.endpoint_panel_frames)
    if len(SPEAKERS) * 2 != config.evaluation.endpoint_panel_samples:
        raise ValueError("endpoint panel sample count differs from fixed speaker panel")
    maximum_frames = max(frames_values)
    base_panels = {
        "train": select_panel(entries, dataset, maximum_frames, {"train"}),
        "song_disjoint_validation": select_panel(
            entries, dataset, maximum_frames, {"validation", "test"}
        ),
    }
    system = build_system(config, len(checkpoint["speaker_to_id"])).to(device).eval()
    payload = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_states": ["raw", "ema"],
        "speaker_conditioning": "correct physical speaker",
        "frames": frames_values,
        "noise_seed": args.seed,
        "times": config.evaluation.rf_audit_timesteps,
        "solver": f"linear Euler, {args.intervals} intervals",
        "results": {},
    }
    for state_name, state in (("raw", checkpoint["model"]), ("ema", checkpoint["ema"])):
        system.load_state_dict(state, strict=True)
        payload["results"][state_name] = {}
        for frames in frames_values:
            payload["results"][state_name][str(frames)] = {}
            for name, base_panel in base_panels.items():
                panel = [
                    (
                        entry,
                        start,
                        {key: value[:frames] for key, value in features.items()},
                    )
                    for entry, start, features in base_panel
                ]
                print(f"auditing {state_name} {frames} {name}", flush=True)
                result = audit_panel(
                    system.model,
                    panel,
                    checkpoint["speaker_to_id"],
                    stats,
                    frames,
                    config.evaluation.rf_audit_timesteps,
                    args,
                    device,
                )
                payload["results"][state_name][str(frames)][name] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
