from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from audit_known_speaker_conditioning import (
    SPEAKERS,
    TIMES,
    aggregate,
    best_start,
    integrate_v3,
    integrate_v4,
    load_v3,
)
from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system


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
                (entry, start, {key: value[start : start + frames] for key, value in features.items()})
            )
    return panel


def wrong_speakers(speakers: list[str]) -> list[str]:
    groups = defaultdict(list)
    for speaker in SPEAKERS:
        groups[speaker.split(":", 1)[0]].append(speaker)
    result = []
    for speaker in speakers:
        group = groups[speaker.split(":", 1)[0]]
        result.append(group[(group.index(speaker) + 1) % len(group)])
    return result


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def sensitivity(left, right, scale):
    left_raw = left.float() * scale
    right_raw = right.float() * scale
    delta = left_raw - right_raw
    sample = delta.square().mean((1, 2)).sqrt()
    reference = left_raw.square().mean((1, 2)).sqrt().clamp_min(1e-12)
    return {
        "delta_rms": float(sample.mean()),
        "relative_to_left_output_rms": float((sample / reference).mean()),
        "sample_delta_rms": [float(value) for value in sample],
    }


def injection_audit(model, correct, null, wrong, times):
    labels = {"correct": correct, "null": null, "wrong": wrong}
    speaker_embed = {name: model.speaker(index) for name, index in labels.items()}
    result = {
        "speaker_embedding": {
            name: {"rms": rms(value), "norm_mean": float(value.float().norm(dim=-1).mean())}
            for name, value in speaker_embed.items()
        },
        "speaker_delta_rms": {
            "correct_null": rms(speaker_embed["correct"] - speaker_embed["null"]),
            "correct_wrong": rms(speaker_embed["correct"] - speaker_embed["wrong"]),
            "wrong_null": rms(speaker_embed["wrong"] - speaker_embed["null"]),
        },
        "by_t": {},
    }
    chunk_names = ("shift_attn", "scale_attn", "gate_attn", "shift_ffn", "scale_ffn", "gate_ffn")
    for value in times:
        time = model.time(torch.full((correct.shape[0],), value, device=correct.device))
        conditioning = {name: time + embed for name, embed in speaker_embed.items()}
        layers = []
        for index, block in enumerate(model.blocks):
            mod = {name: block.modulation(cond[:, None]).chunk(6, dim=-1) for name, cond in conditioning.items()}
            layer = {"layer": index}
            for chunk, chunk_name in enumerate(chunk_names):
                layer[chunk_name] = {
                    "correct_rms": rms(mod["correct"][chunk]),
                    "null_rms": rms(mod["null"][chunk]),
                    "wrong_rms": rms(mod["wrong"][chunk]),
                    "correct_null_delta_rms": rms(mod["correct"][chunk] - mod["null"][chunk]),
                    "correct_wrong_delta_rms": rms(mod["correct"][chunk] - mod["wrong"][chunk]),
                }
            layers.append(layer)
        final = {name: model.final_modulation(cond[:, None]).chunk(2, dim=-1) for name, cond in conditioning.items()}
        result["by_t"][str(value)] = {
            "time_embedding_rms": rms(time),
            "conditioning_rms": {name: rms(cond) for name, cond in conditioning.items()},
            "conditioning_delta_rms": {
                "correct_null": rms(conditioning["correct"] - conditioning["null"]),
                "correct_wrong": rms(conditioning["correct"] - conditioning["wrong"]),
            },
            "layers": layers,
            "final_modulation": {
                branch: {
                    "correct_rms": rms(final["correct"][index]),
                    "null_rms": rms(final["null"][index]),
                    "wrong_rms": rms(final["wrong"][index]),
                    "correct_null_delta_rms": rms(final["correct"][index] - final["null"][index]),
                    "correct_wrong_delta_rms": rms(final["correct"][index] - final["wrong"][index]),
                }
                for index, branch in enumerate(("shift", "scale"))
            },
        }
    return result


def audit_panel(name, panel, args, stats, v3, v4_model, speaker_to_id):
    device = next(v4_model.parameters()).device
    speakers = [entry.speaker_key for entry, _, _ in panel]
    wrong_names = wrong_speakers(speakers)
    raw_mel = torch.stack([stats.denormalize(features["mel"]) for _, _, features in panel]).to(device)
    content = torch.stack([features["content"] for _, _, features in panel]).to(device)
    f0 = torch.stack([features["f0"] for _, _, features in panel]).to(device)
    loudness = torch.stack([features["rms"] for _, _, features in panel]).to(device)
    mask = torch.ones((len(panel), args.frames), dtype=torch.bool, device=device)
    noise = torch.randn(raw_mel.shape, generator=torch.Generator().manual_seed(args.seed)).to(device)
    v3_mel = (raw_mel + 12.0) / 7.0 - 1.0
    v4_mel = stats.normalize(raw_mel)
    sigma = raw_mel.new_tensor(stats.std).view(1, 1, -1)
    v3_ids = torch.zeros(len(panel), dtype=torch.long, device=device)
    correct = torch.tensor([speaker_to_id[value] for value in speakers], device=device)
    wrong = torch.tensor([speaker_to_id[value] for value in wrong_names], device=device)
    null = torch.full_like(correct, v4_model.null_speaker_id)
    ids = {"correct": correct, "null": null, "wrong": wrong}

    with torch.inference_mode():
        injection = injection_audit(v4_model, correct, null, wrong, TIMES)
    result = {"endpoint": {}, "timestep": {}, "injection": injection}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        v3_endpoint = integrate_v3(v3, noise.clone(), content, f0.squeeze(-1), loudness.squeeze(-1), v3_ids, mask, args.intervals).float()
        result["endpoint"]["v3_null"] = aggregate((v3_endpoint + 1) * 7 - 12, raw_mel, speakers)
        endpoints = {}
        for condition, speaker_ids in ids.items():
            endpoint = integrate_v4(v4_model, noise.clone(), content, f0, loudness, speaker_ids, mask, args.intervals).float()
            endpoints[condition] = endpoint
            result["endpoint"][f"v4_{condition}"] = aggregate(stats.denormalize(endpoint), raw_mel, speakers)
        result["endpoint_sensitivity"] = {
            "correct_null": sensitivity(endpoints["correct"], endpoints["null"], sigma),
            "correct_wrong": sensitivity(endpoints["correct"], endpoints["wrong"], sigma),
            "wrong_null": sensitivity(endpoints["wrong"], endpoints["null"], sigma),
        }
        for value in TIMES:
            time = torch.full((len(panel),), value, device=device)
            v3_target = v3_mel - noise
            v3_prediction = v3(x=(1-value)*noise+value*v3_mel, spk=v3_ids, f0=f0.squeeze(-1), rms=loudness.squeeze(-1), cvec=content, time=time, mask=mask, drop_speaker=True).float()
            row = {"metrics": {"v3_null": aggregate(7*v3_prediction, 7*v3_target, speakers)}, "sensitivity": {}}
            predictions = {}
            v4_target = v4_mel - noise
            for condition, speaker_ids in ids.items():
                prediction = v4_model((1-value)*noise+value*v4_mel, content, f0, loudness, speaker_ids, time, mask).float()
                predictions[condition] = prediction
                row["metrics"][f"v4_{condition}"] = aggregate(sigma*prediction, sigma*v4_target, speakers)
            row["sensitivity"] = {
                "correct_null": sensitivity(predictions["correct"], predictions["null"], sigma),
                "correct_wrong": sensitivity(predictions["correct"], predictions["wrong"], sigma),
                "wrong_null": sensitivity(predictions["wrong"], predictions["null"], sigma),
            }
            result["timestep"][str(value)] = row
    result["panel"] = [{"speaker": entry.speaker_key, "wrong_speaker": wrong_names[index], "song": entry.song, "split": entry.split, "entry_id": entry.id, "start_frame": start} for index, (entry, start, _) in enumerate(panel)]
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = V4Config.load(args.v4_config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    dataset = FeatureDataset(entries, config.mel.channels, config.model.content_dim, stats, checkpoint["speaker_to_id"], voiced_crop_probability=0.0)
    v3, _ = load_v3(args.v3_checkpoint, device)
    system = build_system(config, len(checkpoint["speaker_to_id"])).to(device).eval()
    system.load_state_dict(checkpoint["ema"], strict=True)
    panels = {
        "heldout": select_panel(entries, dataset, args.frames, {"validation", "test"}),
        "train": select_panel(entries, dataset, args.frames, {"train"}),
    }
    payload = {
        "schema_version": 1,
        "protocol": {"speakers": list(SPEAKERS), "crops_per_speaker_per_panel": 2, "frames": args.frames, "noise_seed": args.seed, "wrong_speaker": "fixed rotation among four speakers in the same dataset", "v4_state": "EMA"},
        "panels": {name: audit_panel(name, panel, args, stats, v3, system.model, checkpoint["speaker_to_id"]) for name, panel in panels.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for name, panel in payload["panels"].items():
        print(name, {key: value["mse"] for key, value in panel["endpoint"].items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
