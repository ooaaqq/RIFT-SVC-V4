from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset, SampleRequest
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system


TIMES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
EULER_STEPS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def select_panel(entries, synthetic: set[str], size: int, frames: int, seed: int):
    by_speaker: dict[str, dict[str, list[tuple[int, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, entry in enumerate(entries):
        if (
            entry.quality_status == "accepted"
            and entry.split in {"validation", "test"}
            and entry.dataset not in synthetic
            and entry.frames >= frames
        ):
            by_speaker[entry.speaker_key][entry.song].append((index, entry))
    candidates: dict[str, list[tuple[int, object]]] = {}
    for speaker, songs in by_speaker.items():
        selected = []
        for song, records in songs.items():
            selected.append(
                min(records, key=lambda item: stable_key(seed, item[1].id))
            )
        candidates[speaker] = sorted(
            selected, key=lambda item: stable_key(seed, f"{speaker}\0{item[1].song}")
        )
    speakers = sorted(candidates, key=lambda value: stable_key(seed, value))
    panel = []
    round_index = 0
    while len(panel) < size:
        added = 0
        for speaker in speakers:
            if round_index < len(candidates[speaker]):
                panel.append(candidates[speaker][round_index])
                added += 1
                if len(panel) == size:
                    break
        if not added:
            break
        round_index += 1
    if len(panel) < size:
        raise RuntimeError(f"only {len(panel)} eligible speaker-balanced songs")
    return panel


def fixed_noise_and_validation_t(shape, seed: int):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(shape, generator=generator)
    count = shape[0]
    uniform = (
        torch.arange(count, dtype=torch.float32) / count
        + torch.rand(count, generator=generator) / count
    )
    epsilon = torch.finfo(torch.float32).eps
    normal = torch.erfinv(uniform.clamp(epsilon, 1 - epsilon).mul(2).sub(1))
    timestep = torch.sigmoid(normal.mul(math.sqrt(2.0)))
    timestep = timestep[torch.randperm(count, generator=generator)]
    return noise, timestep


def sample_metrics(prediction: torch.Tensor, target: torch.Tensor):
    residual = prediction - target
    return {
        "raw_mse": residual.square().mean((1, 2)).cpu().tolist(),
        "raw_l1": residual.abs().mean((1, 2)).cpu().tolist(),
        "raw_cosine": F.cosine_similarity(
            prediction.flatten(1), target.flatten(1), dim=1
        ).cpu().tolist(),
    }


def evaluate(checkpoint_path, config, stats, tensors, metadata, args, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    system = build_system(config, len(checkpoint["speaker_to_id"])).to(device).eval()
    system.load_state_dict(checkpoint["ema"], strict=True)
    model = system.model
    sigma = torch.tensor(stats.std, dtype=torch.float32, device=device).view(1, 1, -1)
    output = {
        "step": int(checkpoint["step"]),
        "original_validation": {"normalized_flow_mse": []},
        "velocity_by_t": {
            str(value): {"raw_velocity_nmse": [], "normalized_flow_mse": []}
            for value in TIMES
        },
        "endpoint_by_steps": {
            str(value): {"raw_mse": [], "raw_l1": [], "raw_cosine": []}
            for value in EULER_STEPS
        },
    }
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for begin in range(0, len(metadata), args.batch_size):
            end = min(begin + args.batch_size, len(metadata))
            mel = tensors["mel"][begin:end].to(device)
            content = tensors["content"][begin:end].to(device)
            f0 = tensors["f0"][begin:end].to(device)
            rms = tensors["rms"][begin:end].to(device)
            speaker = tensors["speaker"][begin:end].to(device)
            noise = tensors["noise"][begin:end].to(device)
            validation_t = tensors["validation_t"][begin:end].to(device)
            mask = torch.ones((end - begin, args.frames), dtype=torch.bool, device=device)
            target_velocity = mel - noise

            state = (
                (1.0 - validation_t[:, None, None]) * noise
                + validation_t[:, None, None] * mel
            )
            prediction = model(
                state, content, f0, rms, speaker, validation_t, mask
            ).float()
            validation_mse = prediction.sub(target_velocity).square().mean((1, 2))
            output["original_validation"]["normalized_flow_mse"].extend(
                validation_mse.cpu().tolist()
            )

            for value in TIMES:
                timestep = torch.full((end - begin,), value, device=device)
                state = (1.0 - value) * noise + value * mel
                prediction = model(
                    state, content, f0, rms, speaker, timestep, mask
                ).float()
                normalized_mse = prediction.sub(target_velocity).square().mean((1, 2))
                raw_prediction = sigma * prediction
                raw_target = sigma * target_velocity
                raw_mse = raw_prediction.sub(raw_target).square().mean((1, 2))
                raw_energy = raw_target.square().mean((1, 2)).clamp_min(1e-12)
                row = output["velocity_by_t"][str(value)]
                row["normalized_flow_mse"].extend(normalized_mse.cpu().tolist())
                row["raw_velocity_nmse"].extend((raw_mse / raw_energy).cpu().tolist())

            raw_target = stats.denormalize(mel.float())
            for steps in EULER_STEPS:
                generated = noise.clone()
                times = torch.linspace(0.0, 1.0, steps + 1, device=device)
                for index in range(steps):
                    timestep = times[index].expand(end - begin)
                    generated += (times[index + 1] - times[index]) * model(
                        generated, content, f0, rms, speaker, timestep, mask
                    ).float()
                values = sample_metrics(stats.denormalize(generated.float()), raw_target)
                output["endpoint_by_steps"][str(steps)] = {
                    name: output["endpoint_by_steps"][str(steps)][name] + value
                    for name, value in values.items()
                }
            print(
                json.dumps(
                    {"step": output["step"], "completed": end, "total": len(metadata)}
                ),
                flush=True,
            )
    del system, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
    }


def paired(a, b, metadata, bootstrap_samples: int, seed: int, higher_is_better=False):
    left = torch.tensor(a, dtype=torch.float64)
    right = torch.tensor(b, dtype=torch.float64)
    delta = right - left
    wins = delta > 0 if higher_is_better else delta < 0
    by_speaker = defaultdict(list)
    by_song = defaultdict(list)
    for index, row in enumerate(metadata):
        by_speaker[row["speaker"]].append(float(delta[index]))
        by_song[row["song_key"]].append(float(delta[index]))
    speaker_delta = {
        speaker: sum(values) / len(values)
        for speaker, values in sorted(by_speaker.items())
    }
    # Resample independent songs, not crops. This keeps alternate singers or
    # recordings of the same composition in the same bootstrap unit.
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
    boot = song_values[draws].mean(1)
    interval = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=boot.dtype))
    return {
        "definition": "checkpoint_b - checkpoint_a",
        "win_rate_b": float(wins.double().mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "song_bootstrap_mean_delta_95_ci": [float(interval[0]), float(interval[1])],
        "song_bootstrap_units": len(song_values),
        "speaker_macro_delta": sum(speaker_delta.values()) / len(speaker_delta),
        "speaker_delta": speaker_delta,
        "delta_by_sample": delta.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.panel_size < 256:
        raise ValueError("large comparison panel must contain at least 256 crops")
    device = torch.device(args.device)
    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint_a = torch.load(
        args.checkpoint_a, map_location="cpu", weights_only=False, mmap=True
    )
    dataset = FeatureDataset(
        entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        checkpoint_a["speaker_to_id"],
        voiced_crop_probability=0.0,
    )
    selected = select_panel(
        entries,
        set(config.sampling.synthetic_datasets),
        args.panel_size,
        args.frames,
        args.seed,
    )
    items = [
        dataset[SampleRequest(index, args.frames, 2026 + index)]
        for index, _ in selected
    ]
    tensors = {
        name: torch.stack([item[name] for item in items])
        for name in ("mel", "content", "f0", "rms", "speaker")
    }
    tensors["noise"], tensors["validation_t"] = fixed_noise_and_validation_t(
        tensors["mel"].shape, args.seed
    )
    metadata = [
        {
            "entry_id": entry.id,
            "dataset": entry.dataset,
            "speaker": entry.speaker_key,
            "song": entry.song,
            "song_key": f"{entry.dataset}:{entry.song}",
            "split": entry.split,
            "manifest_index": index,
        }
        for index, entry in selected
    ]
    del checkpoint_a
    models = {
        "a": evaluate(
            args.checkpoint_a, config, stats, tensors, metadata, args, device
        ),
        "b": evaluate(
            args.checkpoint_b, config, stats, tensors, metadata, args, device
        ),
    }
    comparisons = {
        "original_validation_normalized_flow_mse": paired(
            models["a"]["original_validation"]["normalized_flow_mse"],
            models["b"]["original_validation"]["normalized_flow_mse"],
            metadata,
            args.bootstrap_samples,
            args.seed,
        ),
        "velocity_by_t": {},
        "endpoint_by_steps": {},
    }
    for value in TIMES:
        key = str(value)
        comparisons["velocity_by_t"][key] = paired(
            models["a"]["velocity_by_t"][key]["raw_velocity_nmse"],
            models["b"]["velocity_by_t"][key]["raw_velocity_nmse"],
            metadata,
            args.bootstrap_samples,
            args.seed + round(value * 100),
        )
    for steps in EULER_STEPS:
        key = str(steps)
        comparisons["endpoint_by_steps"][key] = {
            metric: paired(
                models["a"]["endpoint_by_steps"][key][metric],
                models["b"]["endpoint_by_steps"][key][metric],
                metadata,
                args.bootstrap_samples,
                args.seed + steps,
                higher_is_better=metric == "raw_cosine",
            )
            for metric in ("raw_mse", "raw_l1", "raw_cosine")
        }
    summaries = {}
    for name, model in models.items():
        summaries[name] = {
            "step": model["step"],
            "original_validation_normalized_flow_mse": summarize(
                model["original_validation"]["normalized_flow_mse"]
            ),
            "velocity_by_t": {
                key: summarize(row["raw_velocity_nmse"])
                for key, row in model["velocity_by_t"].items()
            },
            "endpoint_by_steps": {
                key: {metric: summarize(values) for metric, values in row.items()}
                for key, row in model["endpoint_by_steps"].items()
            },
        }
    payload = {
        "schema_version": 1,
        "protocol": {
            "panel": "real-speaker song-disjoint speaker-balanced round-robin",
            "panel_size": len(metadata),
            "frames": args.frames,
            "checkpoint_state": "ema",
            "speaker_conditioning": "correct physical speaker",
            "seed": args.seed,
            "fixed_t": TIMES,
            "euler_steps": EULER_STEPS,
            "bootstrap_samples": args.bootstrap_samples,
            "synthetic_excluded": sorted(config.sampling.synthetic_datasets),
            "one_crop_per_speaker_song": True,
            "unique_song_units": len({row["song_key"] for row in metadata}),
            "physical_speakers": len({row["speaker"] for row in metadata}),
            "dataset_counts": {
                dataset_name: sum(row["dataset"] == dataset_name for row in metadata)
                for dataset_name in sorted({row["dataset"] for row in metadata})
            },
        },
        "samples": metadata,
        "summaries": summaries,
        "comparisons": comparisons,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summaries": summaries, "comparisons": comparisons}), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
