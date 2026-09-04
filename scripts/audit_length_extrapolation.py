from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system
from scripts.audit_v4_fixed_panels import select_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nested 512/768 endpoint errors with exactly shared noise"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--state", choices=("raw", "ema"), default="ema")
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid checkpoint specification: {value!r}")
        result.append((label, Path(path)))
    if len(result) < 2:
        raise ValueError("at least two checkpoints are required")
    return result


def stack_panel(panel, device: torch.device):
    features = {
        name: torch.stack([item[name] for _, _, item in panel]).to(device)
        for name in ("mel", "content", "f0", "rms")
    }
    return features


def integrate(
    model,
    features,
    noise: torch.Tensor,
    speakers: torch.Tensor,
    intervals: int,
) -> torch.Tensor:
    frames = noise.shape[1]
    mask = torch.ones((noise.shape[0], frames), dtype=torch.bool, device=noise.device)
    generated = noise.clone()
    times = torch.linspace(0.0, 1.0, intervals + 1, device=noise.device)
    with torch.inference_mode(), torch.autocast(
        noise.device.type,
        dtype=torch.bfloat16,
        enabled=noise.device.type == "cuda",
    ):
        for index in range(intervals):
            time = times[index].expand(noise.shape[0])
            generated += (times[index + 1] - times[index]) * model(
                generated,
                features["content"],
                features["f0"],
                features["rms"],
                speakers,
                time,
                mask,
            ).float()
    return generated


def slice_features(features, start: int, stop: int):
    return {name: value[:, start:stop] for name, value in features.items()}


def endpoint_record(prediction, target, stats: MelStats) -> dict[str, object]:
    raw_prediction = stats.denormalize(prediction)
    raw_target = stats.denormalize(target)
    error = raw_prediction.sub(raw_target).square().mean(dim=-1)
    return {
        "mean_raw_mse": float(error.mean()),
        "per_frame_raw_mse": error.mean(dim=0).cpu().tolist(),
        "per_sample_raw_mse": error.mean(dim=1).cpu().tolist(),
        "per_sample_per_frame_raw_mse": error.cpu().tolist(),
        "raw_prediction": raw_prediction,
    }


def add_comparison(payload: dict[str, object], labels: list[str]) -> None:
    first, last = labels[0], labels[-1]
    comparison = {}
    for view in ("full_768", "prefix_512", "suffix_512"):
        before = torch.tensor(payload["results"][first][view]["per_frame_raw_mse"])
        after = torch.tensor(payload["results"][last][view]["per_frame_raw_mse"])
        delta = after - before
        before_samples = torch.tensor(
            payload["results"][first][view]["per_sample_per_frame_raw_mse"]
        )
        after_samples = torch.tensor(
            payload["results"][last][view]["per_sample_per_frame_raw_mse"]
        )
        sample_delta = after_samples - before_samples
        comparison[view] = {
            "delta_last_minus_first_per_frame": delta.tolist(),
            "median_delta_per_frame": sample_delta.median(dim=0).values.tolist(),
            "mean_delta": float(delta.mean()),
            "improved_frame_fraction": float((delta < 0).float().mean()),
            "sample_win_rate": float(
                (sample_delta.mean(dim=1) < 0).float().mean()
            ),
            "median_sample_delta": float(sample_delta.mean(dim=1).median()),
        }
        if view == "full_768":
            comparison[view]["regions"] = {
                "0_512": float(delta[:512].mean()),
                "512_768": float(delta[512:].mean()),
                "0_64": float(delta[:64].mean()),
                "448_512": float(delta[448:512].mean()),
                "512_576": float(delta[512:576].mean()),
                "704_768": float(delta[704:].mean()),
            }
    payload["comparison"] = {
        "first": first,
        "last": last,
        **comparison,
    }


def write_plot(payload: dict[str, object], path: Path, hop: int, sample_rate: int):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(payload["results"])

    def smooth(values, width=16):
        values = np.asarray(values)
        return np.convolve(values, np.ones(width) / width, mode="same")

    seconds_768 = np.arange(768) * hop / sample_rate
    seconds_512 = np.arange(512) * hop / sample_rate
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for label in labels:
        axes[0].plot(
            seconds_768,
            smooth(payload["results"][label]["full_768"]["per_frame_raw_mse"]),
            label=f"{label} full 768",
        )
    axes[0].axvline(512 * hop / sample_rate, color="black", linestyle="--")
    axes[0].set(title="Full-768 endpoint error by position", ylabel="raw mel MSE")
    axes[0].legend()

    first, last = labels[0], labels[-1]
    axes[1].plot(
        seconds_768,
        smooth(payload["comparison"]["full_768"]["delta_last_minus_first_per_frame"]),
        label=f"{last} - {first}, sample mean",
    )
    axes[1].plot(
        seconds_768,
        smooth(payload["comparison"]["full_768"]["median_delta_per_frame"]),
        label=f"{last} - {first}, sample median",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(512 * hop / sample_rate, color="black", linestyle="--")
    axes[1].set(
        title="Paired checkpoint delta (negative means later is better)",
        ylabel="raw mel MSE delta",
    )
    axes[1].legend()

    for label in labels:
        full = payload["results"][label]["full_768"]["per_frame_raw_mse"][:512]
        prefix = payload["results"][label]["prefix_512"]["per_frame_raw_mse"]
        axes[2].plot(
            seconds_512,
            smooth(np.asarray(full) - np.asarray(prefix)),
            label=f"{label}: full768 prefix - independent512",
        )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set(
        title="Effect of integrating the same prefix inside a longer sequence",
        xlabel="time from crop start (s)",
        ylabel="raw mel MSE difference",
    )
    axes[2].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    device = torch.device(args.device)
    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    first_checkpoint = torch.load(
        checkpoints[0][1], map_location="cpu", weights_only=False, mmap=True
    )
    speaker_to_id = first_checkpoint["speaker_to_id"]
    dataset = FeatureDataset(
        entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        speaker_to_id,
        voiced_crop_probability=0.0,
    )
    panel = select_panel(
        entries, dataset, 768, {"validation", "test"}
    )
    features = stack_panel(panel, device)
    speakers = torch.tensor(
        [speaker_to_id[entry.speaker_key] for entry, _, _ in panel],
        dtype=torch.long,
        device=device,
    )
    noise = torch.randn(
        features["mel"].shape,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": {
            "state": args.state,
            "speaker_conditioning": "correct physical speaker",
            "panel": "same 16 song-disjoint crops selected at 768 frames",
            "noise": "one 768-frame tensor; nested views share exact values",
            "solver": f"linear Euler, {args.intervals} intervals",
            "seed": args.seed,
            "views": {
                "full_768": [0, 768],
                "prefix_512": [0, 512],
                "suffix_512": [256, 768],
            },
        },
        "samples": [
            {
                "speaker": entry.speaker_key,
                "song": entry.song,
                "entry_id": entry.id,
                "start_frame": start,
            }
            for entry, start, _ in panel
        ],
        "results": {},
    }
    predictions = {}
    for label, path in checkpoints:
        print(f"loading {label}: {path}", flush=True)
        checkpoint = first_checkpoint if path == checkpoints[0][1] else torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        if checkpoint["speaker_to_id"] != speaker_to_id:
            raise ValueError(f"{label}: speaker mapping differs")
        system.load_state_dict(checkpoint[args.state], strict=True)
        payload["results"][label] = {"checkpoint_step": int(checkpoint["step"])}
        predictions[label] = {}
        for view, (start, stop) in {
            "full_768": (0, 768),
            "prefix_512": (0, 512),
            "suffix_512": (256, 768),
        }.items():
            print(f"integrating {label} {view}", flush=True)
            view_features = slice_features(features, start, stop)
            prediction = integrate(
                system.model,
                view_features,
                noise[:, start:stop],
                speakers,
                args.intervals,
            )
            record = endpoint_record(prediction, view_features["mel"], stats)
            predictions[label][view] = record.pop("raw_prediction")
            payload["results"][label][view] = record
        full_prefix = predictions[label]["full_768"][:, :512]
        independent_prefix = predictions[label]["prefix_512"]
        full_suffix = predictions[label]["full_768"][:, 256:]
        independent_suffix = predictions[label]["suffix_512"]
        payload["results"][label]["context_effect_raw_mse"] = {
            "prefix": float(full_prefix.sub(independent_prefix).square().mean()),
            "suffix": float(full_suffix.sub(independent_suffix).square().mean()),
        }
        if checkpoint is not first_checkpoint:
            del checkpoint
        torch.cuda.empty_cache()

    labels = [label for label, _ in checkpoints]
    add_comparison(payload, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.plot is not None:
        write_plot(payload, args.plot, config.hop_length, config.sample_rate)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
