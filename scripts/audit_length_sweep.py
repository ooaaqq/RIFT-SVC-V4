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
from scripts.audit_length_extrapolation import (
    endpoint_record,
    integrate,
    parse_checkpoints,
    slice_features,
    stack_panel,
)
from scripts.audit_v4_fixed_panels import select_panel

LENGTHS = (512, 576, 640, 704, 768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep nested endpoint lengths and compare overlap-add inference"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--state", choices=("raw", "ema"), default="ema")
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stitch_overlapping_windows(front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    if front.shape != back.shape or front.shape[1] != 512:
        raise ValueError("stitching requires two Bx512xC predictions")
    output = front.new_empty((front.shape[0], 768, front.shape[2]))
    output[:, :256] = front[:, :256]
    output[:, 512:] = back[:, 256:]
    weight = torch.linspace(0.0, 1.0, 256, device=front.device).view(1, 256, 1)
    output[:, 256:512] = front[:, 256:] * (1.0 - weight) + back[:, :256] * weight
    return output


def summarize_comparison(before: dict, after: dict) -> dict[str, float]:
    first = torch.tensor(before["per_sample_raw_mse"])
    last = torch.tensor(after["per_sample_raw_mse"])
    delta = last - first
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "win_rate": float((delta < 0).float().mean()),
        "mean_relative_change": float(last.mean() / first.mean() - 1.0),
    }


def summarize_stitch(full: dict, stitched: dict) -> dict[str, float]:
    direct = torch.tensor(full["per_sample_raw_mse"])
    overlap = torch.tensor(stitched["per_sample_raw_mse"])
    delta = overlap - direct
    return {
        "stitched_minus_full_mean_delta": float(delta.mean()),
        "stitched_better_rate": float((delta < 0).float().mean()),
        "stitched_to_full_mean_ratio": float(overlap.mean() / direct.mean()),
        "median_sample_delta": float(delta.median()),
    }


def write_plot(payload: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(payload["results"])
    x = np.asarray(LENGTHS)
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    for label in labels:
        means = [
            payload["results"][label][str(length)]["mean_raw_mse"] for length in LENGTHS
        ]
        axes[0].plot(x, means, marker="o", label=f"{label} single pass")
        stitched = payload["results"][label]["stitched_768"]["mean_raw_mse"]
        axes[0].scatter([768], [stitched], marker="X", s=100, label=f"{label} stitched")
    axes[0].set(
        title="Endpoint error versus visible sequence length",
        xlabel="frames",
        ylabel="raw mel MSE",
    )
    axes[0].legend()

    comparisons = payload["checkpoint_comparison"]
    mean_delta = [comparisons[str(length)]["mean_delta"] for length in LENGTHS]
    median_delta = [comparisons[str(length)]["median_delta"] for length in LENGTHS]
    axes[1].plot(x, mean_delta, marker="o", label="mean sample delta")
    axes[1].plot(x, median_delta, marker="o", label="median sample delta")
    stitched = comparisons["stitched_768"]
    axes[1].scatter(
        [768], [stitched["mean_delta"]], marker="X", s=100, label="stitched mean delta"
    )
    axes[1].scatter(
        [768],
        [stitched["median_delta"]],
        marker="X",
        s=100,
        label="stitched median delta",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(
        title=f"{labels[-1]} minus {labels[0]} (negative means later is better)",
        xlabel="frames",
        ylabel="paired raw mel MSE delta",
    )
    axes[1].legend()
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
    panel = select_panel(entries, dataset, 768, {"validation", "test"})
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
    payload = {
        "schema_version": 1,
        "protocol": {
            "state": args.state,
            "lengths": LENGTHS,
            "panel": "same 16 song-disjoint crops selected at 768 frames",
            "noise": "one Bx768xC tensor; every nested view shares exact values",
            "solver": f"linear Euler, {args.intervals} intervals",
            "stitch": "linear crossfade of [0:512] and [256:768] over 256 frames",
            "seed": args.seed,
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
    for label, checkpoint_path in checkpoints:
        print(f"loading {label}: {checkpoint_path}", flush=True)
        checkpoint = (
            first_checkpoint
            if checkpoint_path == checkpoints[0][1]
            else torch.load(
                checkpoint_path, map_location="cpu", weights_only=False, mmap=True
            )
        )
        if checkpoint["speaker_to_id"] != speaker_to_id:
            raise ValueError(f"{label}: speaker mapping differs")
        system.load_state_dict(checkpoint[args.state], strict=True)
        result = {"checkpoint_step": int(checkpoint["step"])}
        predictions = {}
        for length in LENGTHS:
            print(f"integrating {label} prefix_{length}", flush=True)
            view_features = slice_features(features, 0, length)
            prediction = integrate(
                system.model,
                view_features,
                noise[:, :length],
                speakers,
                args.intervals,
            )
            predictions[length] = prediction
            record = endpoint_record(prediction, view_features["mel"], stats)
            record.pop("raw_prediction")
            result[str(length)] = record
        print(f"integrating {label} suffix_512", flush=True)
        suffix_features = slice_features(features, 256, 768)
        suffix = integrate(
            system.model,
            suffix_features,
            noise[:, 256:],
            speakers,
            args.intervals,
        )
        stitched = stitch_overlapping_windows(predictions[512], suffix)
        stitched_record = endpoint_record(stitched, features["mel"], stats)
        stitched_record.pop("raw_prediction")
        result["stitched_768"] = stitched_record
        result["stitch_vs_full"] = summarize_stitch(result["768"], stitched_record)
        payload["results"][label] = result
        if checkpoint is not first_checkpoint:
            del checkpoint
        torch.cuda.empty_cache()

    first, last = (label for label, _ in (checkpoints[0], checkpoints[-1]))
    payload["checkpoint_comparison"] = {
        str(length): summarize_comparison(
            payload["results"][first][str(length)],
            payload["results"][last][str(length)],
        )
        for length in LENGTHS
    }
    payload["checkpoint_comparison"]["stitched_768"] = summarize_comparison(
        payload["results"][first]["stitched_768"],
        payload["results"][last]["stitched_768"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.plot is not None:
        write_plot(payload, args.plot)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
