from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.evaluate import _reference_crop
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.third_party import PCNSFLock
from rift_v4.train import build_system
from rift_v4.vocoder import (
    _write_waveform,
    load_pc_nsf_generator,
    synthesize_pc_nsf_tensors,
)
from scripts.audit_length_extrapolation import (
    integrate,
    parse_checkpoints,
    stack_panel,
)
from scripts.audit_v4_fixed_panels import select_panel

AUDIT_SONGS = ("美错", "红玫瑰", "送别")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit waveform leakage and ContentVec conditioning in silence"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-nsf-checkout", type=Path, required=True)
    parser.add_argument("--pc-nsf-lock", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path, required=True)
    parser.add_argument("--state", choices=("raw", "ema"), default="ema")
    parser.add_argument("--frames", type=int, default=768)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--silence-rms", type=float, default=1e-3)
    parser.add_argument("--interior-margin-ms", type=float, default=150.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frame_rms(waveform: torch.Tensor, config: V4Config, frames: int) -> torch.Tensor:
    waveform = waveform.float().flatten()
    wanted = frames * config.hop_length
    waveform = F.pad(waveform[:wanted], (0, max(0, wanted - waveform.numel())))
    padding = (config.mel.win_length - config.hop_length) // 2
    padded = F.pad(waveform[None, None], (padding, padding), mode="reflect")[0, 0]
    result = padded.unfold(0, config.mel.win_length, config.hop_length)
    return result[:frames].square().mean(dim=-1).sqrt()


def erode_mask(mask: torch.Tensor, margin: int) -> torch.Tensor:
    if margin <= 0:
        return mask
    width = 2 * margin + 1
    count = F.conv1d(
        mask.float()[None, None],
        torch.ones(1, 1, width),
        padding=margin,
    )[0, 0]
    return count == width


def rms_dbfs(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    selected = values[mask]
    if not selected.numel():
        return {
            "frames": 0,
            "rms_dbfs": None,
            "median_frame_dbfs": None,
            "p95_frame_dbfs": None,
        }
    frame_db = 20.0 * selected.clamp_min(1e-12).log10()
    return {
        "frames": int(selected.numel()),
        "rms_dbfs": float(
            20.0 * selected.square().mean().sqrt().clamp_min(1e-12).log10()
        ),
        "median_frame_dbfs": float(frame_db.median()),
        "p95_frame_dbfs": float(torch.quantile(frame_db, 0.95)),
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def content_diagnostics(
    content: torch.Tensor, silence_centroid: torch.Tensor
) -> dict[str, list[float]]:
    normalized = F.normalize(content.float(), dim=-1)
    centroid = F.normalize(silence_centroid.float(), dim=-1)
    adjacent = F.cosine_similarity(normalized[1:], normalized[:-1], dim=-1)
    adjacent = torch.cat((adjacent.new_tensor([1.0]), adjacent))
    return {
        "norm": content.float().norm(dim=-1).cpu().tolist(),
        "adjacent_cosine": adjacent.cpu().tolist(),
        "silence_centroid_cosine": (normalized @ centroid).cpu().tolist(),
    }


def plot_sample(
    path: Path,
    config: V4Config,
    target_f0: torch.Tensor,
    target_mel: torch.Tensor,
    target_db: torch.Tensor,
    resynth_db: torch.Tensor,
    generated_db: dict[str, torch.Tensor],
    generated_mel: torch.Tensor,
    diagnostics: dict[str, list[float]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    seconds = np.arange(target_f0.numel()) * config.hop_length / config.sample_rate
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    axes[0].plot(seconds, target_db, label="target waveform")
    axes[0].plot(seconds, resynth_db, label="target mel -> PC-NSF")
    for name, values in generated_db.items():
        axes[0].plot(seconds, values, label=name)
    axes[0].axhline(-60.0, color="black", linestyle="--")
    axes[0].set(title="Waveform frame RMS", ylabel="dBFS", ylim=(-120, 0))
    axes[0].legend()

    axes[1].plot(seconds, diagnostics["norm"], label="ContentVec norm")
    axes[1].plot(seconds, diagnostics["adjacent_cosine"], label="adjacent cosine")
    axes[1].plot(
        seconds,
        diagnostics["silence_centroid_cosine"],
        label="cosine to silence centroid",
    )
    axes[1].plot(
        seconds, target_f0 / max(float(target_f0.max()), 1.0), label="F0 normalized"
    )
    axes[1].set(title="Silence conditioning", ylabel="value")
    axes[1].legend()

    axes[2].plot(seconds, target_mel.mean(dim=-1), label="target mean log-mel")
    axes[2].plot(seconds, generated_mel.mean(dim=-1), label="generated mean log-mel")
    axes[2].set(
        title="Mel output energy proxy", xlabel="seconds", ylabel="mean log-mel"
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
    full_panel = select_panel(entries, dataset, args.frames, {"validation", "test"})
    selected = [item for item in full_panel if item[0].song in AUDIT_SONGS]
    if len(selected) != len(AUDIT_SONGS):
        raise ValueError("the fixed panel does not contain every requested audit song")
    full_features = stack_panel(full_panel, torch.device("cpu"))
    quiet = full_features["rms"].squeeze(-1) <= args.silence_rms
    if not quiet.any():
        raise ValueError("fixed panel contains no silence frames")
    silence_centroid = full_features["content"][quiet].mean(dim=0)
    features = stack_panel(selected, device)
    speakers = torch.tensor(
        [speaker_to_id[entry.speaker_key] for entry, _, _ in selected],
        dtype=torch.long,
        device=device,
    )
    noise = torch.randn(
        features["mel"].shape,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)
    lock = PCNSFLock.load(args.pc_nsf_lock)
    lock.validate_contract(config)
    lock.verify_checkout(args.pc_nsf_checkout)
    lock.verify_installed_checkpoint(args.vocoder_checkpoint)
    vocoder = load_pc_nsf_generator(
        args.pc_nsf_checkout, args.vocoder_checkpoint, device, config
    )
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    margin = math.ceil(
        args.interior_margin_ms * config.sample_rate / 1000 / config.hop_length
    )
    sample_static = []
    for index, (entry, start, _) in enumerate(selected):
        target_mel = stats.denormalize(features["mel"][index].cpu())
        target_f0 = features["f0"][index].cpu().squeeze(-1)
        reference = _reference_crop(entry, config, start, args.frames)
        target_resynth = synthesize_pc_nsf_tensors(
            vocoder, target_mel, target_f0, device, config
        )
        target_frame_rms = frame_rms(reference, config, args.frames)
        resynth_frame_rms = frame_rms(target_resynth, config, args.frames)
        silence = target_frame_rms <= args.silence_rms
        interior = erode_mask(silence, margin)
        prefix = f"{index:02d}-{safe_name(entry.speaker_key)}-{safe_name(entry.song)}"
        _write_waveform(
            output_dir / f"{prefix}-target.wav", reference, config.sample_rate
        )
        _write_waveform(
            output_dir / f"{prefix}-target-resynth.wav",
            target_resynth,
            config.sample_rate,
        )
        sample_static.append(
            {
                "entry": entry,
                "start": start,
                "prefix": prefix,
                "target_mel": target_mel,
                "target_f0": target_f0,
                "reference": reference,
                "target_resynth": target_resynth,
                "target_frame_rms": target_frame_rms,
                "resynth_frame_rms": resynth_frame_rms,
                "silence": silence,
                "interior": interior,
                "content": features["content"][index].cpu(),
            }
        )

    payload = {
        "schema_version": 1,
        "protocol": {
            "state": args.state,
            "frames": args.frames,
            "solver": f"linear Euler, {args.intervals} intervals",
            "silence_rms": args.silence_rms,
            "silence_dbfs": 20 * math.log10(args.silence_rms),
            "interior_margin_ms": args.interior_margin_ms,
            "silence_centroid_frames": int(quiet.sum()),
            "conditions": ["baseline", "silence_centroid", "silence_zero"],
        },
        "results": {},
    }
    centroid = silence_centroid.to(device)
    for label, checkpoint_path in checkpoints:
        checkpoint = (
            first_checkpoint
            if checkpoint_path == checkpoints[0][1]
            else torch.load(
                checkpoint_path, map_location="cpu", weights_only=False, mmap=True
            )
        )
        system.load_state_dict(checkpoint[args.state], strict=True)
        conditions = {"baseline": features["content"]}
        quiet_mask = features["rms"].squeeze(-1) <= args.silence_rms
        centroid_content = features["content"].clone()
        centroid_content[quiet_mask] = centroid
        zero_content = features["content"].clone()
        zero_content[quiet_mask] = 0
        conditions["silence_centroid"] = centroid_content
        conditions["silence_zero"] = zero_content
        generated_by_condition = {}
        for condition, content in conditions.items():
            print(f"integrating {label} {condition}", flush=True)
            view_features = {**features, "content": content}
            generated_by_condition[condition] = integrate(
                system.model,
                view_features,
                noise,
                speakers,
                args.intervals,
            )
        label_results = []
        for index, static in enumerate(sample_static):
            masks = {
                "silence": static["silence"],
                "interior_silence": static["interior"],
            }
            target_metrics = {
                name: rms_dbfs(static["target_frame_rms"], mask)
                for name, mask in masks.items()
            }
            resynth_metrics = {
                name: rms_dbfs(static["resynth_frame_rms"], mask)
                for name, mask in masks.items()
            }
            condition_results = {}
            condition_frame_db = {}
            baseline_mel = None
            for condition, generated in generated_by_condition.items():
                raw_mel = stats.denormalize(generated[index].cpu())
                waveform = synthesize_pc_nsf_tensors(
                    vocoder, raw_mel, static["target_f0"], device, config
                )
                waveform_rms = frame_rms(waveform, config, args.frames)
                metrics = {
                    name: rms_dbfs(waveform_rms, mask) for name, mask in masks.items()
                }
                for name in masks:
                    generated_db = metrics[name]["rms_dbfs"]
                    resynth_db = resynth_metrics[name]["rms_dbfs"]
                    metrics[name]["excess_over_target_resynth_db"] = (
                        generated_db - resynth_db
                        if generated_db is not None and resynth_db is not None
                        else None
                    )
                condition_results[condition] = metrics
                condition_frame_db[condition] = (
                    20.0 * waveform_rms.clamp_min(1e-12).log10()
                )
                _write_waveform(
                    output_dir / f"{static['prefix']}-{label}-{condition}.wav",
                    waveform,
                    config.sample_rate,
                )
                if condition == "baseline":
                    baseline_mel = raw_mel
            diagnostics = content_diagnostics(static["content"], silence_centroid)
            if label == checkpoints[-1][0]:
                plot_sample(
                    output_dir / f"{static['prefix']}-{label}.png",
                    config,
                    static["target_f0"],
                    static["target_mel"],
                    20.0 * static["target_frame_rms"].clamp_min(1e-12).log10(),
                    20.0 * static["resynth_frame_rms"].clamp_min(1e-12).log10(),
                    condition_frame_db,
                    baseline_mel,
                    diagnostics,
                )
            label_results.append(
                {
                    "speaker": static["entry"].speaker_key,
                    "song": static["entry"].song,
                    "entry_id": static["entry"].id,
                    "start_frame": static["start"],
                    "target": target_metrics,
                    "target_resynth": resynth_metrics,
                    "generated": condition_results,
                    "content": diagnostics,
                }
            )
        payload["results"][label] = {
            "checkpoint_step": int(checkpoint["step"]),
            "samples": label_results,
        }
        if checkpoint is not first_checkpoint:
            del checkpoint
        torch.cuda.empty_cache()
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
