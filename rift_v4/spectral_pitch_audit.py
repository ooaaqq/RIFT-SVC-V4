from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import V4Config
from .features import MelStats
from .manifest import load_manifest
from .shadow_identity_diagnostic import clustered_mean_interval, distribution
from .shadow_panel import (
    _atomic_json,
    _load_checkpoint,
    file_sha256,
    load_locked_tensors,
    load_or_create_panel_lock,
)
from .shadow_v3_compare import load_v3
from .spectral_detail_audit import coefficient_metrics, integrate, orthonormal_dct
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratify V3/V4 fine spectral error by pitch behavior"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint = _load_checkpoint(args.v4_checkpoint, config)
    speaker_to_id = checkpoint["speaker_to_id"]
    raw_lock = json.loads(args.panel_lock.read_text(encoding="utf-8"))
    lock = load_or_create_panel_lock(
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
    tensors = load_locked_tensors(lock, entries, config, stats, speaker_to_id)
    maximum_frames = max(int(value) for value in lock["protocol"]["frames"])
    noise = torch.randn(
        len(lock["samples"]),
        maximum_frames,
        config.mel.channels,
        generator=torch.Generator().manual_seed(int(lock["protocol"]["seed"])),
    )[:, : args.frames]
    target = stats.denormalize(tensors["mel"][:, : args.frames].float())
    device = torch.device(args.device)

    predictions = {}
    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    predictions["v3_null"] = (
        integrate(
            v3,
            "v3",
            tensors,
            noise,
            args.frames,
            32,
            "euler",
            args.batch_size,
            device,
        )
        + 1.0
    ) * 7.0 - 12.0
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(speaker_to_id)).to(device).eval()
    system.load_state_dict(checkpoint["ema"], strict=True)
    predictions["v4_220_ema_correct"] = stats.denormalize(
        integrate(
            system.model,
            "v4",
            tensors,
            noise,
            args.frames,
            32,
            "euler",
            args.batch_size,
            device,
        )
    )
    strata, thresholds = build_strata(target, tensors, args.frames)
    reports = {
        name: stratified_report(prediction, target, strata, lock["samples"])
        for name, prediction in predictions.items()
    }
    payload = {
        "schema_version": 1,
        "protocol": {
            "panel_lock": str(args.panel_lock),
            "panel_lock_sha256": file_sha256(args.panel_lock),
            "frames": args.frames,
            "solver": "Euler32",
            "dct_band": "32-127",
            "thresholds": thresholds,
            "motion_units": "absolute semitones per 11.61 ms frame",
            "fine_energy": "target raw-log-mel DCT32-127 RMS proxy, not HNR",
        },
        "models": reports,
        "v4_minus_v3": compare_reports(
            reports["v3_null"],
            reports["v4_220_ema_correct"],
            lock["samples"],
            args.bootstrap_samples,
        ),
    }
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def build_strata(target, tensors, frames):
    f0 = tensors["f0"][:, :frames, 0]
    active = tensors["rms"][:, :frames, 0] > 1e-3
    voiced = active & (f0 > 0)
    f0_edges = torch.quantile(f0[voiced], torch.tensor([0.25, 0.5, 0.75]))
    log_f0 = torch.where(voiced, f0.clamp_min(1).log2(), torch.zeros_like(f0))
    motion = torch.full_like(f0, float("nan"))
    adjacent = voiced[:, 1:] & voiced[:, :-1]
    delta = 12.0 * (log_f0[:, 1:] - log_f0[:, :-1]).abs()
    motion[:, 1:] = torch.where(adjacent, delta, torch.full_like(delta, float("nan")))
    dct = orthonormal_dct(target.shape[-1])
    fine_rms = (target @ dct.T)[:, :, 32:].square().mean(-1).sqrt()
    fine_edges = torch.quantile(fine_rms[voiced], torch.tensor([0.25, 0.5, 0.75]))
    strata = {
        "f0": quantile_masks(f0, voiced, f0_edges),
        "pitch_motion": {
            "stable_lt_0.1": voiced & (motion < 0.1),
            "moderate_0.1_0.5": voiced & (motion >= 0.1) & (motion < 0.5),
            "fast_ge_0.5": voiced & (motion >= 0.5),
        },
        "target_fine_energy": quantile_masks(fine_rms, voiced, fine_edges),
    }
    thresholds = {
        "f0_hz_quartiles": [float(value) for value in f0_edges],
        "target_fine_rms_quartiles": [float(value) for value in fine_edges],
        "pitch_motion": [0.1, 0.5],
    }
    return strata, thresholds


def quantile_masks(values, selected, edges):
    return {
        "q1": selected & (values <= edges[0]),
        "q2": selected & (values > edges[0]) & (values <= edges[1]),
        "q3": selected & (values > edges[1]) & (values <= edges[2]),
        "q4": selected & (values > edges[2]),
    }


def stratified_report(prediction, target, strata, metadata):
    dct = orthonormal_dct(target.shape[-1])
    residual = (prediction - target) @ dct.T
    target_coeff = target @ dct.T
    prediction_coeff = prediction @ dct.T
    rows = []
    for sample, item in enumerate(metadata):
        groups = {}
        for family, masks in strata.items():
            groups[family] = {
                name: coefficient_metrics(
                    residual[sample, :, 32:],
                    target_coeff[sample, :, 32:],
                    prediction_coeff[sample, :, 32:],
                    mask[sample],
                    target.shape[-1],
                )
                for name, mask in masks.items()
            }
        rows.append(
            {
                "ordinal": int(item["ordinal"]),
                "song_key": item["song_key"],
                "groups": groups,
            }
        )
    return {"summary": summarize_rows(rows), "samples": rows}


def summarize_rows(rows):
    result = {}
    for family in rows[0]["groups"]:
        result[family] = {}
        for group in rows[0]["groups"][family]:
            result[family][group] = {}
            for metric in ("mse_contribution", "rms_ratio", "correlation"):
                values = [
                    row["groups"][family][group][metric]
                    for row in rows
                    if row["groups"][family][group] is not None
                ]
                result[family][group][metric] = distribution(values)
    return result


def compare_reports(before, after, metadata, bootstrap_samples):
    result = {}
    for family in before["summary"]:
        result[family] = {}
        for group in before["summary"][family]:
            result[family][group] = {}
            for metric in ("mse_contribution", "rms_ratio", "correlation"):
                selected = [
                    (
                        right["groups"][family][group][metric]
                        - left["groups"][family][group][metric],
                        item,
                    )
                    for left, right, item in zip(
                        before["samples"], after["samples"], metadata, strict=True
                    )
                    if left["groups"][family][group] is not None
                    and right["groups"][family][group] is not None
                ]
                values = [value for value, _ in selected]
                report = distribution(values)
                report["song_bootstrap_mean_95_ci"] = clustered_mean_interval(
                    values,
                    [item for _, item in selected],
                    bootstrap_samples,
                    1200 + len(values),
                )
                result[family][group][metric] = report
    return result


if __name__ == "__main__":
    main()
