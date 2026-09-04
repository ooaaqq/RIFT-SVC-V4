from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import V4Config
from .features import MelStats
from .manifest import load_manifest
from .shadow_identity_diagnostic import (
    clustered_mean_interval,
    distribution,
    orthonormal_dct,
)
from .shadow_panel import (
    _atomic_json,
    _load_checkpoint,
    file_sha256,
    load_locked_tensors,
    load_or_create_panel_lock,
    speaker_map_sha256,
)
from .shadow_v3_compare import load_v3
from .train import build_system

BANDS = ((0, 16), (16, 32), (32, 128))
TIMES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit EMA lag, integration, and local velocity spectral detail"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-220-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-230-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoint_220 = _load_checkpoint(args.v4_220_checkpoint, config)
    checkpoint_230 = _load_checkpoint(args.v4_230_checkpoint, config)
    speaker_to_id = checkpoint_220["speaker_to_id"]
    if checkpoint_230["speaker_to_id"] != speaker_to_id:
        raise ValueError("V4 checkpoints have different speaker mappings")
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
    if speaker_map_sha256(speaker_to_id) != lock["source"]["speaker_to_id_sha256"]:
        raise ValueError("speaker mapping differs from shadow lock")
    if args.frames > int(lock["samples"][0]["maximum_frames"]):
        parser.error("requested frames exceed locked crop")
    tensors = load_locked_tensors(lock, entries, config, stats, speaker_to_id)
    maximum_frames = max(int(value) for value in lock["protocol"]["frames"])
    noise = torch.randn(
        len(lock["samples"]),
        maximum_frames,
        config.mel.channels,
        generator=torch.Generator().manual_seed(int(lock["protocol"]["seed"])),
    )[:, : args.frames]
    target_raw = stats.denormalize(tensors["mel"][:, : args.frames].float())
    device = torch.device(args.device)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "panel_lock": str(args.panel_lock),
            "panel_lock_sha256": file_sha256(args.panel_lock),
            "frames": args.frames,
            "noise": "locked 768-frame Gaussian prefix in each native normalized space",
            "bands": [band_name(band) for band in BANDS],
            "times": list(TIMES),
            "bootstrap": "song clustered",
            "bootstrap_samples": args.bootstrap_samples,
            "ema_decay": config.training.ema_decay,
        },
        "checkpoints": {
            "v3": str(args.v3_checkpoint),
            "v4_220": str(args.v4_220_checkpoint),
            "v4_230": str(args.v4_230_checkpoint),
        },
        "coverage": lock["coverage"],
        "endpoints": {},
        "local_velocity": {},
        "comparisons": {},
    }

    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    prediction = integrate(
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
    prediction_raw = (prediction + 1.0) * 7.0 - 12.0
    payload["endpoints"]["v3_null_euler32"] = endpoint_report(
        prediction_raw,
        target_raw,
        tensors,
        lock["samples"],
        args.frames,
        args.bootstrap_samples,
        301,
    )
    payload["local_velocity"]["v3_null"] = local_velocity_report(
        v3,
        "v3",
        tensors,
        noise,
        lock["samples"],
        stats,
        args.frames,
        args.batch_size,
        device,
    )
    _atomic_json(args.output, payload)
    del v3, prediction, prediction_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(speaker_to_id)).to(device).eval()
    endpoint_specs = (
        ("v4_220_raw_euler32", checkpoint_220["model"], 32, "euler"),
        ("v4_220_ema_euler16", checkpoint_220["ema"], 16, "euler"),
        ("v4_220_ema_heun16", checkpoint_220["ema"], 16, "heun"),
        ("v4_220_ema_euler32", checkpoint_220["ema"], 32, "euler"),
        ("v4_220_ema_heun32", checkpoint_220["ema"], 32, "heun"),
        ("v4_220_ema_euler64", checkpoint_220["ema"], 64, "euler"),
        ("v4_230_ema_euler32", checkpoint_230["ema"], 32, "euler"),
    )
    for name, state, steps, method in endpoint_specs:
        system.load_state_dict(state, strict=True)
        prediction = integrate(
            system.model,
            "v4",
            tensors,
            noise,
            args.frames,
            steps,
            method,
            args.batch_size,
            device,
        )
        prediction_raw = stats.denormalize(prediction)
        payload["endpoints"][name] = endpoint_report(
            prediction_raw,
            target_raw,
            tensors,
            lock["samples"],
            args.frames,
            args.bootstrap_samples,
            400 + steps + (100 if method == "heun" else 0),
        )
        _atomic_json(args.output, payload)
        del prediction, prediction_raw

    system.load_state_dict(checkpoint_220["ema"], strict=True)
    payload["local_velocity"]["v4_220_ema_correct"] = local_velocity_report(
        system.model,
        "v4",
        tensors,
        noise,
        lock["samples"],
        stats,
        args.frames,
        args.batch_size,
        device,
    )
    endpoint_pairs = (
        ("v4_220_ema_euler32", "v4_220_raw_euler32"),
        ("v4_220_ema_euler32", "v4_230_ema_euler32"),
        ("v4_220_ema_euler32", "v4_220_ema_euler16"),
        ("v4_220_ema_euler32", "v4_220_ema_heun16"),
        ("v4_220_ema_euler32", "v4_220_ema_heun32"),
        ("v4_220_ema_euler32", "v4_220_ema_euler64"),
        ("v4_220_ema_euler64", "v4_220_ema_heun32"),
    )
    for before, after in endpoint_pairs:
        payload["comparisons"][f"{after}_minus_{before}"] = compare_endpoint_rows(
            payload["endpoints"][before]["samples"],
            payload["endpoints"][after]["samples"],
            lock["samples"],
            args.bootstrap_samples,
            700 + len(payload["comparisons"]),
        )
    payload["comparisons"]["v4_minus_v3_local_velocity"] = compare_local_velocity(
        payload["local_velocity"]["v3_null"],
        payload["local_velocity"]["v4_220_ema_correct"],
        lock["samples"],
        args.bootstrap_samples,
    )
    payload["status"] = "complete"
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


@torch.inference_mode()
def integrate(
    model,
    kind: str,
    tensors: dict[str, torch.Tensor],
    noise: torch.Tensor,
    frames: int,
    steps: int,
    method: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for begin in range(0, len(noise), batch_size):
        end = min(begin + batch_size, len(noise))
        state = noise[begin:end].to(device).clone()
        content = tensors["content"][begin:end, :frames].to(device)
        f0 = tensors["f0"][begin:end, :frames].to(device)
        rms = tensors["rms"][begin:end, :frames].to(device)
        speaker = tensors["speaker"][begin:end].to(device)
        mask = torch.ones(end - begin, frames, dtype=torch.bool, device=device)
        times = torch.linspace(0.0, 1.0, steps + 1, device=device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            for index in range(steps):
                time = times[index].expand(end - begin)
                velocity = predict_velocity(
                    model, kind, state, content, f0, rms, speaker, time, mask
                )
                delta = times[index + 1] - times[index]
                proposal = state + delta * velocity.float()
                if method == "heun" and index + 1 < steps:
                    next_time = times[index + 1].expand(end - begin)
                    next_velocity = predict_velocity(
                        model,
                        kind,
                        proposal,
                        content,
                        f0,
                        rms,
                        speaker,
                        next_time,
                        mask,
                    )
                    state = state + delta * 0.5 * (
                        velocity.float() + next_velocity.float()
                    )
                else:
                    state = proposal
        outputs.append(state.cpu())
        print(
            json.dumps(
                {
                    "stage": "endpoint",
                    "kind": kind,
                    "method": method,
                    "steps": steps,
                    "completed": end,
                }
            ),
            flush=True,
        )
    return torch.cat(outputs)


def predict_velocity(model, kind, state, content, f0, rms, speaker, time, mask):
    if kind == "v3":
        return model(
            x=state,
            spk=torch.zeros_like(speaker),
            f0=f0.squeeze(-1),
            rms=rms.squeeze(-1),
            cvec=content,
            time=time,
            mask=mask,
            drop_speaker=True,
        )
    return model(state, content, f0, rms, speaker, time, mask)


def endpoint_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: list[dict[str, object]],
    frames: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    dct = orthonormal_dct(target.shape[-1])
    active = tensors["rms"][:, :frames, 0] > 1e-3
    voiced = active & (tensors["f0"][:, :frames, 0] > 0)
    rows = []
    for index, item in enumerate(metadata):
        residual_coeff = (prediction[index] - target[index]) @ dct.T
        target_coeff = target[index] @ dct.T
        prediction_coeff = prediction[index] @ dct.T
        row = {
            "ordinal": int(item["ordinal"]),
            "song_key": item["song_key"],
            "bands": {},
        }
        for band in BANDS:
            name = band_name(band)
            row["bands"][name] = {
                region: coefficient_metrics(
                    residual_coeff[:, band[0] : band[1]],
                    target_coeff[:, band[0] : band[1]],
                    prediction_coeff[:, band[0] : band[1]],
                    selected,
                    target.shape[-1],
                )
                for region, selected in (
                    ("active", active[index]),
                    ("voiced", voiced[index]),
                    ("unvoiced", active[index] & ~voiced[index]),
                )
            }
        rows.append(row)
    return {
        "summary": summarize_endpoint_rows(rows, bootstrap_samples, seed),
        "samples": rows,
    }


def coefficient_metrics(
    residual: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    selected: torch.Tensor,
    channels: int,
) -> dict[str, float] | None:
    if not bool(selected.any()):
        return None
    residual = residual[selected]
    target = target[selected]
    prediction = prediction[selected]
    target_energy = target.square().sum()
    prediction_energy = prediction.square().sum()
    target_flat = target.flatten()
    prediction_flat = prediction.flatten()
    return {
        "mse_contribution": float(residual.square().sum() / (len(residual) * channels)),
        "target_rms": float(target.square().mean().sqrt()),
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "rms_ratio": float((prediction_energy / target_energy.clamp_min(1e-12)).sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(target_flat, prediction_flat, dim=0)
        ),
        "correlation": pearson(target_flat, prediction_flat),
    }


def summarize_endpoint_rows(
    rows: list[dict[str, object]], bootstrap_samples: int, seed: int
) -> dict[str, object]:
    result = {}
    for band in BANDS:
        name = band_name(band)
        result[name] = {}
        for region in ("active", "voiced", "unvoiced"):
            result[name][region] = {}
            metrics = [
                "mse_contribution",
                "target_rms",
                "prediction_rms",
                "rms_ratio",
                "cosine",
                "correlation",
            ]
            if any(
                row["bands"][name][region] is not None
                and "nmse" in row["bands"][name][region]
                for row in rows
            ):
                metrics.append("nmse")
            for metric in metrics:
                values = [
                    row["bands"][name][region][metric]
                    for row in rows
                    if row["bands"][name][region] is not None
                ]
                result[name][region][metric] = distribution(values)
    return result


@torch.inference_mode()
def local_velocity_report(
    model,
    kind: str,
    tensors: dict[str, torch.Tensor],
    noise: torch.Tensor,
    metadata: list[dict[str, object]],
    stats: MelStats,
    frames: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    mel = tensors["mel"][:, :frames]
    native_target = (
        target_raw_to_v3(stats.denormalize(mel)) if kind == "v3" else mel
    )
    raw_scale = (
        torch.full((mel.shape[-1],), 7.0)
        if kind == "v3"
        else torch.tensor(stats.std)
    )
    dct = orthonormal_dct(mel.shape[-1])
    active = tensors["rms"][:, :frames, 0] > 1e-3
    voiced = active & (tensors["f0"][:, :frames, 0] > 0)
    result = {}
    for time_value in TIMES:
        rows = []
        for begin in range(0, len(noise), batch_size):
            end = min(begin + batch_size, len(noise))
            native_noise = noise[begin:end]
            state = (
                (1.0 - time_value) * native_noise
                + time_value * native_target[begin:end]
            ).to(device)
            content = tensors["content"][begin:end, :frames].to(device)
            f0 = tensors["f0"][begin:end, :frames].to(device)
            rms = tensors["rms"][begin:end, :frames].to(device)
            speaker = tensors["speaker"][begin:end].to(device)
            mask = torch.ones(end - begin, frames, dtype=torch.bool, device=device)
            time = torch.full((end - begin,), time_value, device=device)
            with torch.autocast(
                device.type, torch.bfloat16, enabled=device.type == "cuda"
            ):
                predicted = predict_velocity(
                    model, kind, state, content, f0, rms, speaker, time, mask
                )
            predicted_raw = predicted.float().cpu() * raw_scale
            target_velocity_raw = (native_target[begin:end] - native_noise) * raw_scale
            error_coeff = (predicted_raw - target_velocity_raw) @ dct.T
            target_coeff = target_velocity_raw @ dct.T
            predicted_coeff = predicted_raw @ dct.T
            for offset in range(end - begin):
                sample = begin + offset
                row = {
                    "ordinal": int(metadata[sample]["ordinal"]),
                    "song_key": metadata[sample]["song_key"],
                    "bands": {},
                }
                for band in BANDS:
                    name = band_name(band)
                    row["bands"][name] = {
                        region: velocity_metrics(
                            error_coeff[offset, :, band[0] : band[1]],
                            target_coeff[offset, :, band[0] : band[1]],
                            predicted_coeff[offset, :, band[0] : band[1]],
                            selected,
                        )
                        for region, selected in (
                            ("active", active[sample]),
                            ("voiced", voiced[sample]),
                            ("unvoiced", active[sample] & ~voiced[sample]),
                        )
                    }
                rows.append(row)
        result[str(time_value)] = {
            "summary": summarize_endpoint_rows(rows, 0, 0),
            "samples": rows,
        }
        print(
            json.dumps(
                {"stage": "local_velocity", "kind": kind, "t": time_value}
            ),
            flush=True,
        )
    return result


def velocity_metrics(error, target, prediction, selected):
    if not bool(selected.any()):
        return None
    error = error[selected]
    target = target[selected]
    prediction = prediction[selected]
    target_energy = target.square().mean()
    return {
        "mse_contribution": float(
            error.square().sum() / (len(error) * BANDS[-1][1])
        ),
        "target_rms": float(target_energy.sqrt()),
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "rms_ratio": float(
            (prediction.square().mean() / target_energy.clamp_min(1e-12)).sqrt()
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                target.flatten(), prediction.flatten(), dim=0
            )
        ),
        "correlation": pearson(target.flatten(), prediction.flatten()),
        "nmse": float(error.square().mean() / target_energy.clamp_min(1e-12)),
    }


def target_raw_to_v3(raw: torch.Tensor) -> torch.Tensor:
    return (raw + 12.0) / 7.0 - 1.0


def compare_endpoint_rows(before, after, metadata, bootstrap_samples, seed):
    result = {}
    for band in BANDS:
        name = band_name(band)
        result[name] = {}
        for region in ("active", "voiced", "unvoiced"):
            result[name][region] = {}
            metrics = ["mse_contribution", "rms_ratio", "cosine", "correlation"]
            if any(
                row["bands"][name][region] is not None
                and "nmse" in row["bands"][name][region]
                for row in before
            ):
                metrics.append("nmse")
            for metric in metrics:
                selected = [
                    (
                        a["bands"][name][region][metric]
                        - b["bands"][name][region][metric],
                        item,
                    )
                    for b, a, item in zip(before, after, metadata, strict=True)
                    if b["bands"][name][region] is not None
                    and a["bands"][name][region] is not None
                ]
                values = [value for value, _ in selected]
                report = distribution(values)
                report["song_bootstrap_mean_95_ci"] = clustered_mean_interval(
                    values, [item for _, item in selected], bootstrap_samples, seed
                )
                result[name][region][metric] = report
    return result


def compare_local_velocity(v3, v4, metadata, bootstrap_samples):
    result = {}
    for time_value in TIMES:
        key = str(time_value)
        result[key] = compare_endpoint_rows(
            v3[key]["samples"],
            v4[key]["samples"],
            metadata,
            bootstrap_samples,
            900 + int(time_value * 100),
        )
    return result


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-12))


def band_name(band: tuple[int, int]) -> str:
    return f"dct_{band[0]}_{band[1] - 1}"


if __name__ == "__main__":
    main()
