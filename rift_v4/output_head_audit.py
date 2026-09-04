from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import V4Config
from .features import MelStats
from .manifest import load_manifest
from .shadow_identity_diagnostic import distribution, orthonormal_dct
from .shadow_panel import _atomic_json, _load_checkpoint, load_locked_tensors
from .shadow_v3_compare import load_v3
from .train import _build_optimizer, build_system

BANDS = ((0, 16), (16, 32), (32, 128))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit output-head DCT weights, gradients, and Adam moments"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--train-lock", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    checkpoints = [
        _load_checkpoint(path, config) for path in args.v4_checkpoint
    ]
    mapping = checkpoints[0]["speaker_to_id"]
    if any(checkpoint["speaker_to_id"] != mapping for checkpoint in checkpoints[1:]):
        raise ValueError("V4 checkpoints have different speaker mappings")
    lock = json.loads(args.train_lock.read_text(encoding="utf-8"))
    tensors = load_locked_tensors(lock, entries, config, stats, mapping)
    frames = int(lock["protocol"]["frames"][0])
    noise = torch.randn(
        len(lock["samples"]),
        frames,
        config.mel.channels,
        generator=torch.Generator().manual_seed(args.seed),
    )
    timesteps = torch.rand(
        len(lock["samples"]), generator=torch.Generator().manual_seed(args.seed + 1)
    )
    scale_v3 = torch.full((config.mel.channels,), 7.0)
    scale_v4 = torch.tensor(stats.std)
    device = torch.device(args.device)
    payload = {
        "schema_version": 1,
        "protocol": {
            "train_lock": str(args.train_lock),
            "samples": len(lock["samples"]),
            "frames": frames,
            "noise_seed": args.seed,
            "timestep_seed": args.seed + 1,
            "bands": [band_name(band) for band in BANDS],
            "weight_basis": "raw-log-mel output delta, DCT along mel rows",
            "gradient": "native training loss gradient mapped to raw output delta",
            "second_moment": (
                "DCT projection under diagonal parameter-coordinate covariance; "
                "cross-row covariance unavailable"
            ),
        },
        "models": {},
    }

    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    v3_gradient, v3_loss = output_gradient(
        v3,
        "v3",
        tensors,
        noise,
        timesteps,
        stats,
        args.batch_size,
        device,
    )
    payload["models"]["v3_300k"] = {
        "weight": projected_matrix_report(v3.output.weight.detach().cpu(), scale_v3),
        "gradient": projected_matrix_report(v3_gradient, scale_v3),
        "native_flow_loss": v3_loss,
        "optimizer": None,
        "optimizer_note": "official checkpoint does not contain optimizer state",
    }
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(mapping)).to(device).eval()
    for checkpoint, path in zip(checkpoints, args.v4_checkpoint, strict=True):
        step = int(checkpoint["step"])
        for state_name in ("model", "ema"):
            system.load_state_dict(checkpoint[state_name], strict=True)
            gradient, loss = output_gradient(
                system.model,
                "v4",
                tensors,
                noise,
                timesteps,
                stats,
                args.batch_size,
                device,
            )
            name = f"v4_{step}_{'raw' if state_name == 'model' else 'ema'}"
            row = {
                "checkpoint": str(path),
                "weight": projected_matrix_report(
                    system.model.output.weight.detach().cpu(), scale_v4
                ),
                "gradient": projected_matrix_report(gradient, scale_v4),
                "native_flow_loss": loss,
            }
            if state_name == "model":
                row["optimizer"] = optimizer_report(
                    system,
                    checkpoint,
                    config,
                    scale_v4,
                )
            else:
                row["optimizer"] = None
            payload["models"][name] = row
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def output_gradient(
    model,
    kind: str,
    tensors: dict[str, torch.Tensor],
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    stats: MelStats,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.output.weight.requires_grad_(True)
    model.output.weight.grad = None
    losses = []
    batches = (len(noise) + batch_size - 1) // batch_size
    for begin in range(0, len(noise), batch_size):
        end = min(begin + batch_size, len(noise))
        mel_v4 = tensors["mel"][begin:end].to(device)
        native_mel = (
            (stats.denormalize(mel_v4) + 12.0) / 7.0 - 1.0
            if kind == "v3"
            else mel_v4
        )
        native_noise = noise[begin:end].to(device)
        time = timesteps[begin:end].to(device)
        expanded = time[:, None, None]
        state = (1.0 - expanded) * native_noise + expanded * native_mel
        target_velocity = native_mel - native_noise
        content = tensors["content"][begin:end].to(device)
        f0 = tensors["f0"][begin:end].to(device)
        rms = tensors["rms"][begin:end].to(device)
        speaker = tensors["speaker"][begin:end].to(device)
        mask = torch.ones(
            end - begin, native_mel.shape[1], dtype=torch.bool, device=device
        )
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            if kind == "v3":
                prediction = model(
                    x=state,
                    spk=torch.zeros_like(speaker),
                    f0=f0.squeeze(-1),
                    rms=rms.squeeze(-1),
                    cvec=content,
                    time=time,
                    mask=mask,
                    drop_speaker=True,
                )
            else:
                prediction = model(state, content, f0, rms, speaker, time, mask)
            loss = (prediction.float() - target_velocity).square().mean()
        (loss / batches).backward()
        losses.append(float(loss.detach()))
    gradient = model.output.weight.grad.detach().float().cpu()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    return gradient, sum(losses) / len(losses)


def projected_matrix_report(matrix: torch.Tensor, raw_scale: torch.Tensor):
    dct = orthonormal_dct(matrix.shape[0])
    projected = dct @ (raw_scale[:, None] * matrix.float())
    row_rms = projected.square().mean(-1).sqrt()
    return band_report(row_rms)


def optimizer_report(system, checkpoint, config, raw_scale):
    if "optimizer" not in checkpoint:
        return None
    _, summary = _build_optimizer(system, config)
    names = summary["groups"]["backbone_decay"]["parameter_names"]
    position = names.index("model.output.weight")
    saved_group = next(
        group
        for group in checkpoint["optimizer"]["param_groups"]
        if group.get("name") == "backbone_decay"
    )
    parameter_id = saved_group["params"][position]
    state = checkpoint["optimizer"]["state"][parameter_id]
    first = state["exp_avg"].float()
    second = state["exp_avg_sq"].float()
    dct = orthonormal_dct(first.shape[0])
    effective = first / (second.sqrt() + float(saved_group.get("eps", 1e-8)))
    effective_raw = dct @ (raw_scale[:, None] * effective)
    directional_second = (dct.square()) @ (
        raw_scale[:, None].square() * second
    )
    return {
        "step": int(state["step"]),
        "exp_avg_sq_directional_rms": band_report(
            directional_second.mean(-1).sqrt()
        ),
        "preconditioned_update_rms": band_report(
            effective_raw.square().mean(-1).sqrt()
        ),
    }


def band_report(values: torch.Tensor):
    result = {}
    low_mean = float(values[BANDS[0][0] : BANDS[0][1]].mean())
    for band in BANDS:
        selected = values[band[0] : band[1]]
        row = distribution([float(value) for value in selected])
        row["relative_to_dct_0_15_mean"] = float(selected.mean()) / low_mean
        result[band_name(band)] = row
    return result


def band_name(band):
    return f"dct_{band[0]}_{band[1] - 1}"


if __name__ == "__main__":
    main()
