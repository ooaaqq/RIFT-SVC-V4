from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .config import V4Config
from .data import FeatureDataset, SampleRequest, collate_features
from .features import MelStats
from .manifest import ManifestEntry, load_manifest
from .model import _modulate, _rotary
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit RF, attention, and AdaLN behavior in a V4 checkpoint"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--samples-per-dataset", type=int, default=8)
    parser.add_argument("--structure-samples-per-dataset", type=int, default=1)
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.05 + 0.1 * index for index in range(10)],
    )
    parser.add_argument(
        "--structure-t-values", type=float, nargs="+", default=[0.1, 0.5, 0.9]
    )
    args = parser.parse_args()
    payload = audit_checkpoint(
        args.config,
        args.manifest,
        args.mel_stats,
        args.checkpoint,
        torch.device(args.device),
        args.frames,
        args.samples_per_dataset,
        args.structure_samples_per_dataset,
        args.t_values,
        args.structure_t_values,
    )
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))


@torch.inference_mode()
def audit_checkpoint(
    config_path: Path,
    manifest_path: Path,
    mel_stats_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    frames: int,
    samples_per_dataset: int,
    structure_samples_per_dataset: int,
    t_values: list[float],
    structure_t_values: list[float],
) -> dict[str, object]:
    if frames <= 1 or min(samples_per_dataset, structure_samples_per_dataset) <= 0:
        raise ValueError("audit panel sizes must be positive")
    _validate_t_values(t_values)
    _validate_t_values(structure_t_values)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = V4Config.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("checkpoint does not use schema 4")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint configuration differs from audit configuration")

    entries = load_manifest(manifest_path)
    speaker_to_id = checkpoint["speaker_to_id"]
    validation_entries = [
        entry
        for entry in entries
        if entry.split == "validation"
        and entry.quality_status == "accepted"
        and entry.speaker_key in speaker_to_id
    ]
    if not validation_entries:
        raise ValueError("checkpoint audit requires accepted validation entries")
    dataset = FeatureDataset(
        validation_entries,
        config.mel.channels,
        config.model.content_dim,
        MelStats.load(mel_stats_path, config.mel.channels),
        speaker_to_id,
        voiced_crop_probability=config.sampling.voiced_crop_probability,
    )
    rf_indices = _balanced_panel_indices(validation_entries, samples_per_dataset)
    structure_indices = _balanced_panel_indices(
        validation_entries, structure_samples_per_dataset
    )
    rf_batch = _load_panel(dataset, rf_indices, frames, 31_000, device)
    structure_batch = _load_panel(dataset, structure_indices, frames, 41_000, device)
    rf_noise = _fixed_noise(rf_batch["mel"], device, 51_000)
    structure_noise = _fixed_noise(structure_batch["mel"], device, 61_000)

    system = build_system(config, len(speaker_to_id)).to(device).eval()
    use_bf16 = config.training.precision == "bf16" and device.type == "cuda"
    rf_by_state: dict[str, list[dict[str, float]]] = {}
    for label, state_key in (("online", "model"), ("ema", "ema")):
        system.load_state_dict(checkpoint[state_key], strict=True)
        rf_by_state[label] = _rf_t_grid(
            system.model, rf_batch, rf_noise, t_values, use_bf16
        )

    system.load_state_dict(checkpoint["model"], strict=True)
    structure = [
        _structure_at_t(system.model, structure_batch, structure_noise, value, use_bf16)
        for value in structure_t_values
    ]
    return {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_step": int(checkpoint["step"]),
        "frames": frames,
        "rf_panel": _panel_descriptor(validation_entries, rf_indices),
        "structure_panel": _panel_descriptor(validation_entries, structure_indices),
        "rf_by_t": rf_by_state,
        "online_structure_by_t": structure,
    }


def _validate_t_values(values: list[float]) -> None:
    if not values or any(
        not math.isfinite(value) or not 0 < value < 1 for value in values
    ):
        raise ValueError("audit timesteps must be finite and inside (0, 1)")


def _balanced_panel_indices(
    entries: list[ManifestEntry], samples_per_dataset: int
) -> list[int]:
    grouped: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(entry.dataset, []).append(index)
    selected: list[int] = []
    for dataset_name in sorted(grouped):
        candidates = sorted(grouped[dataset_name], key=lambda index: entries[index].id)
        count = min(samples_per_dataset, len(candidates))
        positions = [
            ((2 * offset + 1) * len(candidates)) // (2 * count)
            for offset in range(count)
        ]
        selected.extend(candidates[position] for position in positions)
    return selected


def _load_panel(
    dataset: FeatureDataset,
    indices: list[int],
    frames: int,
    seed: int,
    device: torch.device,
) -> dict[str, Tensor]:
    items = [
        dataset[SampleRequest(index, frames, seed + offset)]
        for offset, index in enumerate(indices)
    ]
    return {name: value.to(device) for name, value in collate_features(items).items()}


def _fixed_noise(mel: Tensor, device: torch.device, seed: int) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(mel.shape, device=device, dtype=mel.dtype, generator=generator)


def _panel_descriptor(
    entries: list[ManifestEntry], indices: list[int]
) -> list[dict[str, str]]:
    return [
        {
            "id": entries[index].id,
            "dataset": entries[index].dataset,
            "speaker": entries[index].speaker,
            "song": entries[index].song,
        }
        for index in indices
    ]


@torch.inference_mode()
def _rf_t_grid(
    model,
    batch: dict[str, Tensor],
    noise: Tensor,
    t_values: list[float],
    use_bf16: bool,
) -> list[dict[str, float]]:
    mel = batch["mel"]
    mask = batch["mask"]
    target = mel - noise
    results = []
    for value in t_values:
        timestep = torch.full(
            (mel.shape[0],), value, device=mel.device, dtype=mel.dtype
        )
        noisy = (1 - value) * noise + value * mel
        with torch.autocast(
            device_type=mel.device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            prediction = model(
                noisy,
                batch["content"],
                batch["f0"],
                batch["rms"],
                batch["speaker"],
                timestep,
                mask,
            )
        metrics = _velocity_metrics(prediction.float(), target.float(), mask)
        results.append({"t": value, **metrics})
    return results


def _velocity_metrics(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> dict[str, float]:
    numeric_mask = mask.unsqueeze(-1).to(prediction.dtype)
    dimensions = (
        mask.sum(dim=1).clamp_min(1).to(prediction.dtype) * prediction.shape[-1]
    )
    error = ((prediction - target).square() * numeric_mask).sum((1, 2)) / dimensions
    dot = (prediction * target * numeric_mask).sum((1, 2))
    prediction_norm = (prediction.square() * numeric_mask).sum((1, 2)).sqrt()
    target_norm = (target.square() * numeric_mask).sum((1, 2)).sqrt()
    cosine = dot / (prediction_norm * target_norm).clamp_min(1e-12)
    prediction_rms = prediction_norm / dimensions.sqrt()
    target_rms = target_norm / dimensions.sqrt()
    return {
        "mse": float(error.mean()),
        "cosine": float(cosine.mean()),
        "prediction_rms": float(prediction_rms.mean()),
        "target_rms": float(target_rms.mean()),
    }


@torch.inference_mode()
def _structure_at_t(
    model,
    batch: dict[str, Tensor],
    noise: Tensor,
    t_value: float,
    use_bf16: bool,
) -> dict[str, object]:
    mel = batch["mel"]
    mask = batch["mask"]
    timestep = torch.full((mel.shape[0],), t_value, device=mel.device, dtype=mel.dtype)
    noisy = (1 - t_value) * noise + t_value * mel
    with torch.autocast(
        device_type=mel.device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        voiced = (batch["f0"] > 0).to(batch["f0"].dtype)
        log_f0 = torch.where(voiced.bool(), batch["f0"].clamp_min(1).log2() / 10.0, 0.0)
        x = (
            model.mel_input(noisy)
            + model.content_input(batch["content"])
            + model.pitch_input(torch.cat((log_f0, voiced), dim=-1))
            + model.rms_input(batch["rms"])
        )
        conditioning = (model.time(timestep) + model.speaker(batch["speaker"]))[
            :, None, :
        ]
        layers = []
        for index, block in enumerate(model.blocks):
            x = x * mask.unsqueeze(-1).to(x.dtype)
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = block.modulation(
                conditioning
            ).chunk(6, dim=-1)
            attention_input = _modulate(block.norm1(x), shift_a, scale_a)
            attention_metrics = _attention_metrics(
                block.attention, attention_input, mask
            )
            attended = block.attention(attention_input, mask)
            attention_branch_rms = _valid_rms(attended, mask)
            attention_residual = gate_a * attended
            stream_pre_attn_rms = _valid_rms(x, mask)
            attention_residual_rms = _valid_rms(attention_residual, mask)
            attention_ratio = attention_residual_rms / stream_pre_attn_rms.clamp_min(
                1e-12
            )
            x = x + attention_residual
            stream_pre_ffn_rms = _valid_rms(x, mask)
            feed = block.feed_forward(_modulate(block.norm2(x), shift_f, scale_f), mask)
            feed_branch_rms = _valid_rms(feed, mask)
            feed_residual = gate_f * feed
            feed_residual_rms = _valid_rms(feed_residual, mask)
            feed_ratio = feed_residual_rms / stream_pre_ffn_rms.clamp_min(1e-12)
            x = (x + feed_residual) * mask.unsqueeze(-1).to(x.dtype)
            layers.append(
                {
                    "layer": index,
                    **attention_metrics,
                    "attn_gate_rms": float(gate_a.float().square().mean().sqrt()),
                    "ffn_gate_rms": float(gate_f.float().square().mean().sqrt()),
                    "attn_shift_rms": float(shift_a.float().square().mean().sqrt()),
                    "attn_scale_rms": float(scale_a.float().square().mean().sqrt()),
                    "ffn_shift_rms": float(shift_f.float().square().mean().sqrt()),
                    "ffn_scale_rms": float(scale_f.float().square().mean().sqrt()),
                    "stream_pre_attn_rms": float(stream_pre_attn_rms),
                    "attn_branch_pre_gate_rms": float(attention_branch_rms),
                    "attn_residual_rms": float(attention_residual_rms),
                    "attn_residual_ratio": float(attention_ratio),
                    "stream_pre_ffn_rms": float(stream_pre_ffn_rms),
                    "ffn_branch_pre_gate_rms": float(feed_branch_rms),
                    "ffn_residual_rms": float(feed_residual_rms),
                    "ffn_residual_ratio": float(feed_ratio),
                    "activation_rms": float(_valid_rms(x, mask)),
                }
            )
        shift, scale = model.final_modulation(conditioning).chunk(2, dim=-1)
        velocity = model.output(_modulate(model.final_norm(x), shift, scale))
    return {
        "t": t_value,
        "layers": layers,
        "velocity_rms": float(_valid_rms(velocity.float(), mask)),
    }


def _attention_metrics(attention, x: Tensor, mask: Tensor) -> dict[str, float]:
    batch, frames, _ = x.shape
    q_raw, k_raw, _ = attention.qkv(x).chunk(3, dim=-1)
    q_raw = q_raw.view(batch, frames, attention.heads, attention.head_dim).transpose(
        1, 2
    )
    k_raw = k_raw.view(batch, frames, attention.heads, attention.head_dim).transpose(
        1, 2
    )
    q = attention.q_norm(q_raw)
    k = attention.k_norm(k_raw)
    q, k = _rotary(q, k)
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * attention.scale
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        probabilities = logits.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(-1)
        lengths = mask.sum(-1).clamp_min(2).float().log()[:, None, None]
        normalized_entropy = entropy / lengths
        maximum = probabilities.max(-1).values
        query_mask = mask[:, None, :].expand(-1, attention.heads, -1)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        entropy_by_head = []
        maximum_by_head = []
        logits_std_by_head = []
        for head in range(attention.heads):
            entropy_by_head.append(
                normalized_entropy[:, head][query_mask[:, head]].mean()
            )
            maximum_by_head.append(maximum[:, head][query_mask[:, head]].mean())
            logits_std_by_head.append(logits[:, head][pair_mask[:, 0]].std())
        entropy_values = torch.stack(entropy_by_head)
        maximum_values = torch.stack(maximum_by_head)
        logits_std_values = torch.stack(logits_std_by_head)
    return {
        "attention_entropy_p10": float(torch.quantile(entropy_values, 0.1)),
        "attention_entropy_median": float(entropy_values.median()),
        "attention_entropy_p90": float(torch.quantile(entropy_values, 0.9)),
        "attention_max_probability_median": float(maximum_values.median()),
        "attention_logits_std_median": float(logits_std_values.median()),
        "q_pre_norm_rms": float(_valid_head_rms(q_raw, mask)),
        "k_pre_norm_rms": float(_valid_head_rms(k_raw, mask)),
    }


def _valid_rms(value: Tensor, mask: Tensor) -> Tensor:
    numeric_mask = mask.unsqueeze(-1).to(value.dtype)
    denominator = numeric_mask.sum() * value.shape[-1]
    return (value.float().square() * numeric_mask).sum().div(denominator).sqrt()


def _valid_head_rms(value: Tensor, mask: Tensor) -> Tensor:
    numeric_mask = mask[:, None, :, None].to(value.dtype)
    denominator = numeric_mask.sum() * value.shape[1] * value.shape[-1]
    return (value.float().square() * numeric_mask).sum().div(denominator).sqrt()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    os.close(descriptor)
    try:
        Path(temporary_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
