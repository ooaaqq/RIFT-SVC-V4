#!/usr/bin/env python3
"""Compare BF16 and selective FP8 on one deterministic real-data batch."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rift_v4.config import V4Config  # noqa: E402
from rift_v4.data import FeatureDataset, build_sampler, collate_features  # noqa: E402
from rift_v4.features import MelStats  # noqa: E402
from rift_v4.manifest import load_manifest  # noqa: E402
from rift_v4.performance import (  # noqa: E402
    apply_selective_float8_training,
    compile_model_in_place,
    configure_performance,
)
from rift_v4.train import _build_optimizer, build_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "bf16-reference",
            "gw-hp-compare",
            "bf16-continuation",
            "gw-hp-continuation",
        ),
        required=True,
    )
    parser.add_argument("--batch-index", type=int, default=496)
    parser.add_argument("--updates", type=int, default=50)
    args = parser.parse_args()

    config = V4Config.load(args.config)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    train_entries = [
        entry
        for entry in load_manifest(args.manifest)
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    dataset = FeatureDataset(
        train_entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        checkpoint["speaker_to_id"],
        voiced_crop_probability=config.sampling.voiced_crop_probability,
    )
    sampler = build_sampler(train_entries, config.sampling)
    sampler.set_epoch(0)
    requests = next(
        batch for index, batch in enumerate(sampler) if index == args.batch_index
    )
    if len(requests) != 32 or requests[0].frames != 512:
        raise ValueError("selected sampler batch is not B32 x T512")
    batch = collate_features([dataset[request] for request in requests])
    batch.pop("requested_length")
    batch.pop("length")

    device = torch.device("cuda")
    performance = replace(
        config.performance,
        compile_model=True,
        compile_mode="max-autotune",
        float8_training=args.mode.startswith("gw-hp"),
        float8_recipe="rowwise_with_gw_hp",
    )
    runtime = configure_performance(performance, device)
    system = build_system(config, len(checkpoint["speaker_to_id"])).to(device)
    system.load_state_dict(checkpoint["model"], strict=True)
    fp8 = apply_selective_float8_training(system.model, performance, device)
    compile_model_in_place(system.model, performance)
    system.train()

    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    if args.mode.endswith("continuation"):
        _run_continuation(
            args,
            config,
            checkpoint,
            system,
            batch,
            fp8,
            runtime,
        )
        return
    generator = torch.Generator(device=device).manual_seed(20260903)
    noise = torch.randn(
        batch["mel"].shape,
        device=device,
        dtype=batch["mel"].dtype,
        generator=generator,
    )
    timestep = torch.linspace(0.05, 0.95, len(requests), device=device)
    expanded_t = timestep[:, None, None]
    noisy = (1 - expanded_t) * noise + expanded_t * batch["mel"]
    target = batch["mel"] - noise
    weights = batch["mask"].unsqueeze(-1).to(batch["mel"].dtype)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = system.model(
            noisy,
            batch["content"],
            batch["f0"],
            batch["rms"],
            batch["speaker"],
            timestep,
            batch["mask"],
        )
        loss = ((prediction - target).square() * weights).sum() / (
            weights.sum() * config.mel.channels
        )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().to(torch.bfloat16)
        for name, parameter in system.named_parameters()
        if parameter.grad is not None
    }
    grad_norm = sum(
        gradient.float().square().sum().item() for gradient in gradients.values()
    ) ** 0.5

    result: dict[str, object] = {
        "mode": args.mode,
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "batch_size": len(requests),
        "sequence_length": int(batch["mel"].shape[1]),
        "valid_frames": int(batch["mask"].sum()),
        "performance": runtime,
        "float8": fp8,
    }
    if args.mode == "bf16-reference":
        torch.save(
            {"prediction": prediction.detach().cpu().float(), "gradients": gradients},
            args.reference,
        )
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        expected = reference["prediction"].float()
        actual = prediction.detach().cpu().float()
        difference = actual - expected
        result["prediction"] = {
            "cosine": float(torch.nn.functional.cosine_similarity(
                actual.reshape(1, -1), expected.reshape(1, -1)
            )),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected).clamp_min(1e-30)
            ),
            "max_abs": float(difference.abs().max()),
        }
        result["gradient"] = _gradient_comparison(
            gradients, reference["gradients"]
        )
    print(json.dumps(result, indent=2))


def _run_continuation(
    args: argparse.Namespace,
    config: V4Config,
    checkpoint: dict[str, object],
    system: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    fp8: dict[str, object],
    runtime: dict[str, object],
) -> None:
    optimizer, _ = _build_optimizer(system, config)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = (
            config.training.speaker_learning_rate
            if group.get("name") == "speaker"
            else config.training.learning_rate
        )
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    initial = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in system.named_parameters()
    }
    trajectory = []
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = system(batch).total
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 10 == 0 or update == args.updates:
            trajectory.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "grad_norm_before_clip": float(grad_norm),
                }
            )
    updates = {
        name: (parameter.detach().cpu().float() - initial[name]).to(torch.bfloat16)
        for name, parameter in system.named_parameters()
    }
    result: dict[str, object] = {
        "mode": args.mode,
        "updates": args.updates,
        "trajectory": trajectory,
        "performance": runtime,
        "float8": fp8,
    }
    if args.mode == "bf16-continuation":
        torch.save({"updates": updates}, args.reference)
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        result["parameter_update"] = _gradient_comparison(
            updates, reference["updates"]
        )
    print(json.dumps(result, indent=2))


def _gradient_comparison(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, object]:
    totals: dict[str, list[float]] = {}
    for name, gradient in actual.items():
        reference = expected[name].float()
        value = gradient.float()
        layer = (
            name.split(".blocks.", 1)[1].split(".", 1)[0]
            if ".blocks." in name
            else "other"
        )
        for group in ("all", f"layer_{layer}"):
            accumulator = totals.setdefault(group, [0.0, 0.0, 0.0])
            accumulator[0] += torch.dot(value.reshape(-1), reference.reshape(-1)).item()
            accumulator[1] += value.square().sum().item()
            accumulator[2] += reference.square().sum().item()
    return {
        group: {
            "cosine": values[0] / (values[1] * values[2]) ** 0.5,
            "norm_ratio": (values[1] / values[2]) ** 0.5,
        }
        for group, values in totals.items()
    }


if __name__ == "__main__":
    main()
