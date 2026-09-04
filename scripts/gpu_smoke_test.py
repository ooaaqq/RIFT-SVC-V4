#!/usr/bin/env python3
"""Measure V4 synthetic training memory and step throughput on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rift_v4.config import PerformanceConfig, V4Config  # noqa: E402
from rift_v4.flow import FlowMatchingSystem  # noqa: E402
from rift_v4.model import RIFTV4  # noqa: E402
from rift_v4.performance import (  # noqa: E402
    apply_selective_float8_training,
    compile_model_in_place,
    configure_performance,
)
from rift_v4.train import _build_optimizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--canonical-buckets", action="store_true")
    parser.add_argument(
        "--shared-model",
        action="store_true",
        help="reuse one model/optimizer across all requested shapes",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--model-mode", choices=("config", "eager", "compile"), default="config"
    )
    parser.add_argument(
        "--float8-mode", choices=("config", "off", "on"), default="config"
    )
    parser.add_argument(
        "--float8-recipe",
        choices=("rowwise_with_gw_hp", "rowwise", "tensorwise"),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune", "max-autotune-no-cudagraphs"),
    )
    args = parser.parse_args()
    config = V4Config.load(args.config)
    if args.canonical_buckets and args.frames:
        parser.error("--canonical-buckets and --frames are mutually exclusive")
    shapes = (
        [
            (
                min(
                    config.sampling.batch_size,
                    config.sampling.batch_frame_budget // frames,
                ),
                frames,
            )
            for frames in config.sampling.frame_buckets
        ]
        if args.canonical_buckets
        else [
            (max(1, budget // args.sequence_length), args.sequence_length)
            for budget in (args.frames or [config.sampling.batch_frame_budget])
        ]
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support BF16, required by V4")
    compile_model = (
        config.performance.compile_model
        if args.model_mode == "config"
        else args.model_mode == "compile"
    )
    performance = replace(config.performance, compile_model=compile_model)
    float8_training = (
        performance.float8_training
        if args.float8_mode == "config"
        else args.float8_mode == "on"
    )
    performance = replace(performance, float8_training=float8_training)
    if args.float8_recipe is not None:
        if not performance.float8_training:
            parser.error("--float8-recipe requires FP8 mode")
        performance = replace(performance, float8_recipe=args.float8_recipe)
    if args.compile_mode is not None:
        performance = replace(performance, compile_mode=args.compile_mode)
    if performance.float8_training and not performance.compile_model:
        raise SystemExit("FP8 smoke requires compiled model mode")
    runtime = configure_performance(performance, device)

    shared = (
        build_benchmark_system(config, device, performance)
        if args.shared_model
        else None
    )
    results = []
    for batch_size, sequence_length in shapes:
        if shared is None:
            result = benchmark(
                config,
                device,
                batch_size,
                sequence_length,
                args.warmup,
                args.steps,
                performance,
            )
        else:
            system, optimizer, metadata = shared
            result = benchmark_shape(
                system,
                optimizer,
                metadata,
                config,
                device,
                batch_size,
                sequence_length,
                args.warmup,
                args.steps,
            )
        result["frame_budget"] = batch_size * sequence_length
        result["batch_size"] = batch_size
        result["sequence_length"] = sequence_length
        results.append(result)
        print(json.dumps(result, indent=2))

    summary = {
        "gpu": torch.cuda.get_device_name(device),
        "performance": runtime,
        "results": results,
    }
    if args.canonical_buckets:
        probabilities = config.sampling.bucket_probabilities
        expected_frames = sum(
            probability * result["frame_budget"]
            for probability, result in zip(probabilities, results, strict=True)
        )
        expected_seconds = sum(
            probability * result["seconds_per_step"]
            for probability, result in zip(probabilities, results, strict=True)
        )
        summary["weighted_materialized_frames_per_second"] = (
            expected_frames / expected_seconds
        )
    print(
        json.dumps(
            summary
        )
    )


def benchmark(
    config: V4Config,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    steps: int,
    performance: PerformanceConfig,
) -> dict[str, object]:
    system, optimizer, metadata = build_benchmark_system(config, device, performance)
    return benchmark_shape(
        system,
        optimizer,
        metadata,
        config,
        device,
        batch_size,
        sequence_length,
        warmup,
        steps,
    )


def build_benchmark_system(
    config: V4Config,
    device: torch.device,
    performance: PerformanceConfig,
) -> tuple[FlowMatchingSystem, torch.optim.Optimizer, dict[str, object]]:
    model = RIFTV4(
        config.mel.channels,
        config.model.content_dim,
        1 if config.training.freeze_timestep_and_modulation else 8,
        config.model.dim,
        config.model.depth,
        config.model.head_dim,
        config.model.ff_hidden_dim,
        config.model.kernel_size,
    ).to(device)
    system = FlowMatchingSystem(
        model=model,
        speaker_drop_probability=config.training.speaker_drop_probability,
    ).to(device)
    float8_summary = apply_selective_float8_training(system.model, performance, device)
    optimizer, optimizer_summary = _build_optimizer(system, config)
    compiled = compile_model_in_place(system.model, performance)
    return system, optimizer, {
        "float8": float8_summary,
        "optimizer": optimizer_summary,
        "model_compiled": compiled,
    }


def benchmark_shape(
    system: FlowMatchingSystem,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, object],
    config: V4Config,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    tensors = {
        "mel": torch.randn(
            batch_size, sequence_length, config.mel.channels, device=device
        ),
        "content": torch.randn(
            batch_size, sequence_length, config.model.content_dim, device=device
        ),
        "f0": torch.rand(batch_size, sequence_length, 1, device=device) * 500 + 80,
        "rms": torch.randn(batch_size, sequence_length, 1, device=device),
        "speaker": torch.zeros(batch_size, dtype=torch.long, device=device),
        "mask": _padded_mask(batch_size, sequence_length, device),
    }

    memory_before = _cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step(system, optimizer, tensors)
    torch.cuda.synchronize(device)
    memory_after_warmup = _cuda_memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        step(system, optimizer, tensors)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    valid_frames = int(tensors["mask"].sum())
    optimizer_summary = metadata["optimizer"]
    assert isinstance(optimizer_summary, dict)
    return {
        "seconds_per_step": elapsed / steps,
        "steps_per_second": steps / elapsed,
        "valid_frames_per_second": valid_frames * steps / elapsed,
        "materialized_frames_per_second": (
            batch_size * sequence_length * steps / elapsed
        ),
        "requested_frames_per_second": (batch_size * sequence_length * steps / elapsed),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "memory_before_gib": memory_before,
        "memory_after_warmup_gib": memory_after_warmup,
        "memory_after_timing_gib": _cuda_memory(device),
        "trainable_parameters": int(optimizer_summary["trainable_parameters"]),
        "frozen_parameters": int(optimizer_summary["frozen_parameters"]),
        "model_compiled": metadata["model_compiled"],
        "fused_adamw_active": optimizer_summary["fused_adamw_active"],
        "float8": metadata["float8"],
    }


def _cuda_memory(device: torch.device) -> dict[str, float]:
    return {
        "allocated": torch.cuda.memory_allocated(device) / 2**30,
        "reserved": torch.cuda.memory_reserved(device) / 2**30,
    }


def step(system, optimizer, tensors) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = system(tensors).total
    loss.backward()
    optimizer.step()


def _padded_mask(
    batch_size: int, sequence_length: int, device: torch.device
) -> torch.Tensor:
    minimum = max(1, sequence_length * 3 // 4)
    lengths = torch.linspace(
        sequence_length,
        minimum,
        batch_size,
        device=device,
        dtype=torch.float32,
    ).round()
    positions = torch.arange(sequence_length, device=device)
    return positions.unsqueeze(0) < lengths.to(torch.long).unsqueeze(1)


if __name__ == "__main__":
    main()
