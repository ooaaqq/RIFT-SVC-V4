from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import V4Config
from .data import (
    FeatureDataset,
    HierarchicalBatchSampler,
    SampleRequest,
    build_sampler,
    collate_features,
)
from .features import MelStats
from .flow import FlowMatchingSystem
from .manifest import ManifestEntry, load_manifest
from .model import RIFTV4
from .performance import (
    apply_selective_float8_training,
    compile_model_in_place,
    configure_performance,
    software_versions,
)


def build_system(config: V4Config, num_speakers: int) -> FlowMatchingSystem:
    model = RIFTV4(
        mel_channels=config.mel.channels,
        content_dim=config.model.content_dim,
        num_speakers=num_speakers,
        dim=config.model.dim,
        depth=config.model.depth,
        head_dim=config.model.head_dim,
        ff_hidden_dim=config.model.ff_hidden_dim,
        kernel_size=config.model.kernel_size,
    )
    return FlowMatchingSystem(
        model=model,
        speaker_drop_probability=config.training.speaker_drop_probability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or explicitly train RIFT-SVC V4"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/v4"))
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="start a one-singer fine-tune from the EMA of a foundation checkpoint",
    )
    parser.add_argument(
        "--stop-at-step",
        type=int,
        help=("stop at this step; it must not exceed configured max_steps"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--resume-lr-scale",
        type=float,
        help="multiply every optimizer-group LR after resuming (for manual decay)",
    )
    parser.add_argument(
        "--performance-recipe",
        choices=("bf16", "rowwise_with_gw_hp", "rowwise", "tensorwise"),
        help="explicitly override the training precision recipe",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune", "max-autotune-no-cudagraphs"),
        help="explicitly override torch.compile mode",
    )
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if args.resume is not None and args.initialize_from is not None:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if args.resume_lr_scale is not None and args.resume is None:
        parser.error("--resume-lr-scale requires --resume")
    if args.resume_lr_scale is not None and args.resume_lr_scale <= 0:
        parser.error("--resume-lr-scale must be positive")

    config = V4Config.load(args.config)
    performance_overrides: dict[str, object] = {}
    if args.performance_recipe is not None:
        performance_overrides["float8_training"] = args.performance_recipe != "bf16"
        if args.performance_recipe != "bf16":
            performance_overrides["float8_recipe"] = args.performance_recipe
    if args.compile_mode is not None:
        performance_overrides["compile_mode"] = args.compile_mode
    if performance_overrides:
        config = replace(
            config,
            performance=replace(config.performance, **performance_overrides),
        )
        config.validate()
    entries = load_manifest(args.manifest)
    accepted = [entry for entry in entries if entry.quality_status == "accepted"]
    summary = {
        "recordings": len(entries),
        "accepted": len(accepted),
        "train_accepted": sum(entry.split == "train" for entry in accepted),
        "datasets": sorted({entry.dataset for entry in accepted}),
        "speakers": len({entry.speaker_key for entry in accepted}),
        "accepted_hours": sum(entry.duration_seconds for entry in accepted) / 3600,
        "config": asdict(config),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute_training:
        print("validation only; pass --execute-training to allocate a model and train")
        return
    _train(
        config,
        entries,
        args.output,
        args.mel_stats,
        args.device,
        args.num_workers,
        args.resume,
        args.initialize_from,
        args.stop_at_step,
        args.resume_lr_scale,
        frozenset(performance_overrides),
    )


def _train(
    config: V4Config,
    entries: list[ManifestEntry],
    output: Path,
    mel_stats_path: Path,
    device_name: str,
    num_workers: int,
    resume: Path | None,
    initialize_from: Path | None,
    stop_at_step: int | None,
    resume_lr_scale: float | None,
    performance_override_fields: frozenset[str],
) -> None:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if stop_at_step is not None and stop_at_step <= 0:
        raise ValueError("stop-at-step must be positive")
    target_steps = stop_at_step or config.training.max_steps
    if target_steps > config.training.max_steps:
        raise ValueError("stop-at-step cannot exceed configured max_steps")
    torch.manual_seed(config.sampling.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.sampling.seed)
    train_entries = [
        entry
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    if not train_entries:
        raise ValueError("no accepted training entries")
    _verify_content_provenance(train_entries)
    mel_stats = MelStats.load(mel_stats_path, config.mel.channels)
    dataset = FeatureDataset(
        train_entries,
        config.mel.channels,
        config.model.content_dim,
        mel_stats,
        voiced_crop_probability=config.sampling.voiced_crop_probability,
    )
    validation_entries = [
        entry
        for entry in entries
        if entry.split == "validation"
        and entry.quality_status == "accepted"
        and entry.speaker_key in dataset.speaker_to_id
    ]
    validation_dataset = FeatureDataset(
        validation_entries,
        config.mel.channels,
        config.model.content_dim,
        mel_stats,
        dataset.speaker_to_id,
        voiced_crop_probability=config.sampling.voiced_crop_probability,
    )
    sampler = build_sampler(train_entries, config.sampling)
    validation_weights = dict(
        zip(sampler.datasets, sampler.dataset_probabilities, strict=True)
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=num_workers,
        pin_memory=device_name.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    device = torch.device(device_name)
    performance_runtime = configure_performance(config.performance, device)
    system = build_system(config, len(dataset.speaker_to_id)).to(device)
    initialization = None
    if initialize_from is not None:
        initialization = _initialize_from_ema(
            system, initialize_from, dataset.speaker_to_id, config
        )
    resume_payload = None
    if resume is not None:
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
    performance_fork = (
        _validate_resume_config(
            resume_payload.get("config") if isinstance(resume_payload, dict) else None,
            asdict(config),
            performance_override_fields,
        )
        if resume is not None
        else None
    )
    performance_runtime["float8"] = apply_selective_float8_training(
        system.model, config.performance, device
    )
    optimizer, optimizer_summary = _build_optimizer(system, config)
    performance_runtime["model_compiled"] = compile_model_in_place(
        system.model, config.performance
    )
    base_learning_rates = [group["lr"] for group in optimizer.param_groups]
    use_bf16 = config.training.precision == "bf16" and device.type == "cuda"
    output.mkdir(parents=True, exist_ok=True)
    _prepare_run_directory(output, resume)
    _write_run_metadata(
        output,
        config,
        train_entries,
        mel_stats_path,
        device,
        num_workers,
        resume,
        initialize_from,
        initialization,
        optimizer_summary,
        performance_runtime,
        target_steps,
        performance_fork,
    )
    _write_sampling_audit(output, sampler, config)
    _append_run_event(
        output,
        "start",
        {
            "target_steps": target_steps,
            "resume_checkpoint": resume.resolve().as_posix() if resume else None,
            "initialize_from": (
                initialize_from.resolve().as_posix() if initialize_from else None
            ),
            "optimizer": optimizer_summary,
            "performance": performance_runtime,
            "learning_rate_schedule": config.training.learning_rate_schedule,
            "performance_fork": performance_fork,
        },
    )
    step = 0
    epoch = 0
    batch_offset = 0
    best_validation = float("inf")
    ema = {name: value.detach().clone() for name, value in system.state_dict().items()}
    learning_rate_multiplier = 1.0
    if resume is not None:
        checkpoint = resume_payload
        if not isinstance(checkpoint, dict):
            raise ValueError("resume checkpoint payload is invalid")
        if checkpoint.get("schema_version") != 4:
            raise ValueError("resume checkpoint does not use schema 4")
        if checkpoint.get("checkpoint_kind") != "full":
            raise ValueError("resume requires a full checkpoint")
        if checkpoint.get("speaker_to_id") != dataset.speaker_to_id:
            raise ValueError("resume checkpoint speaker mapping differs")
        if checkpoint.get("manifest_sha256") != _manifest_digest(train_entries):
            raise ValueError("resume checkpoint training manifest differs")
        if checkpoint.get("mel_stats") != asdict(mel_stats):
            raise ValueError("resume checkpoint mel statistics differ")
        if checkpoint.get("software") != software_versions(config.performance):
            raise ValueError("resume checkpoint software versions differ")
        system.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema = {name: value.to(device) for name, value in checkpoint["ema"].items()}
        step = int(checkpoint["step"])
        epoch = int(checkpoint["epoch"])
        batch_offset = int(checkpoint["batch_offset"])
        if batch_offset >= len(sampler):
            epoch += batch_offset // len(sampler)
            batch_offset %= len(sampler)
        best_validation = (
            float(checkpoint.get("best_validation", float("inf")))
            if checkpoint.get("validation_protocol") == 6
            else float("inf")
        )
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        learning_rate_multiplier = float(
            checkpoint.get("learning_rate_multiplier", 1.0)
        )
        if not math.isfinite(learning_rate_multiplier) or learning_rate_multiplier <= 0:
            raise ValueError("checkpoint learning-rate multiplier is invalid")
        if resume_lr_scale is not None:
            previous_multiplier = learning_rate_multiplier
            learning_rate_multiplier *= resume_lr_scale
            _append_run_event(
                output,
                "manual_learning_rate_decay",
                {
                    "step": step,
                    "scale": resume_lr_scale,
                    "previous_multiplier": previous_multiplier,
                    "learning_rate_multiplier": learning_rate_multiplier,
                },
            )
    if performance_fork:
        _append_run_event(
            output,
            "performance_fork",
            {"step": step, "changes": performance_fork},
        )
    performance_runtime["compile_warmup"] = _warm_compile_buckets(
        system,
        config,
        device,
        len(dataset.speaker_to_id),
        use_bf16,
    )
    _append_run_event(
        output,
        "compile_warmup",
        {"step": step, **performance_runtime["compile_warmup"]},
    )
    system.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    log_started = time.perf_counter()
    logged_step = step
    logged_frames = 0
    logged_materialized_frames = 0
    logged_requested_frames = 0
    logged_samples = 0
    shape_window: deque[tuple[int, int, int, int]] = deque(maxlen=1000)
    while step < target_steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_offset:
                continue
            if step >= target_steps:
                break
            requested_lengths = batch.pop("requested_length")
            requested = int(requested_lengths[0])
            materialized = int(batch["mel"].shape[1])
            batch_size = int(batch["length"].shape[0])
            valid = int(batch["length"].sum())
            logged_frames += valid
            logged_materialized_frames += batch_size * materialized
            logged_requested_frames += batch_size * requested
            logged_samples += int(batch["length"].shape[0])
            shape_window.append((requested, materialized, valid, batch_size))
            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }
            learning_rate_scale = _base_learning_rate_scale(config, step)
            learning_rates = [
                base_learning_rate * learning_rate_scale * learning_rate_multiplier
                for base_learning_rate in base_learning_rates
            ]
            for group, learning_rate in zip(
                optimizer.param_groups, learning_rates, strict=True
            ):
                group["lr"] = learning_rate
            accumulation = config.training.gradient_accumulation_steps
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = system(batch)
            _require_finite(loss.total, "total loss", step)
            (loss.total / accumulation).backward()
            micro_step += 1
            batch_offset = batch_index + 1
            if micro_step % accumulation:
                continue
            grad_norm = nn.utils.clip_grad_norm_(
                system.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=True,
            )
            _require_finite(grad_norm, "gradient norm", step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _update_ema(ema, system, config.training.ema_decay)
            step += 1
            if step % config.training.log_every_steps == 0:
                elapsed = time.perf_counter() - log_started
                steps = step - logged_step
                details = {
                    "step": step,
                    "total_loss": float(loss.total.detach()),
                    "flow": float(loss.flow.detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "speaker_learning_rate": next(
                        (
                            group["lr"]
                            for group in optimizer.param_groups
                            if group.get("name") == "speaker"
                        ),
                        optimizer.param_groups[0]["lr"],
                    ),
                    "learning_rate_multiplier": learning_rate_multiplier,
                    "grad_norm": float(grad_norm.detach()),
                    "steps_per_second": steps / max(elapsed, 1e-9),
                    "frames_per_second": logged_frames / max(elapsed, 1e-9),
                    "valid_frames_per_second": logged_frames / max(elapsed, 1e-9),
                    "materialized_frames_per_second": (
                        logged_materialized_frames / max(elapsed, 1e-9)
                    ),
                    "requested_frames_per_second": (
                        logged_requested_frames / max(elapsed, 1e-9)
                    ),
                    "samples_per_second": logged_samples / max(elapsed, 1e-9),
                    "max_cuda_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
                details.update(_shape_statistics(shape_window, config))
                print(json.dumps(details), flush=True)
                _append_run_event(output, "train", details)
                log_started = time.perf_counter()
                logged_step = step
                logged_frames = 0
                logged_materialized_frames = 0
                logged_requested_frames = 0
                logged_samples = 0
            full_checkpoint_due = (
                step % config.training.resume_checkpoint_every_steps == 0
            )
            audit_checkpoint_due = (
                step % config.training.audit_checkpoint_every_steps == 0
                and not full_checkpoint_due
            )
            if audit_checkpoint_due:
                _checkpoint(
                    output / f"step-{step:07d}.pt",
                    system,
                    optimizer,
                    ema,
                    learning_rate_multiplier,
                    step,
                    epoch,
                    batch_offset,
                    best_validation,
                    config,
                    dataset.speaker_to_id,
                    train_entries,
                    mel_stats,
                    kind="audit",
                )
            if full_checkpoint_due:
                # Persist resumable state before optional validation/audits. A
                # diagnostic failure must never discard the preceding 5k steps.
                _checkpoint(
                    output / f"resume-step-{step:07d}.pt",
                    system,
                    optimizer,
                    ema,
                    learning_rate_multiplier,
                    step,
                    epoch,
                    batch_offset,
                    best_validation,
                    config,
                    dataset.speaker_to_id,
                    train_entries,
                    mel_stats,
                    kind="full",
                )
            if (
                validation_entries
                and step % config.training.validation_every_steps == 0
            ):
                # Validation has variable batch and sequence shapes. Keep the three
                # canonical training graphs specialized and evaluate these rare
                # shapes eagerly instead of compiling a graph for every recording.
                with (
                    torch.compiler.set_stance("force_eager"),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ),
                ):
                    online_validation = _validation_loss(
                        system,
                        validation_dataset,
                        device,
                        config.sampling.frame_buckets[-1],
                        config.sampling.batch_size,
                        validation_weights,
                        config.sampling.synthetic_datasets,
                        config.evaluation.validation_recordings_per_song,
                    )
                    with _use_ema_parameters(system, ema):
                        validation = _validation_loss(
                            system,
                            validation_dataset,
                            device,
                            config.sampling.frame_buckets[-1],
                            config.sampling.batch_size,
                            validation_weights,
                            config.sampling.synthetic_datasets,
                            config.evaluation.validation_recordings_per_song,
                        )
                validation_path = _write_validation_details(
                    output, step, validation, online_validation
                )
                selection = float(validation["real_speaker_macro_flow"])
                details = {
                    "step": step,
                    "validation_flow": selection,
                    "validation_selection_metric": "real_speaker_macro_flow",
                    "validation_mixture_weighted_flow": validation[
                        "mixture_weighted_flow"
                    ],
                    "validation_dataset_macro_flow": validation["dataset_macro_flow"],
                    "validation_real_dataset_macro_flow": validation[
                        "real_dataset_macro_flow"
                    ],
                    "validation_by_dataset": validation["by_dataset"],
                    "online_validation_flow": online_validation[
                        "real_speaker_macro_flow"
                    ],
                    "online_validation_by_dataset": online_validation["by_dataset"],
                    "condition_f0_voicing": online_validation["condition_f0_voicing"],
                    "validation_details": validation_path.relative_to(
                        output
                    ).as_posix(),
                }
                print(json.dumps(details), flush=True)
                _append_run_event(output, "validation", details)
                if selection < best_validation:
                    best_validation = selection
                    _checkpoint(
                        output / "best-flow-health.pt",
                        system,
                        optimizer,
                        ema,
                        learning_rate_multiplier,
                        step,
                        epoch,
                        batch_offset,
                        best_validation,
                        config,
                        dataset.speaker_to_id,
                        train_entries,
                        mel_stats,
                        kind="audit",
                    )
        if step >= target_steps:
            break
        epoch += 1
        batch_offset = 0
    _checkpoint(
        output / "final.pt",
        system,
        optimizer,
        ema,
        learning_rate_multiplier,
        step,
        epoch,
        batch_offset,
        best_validation,
        config,
        dataset.speaker_to_id,
        train_entries,
        mel_stats,
        kind="full",
    )


def _verify_content_provenance(entries: list[ManifestEntry]) -> None:
    missing = [
        entry.id
        for entry in entries
        if not entry.content_feature_path
        or not entry.content_encoder_id
        or not entry.content_encoder_sha256
    ]
    if missing:
        raise ValueError(
            f"training content features lack encoder provenance: {missing[:5]}"
        )
    hashes = {entry.content_encoder_sha256 for entry in entries}
    identifiers = {entry.content_encoder_id for entry in entries}
    if len(hashes) != 1 or len(identifiers) != 1:
        raise ValueError("RIFT training cannot mix content encoder versions")
    if not next(iter(identifiers)).startswith("contentvec-dualphase10ms-v1:"):
        raise ValueError("RIFT training requires dual-phase ContentVec features")
    absent = [
        entry.content_feature_path
        for entry in entries
        if not Path(entry.content_feature_path or "").is_file()
    ]
    if absent:
        raise FileNotFoundError(f"missing raw content features: {absent[:5]}")


@torch.inference_mode()
def _validation_loss(
    system: FlowMatchingSystem,
    dataset: FeatureDataset,
    device: torch.device,
    frames: int,
    batch_size: int,
    dataset_probabilities: dict[str, float],
    synthetic_datasets: tuple[str, ...] = (),
    recordings_per_song: int = 1,
) -> dict[str, object]:
    if recordings_per_song <= 0:
        raise ValueError("validation recordings per song must be positive")
    training_modes = {module: module.training for module in system.modules()}
    system.eval()
    losses_by_song: dict[str, float] = {}
    losses_by_speaker: dict[str, float] = {}
    losses_by_dataset: dict[str, float] = {}
    f0_by_dataset: dict[str, list[torch.Tensor]] = {}
    try:
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2026)
            hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
            for index, entry in enumerate(dataset.entries):
                hierarchy.setdefault(entry.dataset, {}).setdefault(
                    entry.speaker, {}
                ).setdefault(entry.song, []).append(index)
            for dataset_name, speakers in sorted(hierarchy.items()):
                dataset_speaker_losses = []
                for speaker_name, songs in sorted(speakers.items()):
                    selected_records: list[tuple[str, int]] = []
                    for song_name, candidates in sorted(songs.items()):
                        ranked = sorted(
                            candidates,
                            key=lambda index: hashlib.sha256(
                                f"2026\0{dataset.entries[index].id}".encode()
                            ).digest(),
                        )
                        selected_records.extend(
                            (song_name, index) for index in ranked[:recordings_per_song]
                        )
                    song_values: dict[str, list[float]] = {
                        song_name: [] for song_name in songs
                    }
                    for start in range(0, len(selected_records), batch_size):
                        selected = selected_records[start : start + batch_size]
                        items = [
                            dataset[SampleRequest(index, frames, 2026 + index)]
                            for _, index in selected
                        ]
                        batch = collate_features(items)
                        batch.pop("requested_length")
                        valid_f0 = batch["f0"].squeeze(-1)[batch["mask"]]
                        f0_by_dataset.setdefault(dataset_name, []).append(valid_f0)
                        batch = {
                            name: value.to(device) for name, value in batch.items()
                        }
                        result = system(batch)
                        if not hasattr(result, "flow_by_sample"):
                            raise TypeError("validation requires per-sample flow loss")
                        for (song_name, _), value in zip(
                            selected, result.flow_by_sample, strict=True
                        ):
                            song_values[song_name].append(float(value))
                    speaker_song_losses = []
                    for song_name, values in sorted(song_values.items()):
                        loss = sum(values) / len(values)
                        losses_by_song[f"{dataset_name}:{speaker_name}:{song_name}"] = (
                            loss
                        )
                        speaker_song_losses.append(loss)
                    speaker_loss = sum(speaker_song_losses) / len(speaker_song_losses)
                    losses_by_speaker[f"{dataset_name}:{speaker_name}"] = speaker_loss
                    dataset_speaker_losses.append(speaker_loss)
                losses_by_dataset[dataset_name] = sum(dataset_speaker_losses) / len(
                    dataset_speaker_losses
                )
    finally:
        for module, training in training_modes.items():
            module.train(training)
    weights = {name: dataset_probabilities[name] for name in losses_by_dataset}
    denominator = sum(weights.values())
    weighted = (
        sum(losses_by_dataset[name] * weights[name] for name in losses_by_dataset)
        / denominator
    )
    real_datasets = set(losses_by_dataset) - set(synthetic_datasets)
    real_speakers = {
        name: value
        for name, value in losses_by_speaker.items()
        if name.split(":", 1)[0] in real_datasets
    }
    if not real_datasets or not real_speakers:
        raise ValueError("validation has no real dataset speakers")
    f0_statistics = {
        "definition": "conditioning F0 on the deterministic validation crops",
        "overall": _f0_voicing_summary(
            torch.cat([value for values in f0_by_dataset.values() for value in values])
        ),
        "by_dataset": {
            name: _f0_voicing_summary(torch.cat(values))
            for name, values in sorted(f0_by_dataset.items())
        },
    }
    return {
        "protocol": 6,
        "selection_metric": "real_speaker_macro_flow",
        "mixture_weighted_flow": weighted,
        "dataset_macro_flow": sum(losses_by_dataset.values()) / len(losses_by_dataset),
        "real_dataset_macro_flow": sum(
            losses_by_dataset[name] for name in real_datasets
        )
        / len(real_datasets),
        "speaker_macro_flow": sum(losses_by_speaker.values()) / len(losses_by_speaker),
        "real_speaker_macro_flow": sum(real_speakers.values()) / len(real_speakers),
        "by_dataset": losses_by_dataset,
        "by_speaker": losses_by_speaker,
        "by_song": losses_by_song,
        "condition_f0_voicing": f0_statistics,
    }


def _f0_voicing_summary(f0: torch.Tensor) -> dict[str, float | int]:
    values = f0.detach().float().cpu().flatten()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("validation F0 must be non-empty and finite")
    voiced = values[values > 0]
    result: dict[str, float | int] = {
        "frames": values.numel(),
        "voiced_frames": voiced.numel(),
        "voiced_ratio": float(voiced.numel() / values.numel()),
    }
    if voiced.numel():
        quantiles = torch.quantile(
            voiced, torch.tensor([0.05, 0.5, 0.95], dtype=voiced.dtype)
        )
        result.update(
            {
                "f0_hz_p05": float(quantiles[0]),
                "f0_hz_median": float(quantiles[1]),
                "f0_hz_p95": float(quantiles[2]),
            }
        )
    return result


@torch.no_grad()
def _update_ema(
    ema: dict[str, torch.Tensor], system: FlowMatchingSystem, decay: float
) -> None:
    for name, value in system.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1.0 - decay)
        else:
            ema[name].copy_(value)


@contextmanager
def _use_ema_parameters(system: FlowMatchingSystem, ema: dict[str, torch.Tensor]):
    """Temporarily swap EMA weights into the live model without another copy."""

    state = system.state_dict()
    if state.keys() != ema.keys():
        raise ValueError("EMA state does not match the model")
    swapped: list[str] = []
    try:
        with torch.no_grad():
            for name, value in state.items():
                temporary = value.detach().clone()
                value.copy_(ema[name])
                ema[name].copy_(temporary)
                swapped.append(name)
        yield
    finally:
        with torch.no_grad():
            for name in reversed(swapped):
                value = state[name]
                temporary = value.detach().clone()
                value.copy_(ema[name])
                ema[name].copy_(temporary)


def _checkpoint(
    path: Path,
    system: FlowMatchingSystem,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    learning_rate_multiplier: float,
    step: int,
    epoch: int,
    batch_offset: int,
    best_validation: float,
    config: V4Config,
    speaker_to_id: dict[str, int],
    entries: list[ManifestEntry],
    mel_stats: MelStats,
    *,
    kind: str,
) -> None:
    if kind not in {"audit", "full"}:
        raise ValueError("checkpoint kind must be audit or full")
    _require_finite_state(system, optimizer, ema, step)
    logged_best_validation = best_validation if math.isfinite(best_validation) else None
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    os.close(descriptor)
    try:
        payload = {
            "schema_version": 4,
            "checkpoint_kind": kind,
            "validation_protocol": 6,
            "step": step,
            "best_validation": best_validation,
            "model": system.state_dict(),
            "ema": ema,
            "learning_rate_multiplier": learning_rate_multiplier,
            "config": asdict(config),
            "software": software_versions(config.performance),
            "speaker_to_id": speaker_to_id,
            "manifest_sha256": _manifest_digest(entries),
            "mel_stats": asdict(mel_stats),
        }
        if kind == "full":
            payload.update(
                {
                    "epoch": epoch,
                    "batch_offset": batch_offset,
                    "optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                }
            )
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
        index_path = path.parent / "checkpoint_index.jsonl"
        with index_path.open("a", encoding="utf-8") as index:
            index.write(
                json.dumps(
                    {
                        "saved_at": time.time(),
                        "kind": kind,
                        "path": path.name,
                        "step": step,
                        "epoch": epoch,
                        "best_validation": logged_best_validation,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        _append_run_event(
            path.parent,
            "checkpoint",
            {
                "kind": kind,
                "path": path.name,
                "step": step,
                "epoch": epoch,
                "best_validation": logged_best_validation,
            },
        )
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _require_finite(value: torch.Tensor, name: str, step: int) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"non-finite {name} at step {step}")


def _require_finite_state(
    system: FlowMatchingSystem,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    step: int,
) -> None:
    for name, value in system.state_dict().items():
        if value.is_floating_point():
            _require_finite(value, f"model state {name}", step)
    for name, value in ema.items():
        if value.is_floating_point():
            _require_finite(value, f"EMA state {name}", step)
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                _require_finite(value, f"optimizer state {name}", step)


def _manifest_digest(entries: list[ManifestEntry]) -> str:
    payload = "\n".join(
        json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False)
        for entry in sorted(entries, key=lambda item: item.id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _initialize_from_ema(
    system: FlowMatchingSystem,
    checkpoint_path: Path,
    target_speaker_to_id: dict[str, int],
    config: V4Config,
) -> dict[str, object]:
    if len(target_speaker_to_id) != 1 or set(target_speaker_to_id.values()) != {0}:
        raise ValueError("foundation initialization requires exactly one target singer")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("schema_version") != 4 or "ema" not in checkpoint:
        raise ValueError("foundation checkpoint must be schema 4 with EMA weights")
    source_config = checkpoint.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("foundation checkpoint has no configuration")
    expected_contract = {
        "sample_rate": config.sample_rate,
        "hop_length": config.hop_length,
        "mel": asdict(config.mel),
        "model": asdict(config.model),
        "content_encoder": asdict(config.content_encoder),
    }
    mismatches = [
        name
        for name, value in expected_contract.items()
        if source_config.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "foundation architecture/feature contract differs: " + ", ".join(mismatches)
        )
    source_speakers = checkpoint.get("speaker_to_id")
    if not isinstance(source_speakers, dict) or not source_speakers:
        raise ValueError("foundation checkpoint has no speaker mapping")
    source_state = checkpoint["ema"]
    if not isinstance(source_state, dict):
        raise ValueError("foundation EMA state is invalid")
    target_state = system.state_dict()
    speaker_key = "model.speaker.weight"
    if speaker_key not in source_state or speaker_key not in target_state:
        raise ValueError("foundation checkpoint has no speaker embedding")
    source_embedding = source_state[speaker_key]
    source_null_id = len(source_speakers)
    if source_embedding.shape != (source_null_id + 1, config.model.dim):
        raise ValueError("foundation speaker embedding shape is inconsistent")
    synthetic = set(source_config.get("sampling", {}).get("synthetic_datasets", []))
    real_ids = sorted(
        int(index)
        for speaker, index in source_speakers.items()
        if speaker.split(":", 1)[0] not in synthetic
    )
    if not real_ids or real_ids[0] < 0 or real_ids[-1] >= source_null_id:
        raise ValueError("foundation checkpoint has no valid real-speaker embeddings")
    adapted = {
        name: value for name, value in source_state.items() if name != speaker_key
    }
    expected_keys = set(target_state) - {speaker_key}
    if set(adapted) != expected_keys:
        missing = sorted(expected_keys - set(adapted))[:5]
        unexpected = sorted(set(adapted) - expected_keys)[:5]
        raise ValueError(
            f"foundation state keys differ: missing={missing}, unexpected={unexpected}"
        )
    target_embedding = target_state[speaker_key].detach().clone()
    target_embedding[0].copy_(source_embedding[real_ids].mean(dim=0))
    target_embedding[1].copy_(source_embedding[source_null_id])
    adapted[speaker_key] = target_embedding
    system.load_state_dict(adapted, strict=True)
    return {
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": _file_digest(checkpoint_path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_state": "ema",
        "source_real_speakers_averaged": len(real_ids),
        "target_speaker": next(iter(target_speaker_to_id)),
        "speaker_initialization": "mean_real_speakers",
        "null_initialization": "source_null_speaker",
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PERFORMANCE_FORK_FIELDS = frozenset(
    {"float8_training", "float8_recipe", "compile_mode"}
)


def _validate_resume_config(
    checkpoint: object,
    current: dict[str, object],
    override_fields: frozenset[str],
) -> dict[str, dict[str, object]] | None:
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint configuration is invalid")
    unexpected = override_fields - _PERFORMANCE_FORK_FIELDS
    if unexpected:
        raise ValueError(f"unsupported performance fork fields: {sorted(unexpected)}")
    previous = copy.deepcopy(checkpoint)
    comparable = copy.deepcopy(current)
    previous_performance = previous.get("performance")
    current_performance = comparable.get("performance")
    if not isinstance(previous_performance, dict) or not isinstance(
        current_performance, dict
    ):
        raise ValueError("resume checkpoint performance configuration is invalid")
    changes: dict[str, dict[str, object]] = {}
    for field in override_fields:
        old = previous_performance.get(field)
        new = current_performance.get(field)
        if old != new:
            changes[field] = {"from": old, "to": new}
        current_performance[field] = old
    if previous != comparable:
        raise ValueError(
            "resume checkpoint configuration differs outside the explicit "
            "performance fork"
        )
    return changes or None


def _warm_compile_buckets(
    system: FlowMatchingSystem,
    config: V4Config,
    device: torch.device,
    num_speakers: int,
    use_bf16: bool,
) -> dict[str, object]:
    if not config.performance.compile_model:
        return {"enabled": False, "reason": "model_not_compiled"}
    if not config.performance.compile_warmup_buckets:
        return {"enabled": False, "reason": "disabled_by_config"}
    if device.type != "cuda":
        return {"enabled": False, "reason": "cuda_required"}

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    was_training = system.training
    timings: list[dict[str, object]] = []
    try:
        system.train()
        for frames in config.sampling.frame_buckets:
            batch_size = min(
                config.sampling.batch_size,
                config.sampling.batch_frame_budget // frames,
            )
            batch = {
                "mel": torch.randn(
                    batch_size, frames, config.mel.channels, device=device
                ),
                "content": torch.randn(
                    batch_size, frames, config.model.content_dim, device=device
                ),
                "f0": torch.rand(batch_size, frames, 1, device=device) * 500 + 80,
                "rms": torch.randn(batch_size, frames, 1, device=device),
                "speaker": torch.randint(num_speakers, (batch_size,), device=device),
                "mask": torch.ones(batch_size, frames, dtype=torch.bool, device=device),
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(config.performance.compile_warmup_steps_per_bucket):
                system.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                ):
                    loss = system(batch).total
                _require_finite(loss, "compile warmup loss", 0)
                loss.backward()
            torch.cuda.synchronize(device)
            timings.append(
                {
                    "batch_size": batch_size,
                    "frames": frames,
                    "steps": config.performance.compile_warmup_steps_per_bucket,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    finally:
        system.zero_grad(set_to_none=True)
        system.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    return {"enabled": True, "shapes": timings}


def _shape_statistics(
    observations: deque[tuple[int, int, int, int]], config: V4Config
) -> dict[str, object]:
    by_bucket: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    canonical = 0
    noncanonical_lengths: set[int] = set()
    for requested, materialized, valid, batch_size in observations:
        by_bucket[requested].append((materialized, valid, batch_size))
        expected_batch = min(
            config.sampling.batch_size,
            config.sampling.batch_frame_budget // requested,
        )
        if materialized == requested and batch_size == expected_batch:
            canonical += 1
        else:
            noncanonical_lengths.add(materialized)
    count = len(observations)
    noncanonical_rate = {
        str(bucket): sum(materialized != bucket for materialized, _, _ in rows)
        / len(rows)
        for bucket, rows in sorted(by_bucket.items())
    }
    padding_ratio = {
        str(bucket): 1
        - sum(valid for _, valid, _ in rows)
        / sum(materialized * batch_size for materialized, _, batch_size in rows)
        for bucket, rows in sorted(by_bucket.items())
    }
    return {
        "canonical_shape_rate": canonical / count if count else 0.0,
        "unique_noncanonical_T_last_1000": sorted(noncanonical_lengths),
        "noncanonical_shape_rate_by_bucket": noncanonical_rate,
        "padding_ratio_by_bucket": padding_ratio,
        "shape_window_steps": count,
    }


def _base_learning_rate_scale(config: V4Config, step: int) -> float:
    if step < config.training.warmup_steps:
        return (step + 1) / config.training.warmup_steps
    if config.training.learning_rate_schedule == "constant_after_warmup":
        return 1.0
    progress = min(
        1.0,
        (step - config.training.warmup_steps)
        / max(1, config.training.max_steps - config.training.warmup_steps),
    )
    minimum_ratio = config.training.min_learning_rate_ratio
    if minimum_ratio is None:
        raise RuntimeError("cosine schedule has no minimum LR ratio")
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum_ratio + (1 - minimum_ratio) * cosine


def _build_optimizer(
    system: FlowMatchingSystem,
    config: V4Config,
) -> tuple[torch.optim.AdamW, dict[str, object]]:
    model = system.model
    if config.training.freeze_timestep_and_modulation:
        for parameter in model.time.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks:
            for parameter in block.modulation.parameters():
                parameter.requires_grad_(False)
        for parameter in model.final_modulation.parameters():
            parameter.requires_grad_(False)

    modules = dict(system.named_modules())
    assignments: dict[str, list[tuple[str, nn.Parameter]]] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "speaker": [],
    }
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name, parameter_name = name.rsplit(".", 1)
        module = modules[module_name]
        if module is model.speaker:
            if parameter_name != "weight":
                raise ValueError(f"unexpected speaker parameter: {name}")
            group = "speaker"
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if parameter_name == "weight":
                group = "backbone_decay"
            elif parameter_name == "bias":
                group = "backbone_no_decay"
            else:
                raise ValueError(f"unexpected projection parameter: {name}")
        else:
            raise ValueError(
                f"trainable parameter has no optimizer classification: {name} "
                f"({type(module).__name__})"
            )
        assignments[group].append((name, parameter))

    trainable = {
        id(parameter): name
        for name, parameter in system.named_parameters()
        if parameter.requires_grad
    }
    assigned = [
        (name, parameter)
        for values in assignments.values()
        for name, parameter in values
    ]
    assigned_ids = [id(parameter) for _, parameter in assigned]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("optimizer parameter groups contain duplicates")
    if set(assigned_ids) != set(trainable):
        missing = sorted(
            trainable[value] for value in set(trainable) - set(assigned_ids)
        )
        raise ValueError(f"optimizer parameter groups are incomplete: {missing[:5]}")
    if [name for name, _ in assignments["speaker"]] != ["model.speaker.weight"]:
        raise ValueError(
            "speaker optimizer group must contain only model.speaker.weight"
        )
    if not assignments["backbone_decay"] or not assignments["backbone_no_decay"]:
        raise ValueError("optimizer requires decay and no-decay backbone parameters")

    groups = [
        {
            "name": "backbone_decay",
            "params": [parameter for _, parameter in assignments["backbone_decay"]],
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        },
        {
            "name": "backbone_no_decay",
            "params": [parameter for _, parameter in assignments["backbone_no_decay"]],
            "lr": config.training.learning_rate,
            "weight_decay": 0.0,
        },
        {
            "name": "speaker",
            "params": [parameter for _, parameter in assignments["speaker"]],
            "lr": config.training.speaker_learning_rate,
            "weight_decay": 0.0,
        },
    ]
    optimizer_device = next(system.parameters()).device
    fused = config.performance.fused_adamw and optimizer_device.type == "cuda"
    optimizer = torch.optim.AdamW(groups, fused=fused)
    total = sum(parameter.numel() for parameter in system.parameters())
    trainable_parameters = sum(parameter.numel() for _, parameter in assigned)
    summary: dict[str, object] = {
        "total_parameters": total,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total - trainable_parameters,
        "groups": {
            group["name"]: {
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "parameter_tensors": len(assignments[group["name"]]),
                "parameter_names": [name for name, _ in assignments[group["name"]]],
                "base_learning_rate": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        },
    }
    summary["fused_adamw_requested"] = config.performance.fused_adamw
    summary["fused_adamw_active"] = fused
    return optimizer, summary


def _write_run_metadata(
    output: Path,
    config: V4Config,
    entries: list[ManifestEntry],
    mel_stats_path: Path,
    device: torch.device,
    num_workers: int,
    resume: Path | None,
    initialize_from: Path | None,
    initialization: dict[str, object] | None,
    optimizer_summary: dict[str, object],
    performance_runtime: dict[str, object],
    target_steps: int,
    performance_fork: dict[str, dict[str, object]] | None,
) -> None:
    path = output / "run_metadata.json"
    accepted = [entry for entry in entries if entry.quality_status == "accepted"]
    payload = {
        "schema_version": 3,
        "created_at_unix": time.time(),
        "config": asdict(config),
        "manifest_sha256": _manifest_digest(entries),
        "training_manifest_sha256": _manifest_digest(
            [entry for entry in accepted if entry.split == "train"]
        ),
        "recordings": len(entries),
        "accepted_recordings": len(accepted),
        "accepted_hours": sum(entry.duration_seconds for entry in accepted) / 3600,
        "speakers": len({entry.speaker_key for entry in accepted}),
        "datasets": sorted({entry.dataset for entry in accepted}),
        "mel_stats": mel_stats_path.resolve().as_posix(),
        "mel_stats_sha256": _file_digest(mel_stats_path),
        "device": str(device),
        "num_workers": num_workers,
        "resume_checkpoint": resume.resolve().as_posix() if resume else None,
        "initialize_from": (
            initialize_from.resolve().as_posix() if initialize_from else None
        ),
        "initialization": initialization,
        "optimizer": optimizer_summary,
        "performance": performance_runtime,
        "performance_fork": performance_fork,
        "stop_at_step": target_steps,
        "command": sys.argv,
        "source": _source_inventory(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            {
                "name": torch.cuda.get_device_name(device),
                "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            if device.type == "cuda"
            else None
        ),
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_fields = (
            "config",
            "manifest_sha256",
            "training_manifest_sha256",
            "mel_stats_sha256",
            "optimizer",
            "source",
        )
        mismatches = [
            name for name in stable_fields if existing.get(name) != payload.get(name)
        ]
        if mismatches:
            raise ValueError(
                "resume run identity differs from existing metadata: "
                + ", ".join(mismatches)
            )
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _source_inventory() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for directory in ("config", "rift_v4", "scripts", "third_party"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".json", ".py", ".sh"}
        )
    files = {
        path.relative_to(root).as_posix(): _file_digest(path)
        for path in sorted(set(paths))
        if path.is_file()
    }
    aggregate = hashlib.sha256(
        "\n".join(f"{name}\0{digest}" for name, digest in files.items()).encode()
    ).hexdigest()
    return {"sha256": aggregate, "files": files}


def _write_sampling_audit(
    output: Path, sampler: HierarchicalBatchSampler, config: V4Config
) -> None:
    path = output / "sampling-audit.json"
    payload = sampler.sampling_audit(
        max_steps=config.training.max_steps,
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        speaker_drop_probability=config.training.speaker_drop_probability,
    )
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("sampling audit differs from the existing run")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_validation_details(
    output: Path,
    step: int,
    ema: dict[str, object],
    online: dict[str, object],
) -> Path:
    directory = output / "validation"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{step:07d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"step": step, "ema": ema, "online": online},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _prepare_run_directory(output: Path, resume: Path | None) -> None:
    """Prevent a fresh run from inheriting identity records from an old run."""

    if resume is not None:
        return
    identity_files = (
        "run_metadata.json",
        "run_events.jsonl",
        "checkpoint_index.jsonl",
        "sampling-audit.json",
    )
    stale = [output / name for name in identity_files if (output / name).exists()]
    stale.extend(output.glob("*.pt"))
    stale.extend((output / "validation").glob("*.json"))
    if stale:
        names = ", ".join(path.name for path in stale[:5])
        suffix = "..." if len(stale) > 5 else ""
        raise ValueError(
            f"fresh run directory is not empty of run state: {names}{suffix}; "
            "choose a new directory or pass an explicit resume checkpoint"
        )


def _append_run_event(output: Path, event: str, details: dict[str, object]) -> None:
    payload = {"timestamp": time.time(), "event": event, **details}
    with (output / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
