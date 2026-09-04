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

from .config import V4Config
from .data import FeatureDataset, build_sampler
from .features import MelStats
from .manifest import load_manifest
from .train import _validation_loss, build_system


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate last-block attention and FFN residual interventions"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state", choices=("ema", "model"), default="ema")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    args = parser.parse_args()
    run_sweep(
        args.config,
        args.manifest,
        args.mel_stats,
        args.checkpoint,
        args.output,
        torch.device(args.device),
        args.batch_size,
        args.state,
        tuple(args.alphas),
    )


@torch.inference_mode()
def run_sweep(
    config_path: Path,
    manifest_path: Path,
    mel_stats_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    state: str,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not alphas or any(not math.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be finite and non-negative")

    config = V4Config.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("checkpoint does not use schema 4")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint configuration differs from sweep configuration")
    if state not in checkpoint:
        raise ValueError(f"checkpoint has no {state} state")

    entries = load_manifest(manifest_path)
    train_entries = [
        entry
        for entry in entries
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    speaker_to_id = checkpoint["speaker_to_id"]
    validation_entries = [
        entry
        for entry in entries
        if entry.split == "validation"
        and entry.quality_status == "accepted"
        and entry.speaker_key in speaker_to_id
    ]
    if not validation_entries:
        raise ValueError("residual sweep requires accepted validation entries")

    dataset = FeatureDataset(
        validation_entries,
        config.mel.channels,
        config.model.content_dim,
        MelStats.load(mel_stats_path, config.mel.channels),
        speaker_to_id,
        voiced_crop_probability=config.sampling.voiced_crop_probability,
    )
    sampler = build_sampler(train_entries, config.sampling)
    validation_weights = dict(
        zip(sampler.datasets, sampler.dataset_probabilities, strict=True)
    )
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    system.load_state_dict(checkpoint[state], strict=True)
    last_block = system.model.blocks[-1]

    variants = []
    seen = set()
    for group in ("attention", "ffn", "joint"):
        for alpha in alphas:
            scales = {
                "attention": (alpha, 1.0),
                "ffn": (1.0, alpha),
                "joint": (alpha, alpha),
            }[group]
            if scales not in seen:
                seen.add(scales)
                variants.append(scales)

    results: dict[str, object] = {}
    started = time.time()
    for attention_scale, ffn_scale in variants:
        last_block.attention_residual_scale = attention_scale
        last_block.ffn_residual_scale = ffn_scale
        key = f"attention={attention_scale:g},ffn={ffn_scale:g}"
        result = _validation_loss(
            system,
            dataset,
            device,
            config.sampling.frame_buckets[-1],
            batch_size,
            validation_weights,
            config.sampling.synthetic_datasets,
            config.evaluation.validation_recordings_per_song,
        )
        results[key] = result
        payload = {
            "schema_version": 1,
            "checkpoint": checkpoint_path.resolve().as_posix(),
            "checkpoint_step": int(checkpoint["step"]),
            "state": state,
            "last_block": len(system.model.blocks) - 1,
            "alphas": alphas,
            "validation_protocol": 6,
            "batch_size": batch_size,
            "started_at_unix": started,
            "updated_at_unix": time.time(),
            "results": results,
        }
        _write_json_atomic(output_path, payload)
        print(
            json.dumps(
                {
                    "variant": key,
                    "real_speaker_macro_flow": result["real_speaker_macro_flow"],
                    "real_dataset_macro_flow": result["real_dataset_macro_flow"],
                    "mixture_weighted_flow": result["mixture_weighted_flow"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
