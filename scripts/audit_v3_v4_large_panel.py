from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

from audit_checkpoint_pair_large_panel import paired, sample_metrics, summarize
from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset, SampleRequest
from rift_v4.features import MelStats
from rift_v4.manifest import load_manifest
from rift_v4.train import build_system


EULER_STEPS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_v3(source: Path, checkpoint_path: Path, device: torch.device):
    package = types.ModuleType("rift_svc")
    package.__path__ = [str(source.resolve() / "rift_svc")]
    sys.modules["rift_svc"] = package
    from rift_svc.dit import DiT

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint["hyper_parameters"]["cfg"]
    model = DiT(num_speaker=1, **config["model"])
    prefix = "model.transformer."
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["spk_embed.weight"] or unexpected:
        raise RuntimeError(
            f"V3 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    del checkpoint
    return model.to(device).eval()


def load_locked_panel(args, config, stats, entries, speaker_to_id):
    reference = json.loads(args.panel_reference.read_text())
    metadata = reference["samples"]
    frames = int(reference["protocol"]["frames"])
    seed = int(reference["protocol"]["seed"])
    dataset = FeatureDataset(
        entries,
        config.mel.channels,
        config.model.content_dim,
        stats,
        speaker_to_id,
        voiced_crop_probability=0.0,
    )
    items = []
    for row in metadata:
        index = int(row["manifest_index"])
        if entries[index].id != row["entry_id"]:
            raise RuntimeError(
                f"panel lock mismatch at manifest index {index}: "
                f"{entries[index].id!r} != {row['entry_id']!r}"
            )
        items.append(dataset[SampleRequest(index, frames, 2026 + index)])
    tensors = {
        name: torch.stack([item[name] for item in items])
        for name in ("mel", "content", "f0", "rms", "speaker")
    }
    generator = torch.Generator().manual_seed(seed)
    tensors["noise"] = torch.randn(tensors["mel"].shape, generator=generator)
    return metadata, tensors, frames, seed


def append_metrics(output, steps, prediction, target):
    values = sample_metrics(prediction, target)
    row = output[str(steps)]
    for name, samples in values.items():
        row[name].extend(samples)


def evaluate_v3(model, tensors, stats, frames, batch_size, device):
    output = {
        str(steps): {"raw_mse": [], "raw_l1": [], "raw_cosine": []}
        for steps in EULER_STEPS
    }
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for begin in range(0, len(tensors["mel"]), batch_size):
            end = min(begin + batch_size, len(tensors["mel"]))
            raw_target = stats.denormalize(tensors["mel"][begin:end].to(device).float())
            mel = (raw_target + 12.0) / 7.0 - 1.0
            content = tensors["content"][begin:end].to(device)
            f0 = tensors["f0"][begin:end].to(device).squeeze(-1)
            rms = tensors["rms"][begin:end].to(device).squeeze(-1)
            noise = tensors["noise"][begin:end].to(device)
            speaker = torch.zeros(end - begin, dtype=torch.long, device=device)
            mask = torch.ones((end - begin, frames), dtype=torch.bool, device=device)
            for steps in EULER_STEPS:
                state = noise.clone()
                times = torch.linspace(0.0, 1.0, steps + 1, device=device)
                for index in range(steps):
                    state += (times[index + 1] - times[index]) * model(
                        x=state,
                        spk=speaker,
                        f0=f0,
                        rms=rms,
                        cvec=content,
                        time=times[index].expand(end - begin),
                        mask=mask,
                        drop_speaker=True,
                    ).float()
                append_metrics(output, steps, (state + 1.0) * 7.0 - 12.0, raw_target)
            print(json.dumps({"model": "v3_null", "completed": end}), flush=True)
    return output


def evaluate_v4(model, tensors, stats, frames, batch_size, device):
    outputs = {
        condition: {
            str(steps): {"raw_mse": [], "raw_l1": [], "raw_cosine": []}
            for steps in EULER_STEPS
        }
        for condition in ("null", "correct")
    }
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for begin in range(0, len(tensors["mel"]), batch_size):
            end = min(begin + batch_size, len(tensors["mel"]))
            mel = tensors["mel"][begin:end].to(device)
            raw_target = stats.denormalize(mel.float())
            content = tensors["content"][begin:end].to(device)
            f0 = tensors["f0"][begin:end].to(device)
            rms = tensors["rms"][begin:end].to(device)
            noise = tensors["noise"][begin:end].to(device)
            correct = tensors["speaker"][begin:end].to(device)
            conditions = {
                "null": torch.full_like(correct, model.null_speaker_id),
                "correct": correct,
            }
            mask = torch.ones((end - begin, frames), dtype=torch.bool, device=device)
            for condition, speaker in conditions.items():
                for steps in EULER_STEPS:
                    state = noise.clone()
                    times = torch.linspace(0.0, 1.0, steps + 1, device=device)
                    for index in range(steps):
                        state += (times[index + 1] - times[index]) * model(
                            state,
                            content,
                            f0,
                            rms,
                            speaker,
                            times[index].expand(end - begin),
                            mask,
                        ).float()
                    append_metrics(
                        outputs[condition],
                        steps,
                        stats.denormalize(state.float()),
                        raw_target,
                    )
            print(json.dumps({"model": "v4", "completed": end}), flush=True)
    return outputs


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = V4Config.load(args.v4_config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    v4_checkpoint = torch.load(
        args.v4_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    metadata, tensors, frames, seed = load_locked_panel(
        args,
        config,
        stats,
        entries,
        v4_checkpoint["speaker_to_id"],
    )

    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    models = {"v3_null": evaluate_v3(
        v3, tensors, stats, frames, args.batch_size, device
    )}
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    system = build_system(config, len(v4_checkpoint["speaker_to_id"])).to(device).eval()
    system.load_state_dict(v4_checkpoint["ema"], strict=True)
    v4_outputs = evaluate_v4(
        system.model, tensors, stats, frames, args.batch_size, device
    )
    models["v4_null"] = v4_outputs["null"]
    models["v4_correct"] = v4_outputs["correct"]

    summaries = {
        name: {
            steps: {metric: summarize(values) for metric, values in row.items()}
            for steps, row in outputs.items()
        }
        for name, outputs in models.items()
    }
    comparisons = {}
    for left_name, right_name in (
        ("v3_null", "v4_null"),
        ("v3_null", "v4_correct"),
        ("v4_null", "v4_correct"),
    ):
        comparison_name = f"{right_name}_minus_{left_name}"
        comparisons[comparison_name] = {}
        for steps in EULER_STEPS:
            key = str(steps)
            comparisons[comparison_name][key] = {
                metric: paired(
                    models[left_name][key][metric],
                    models[right_name][key][metric],
                    metadata,
                    args.bootstrap_samples,
                    seed + steps,
                    higher_is_better=metric == "raw_cosine",
                )
                for metric in ("raw_mse", "raw_l1", "raw_cosine")
            }

    payload = {
        "schema_version": 1,
        "protocol": {
            "panel_reference": str(args.panel_reference),
            "panel_size": len(metadata),
            "unique_song_units": len({row["song_key"] for row in metadata}),
            "physical_speakers": len({row["speaker"] for row in metadata}),
            "frames": frames,
            "noise_seed": seed,
            "noise": "identical standard Gaussian tensor in each model's native normalized space",
            "solver": "linear Euler",
            "euler_steps": EULER_STEPS,
            "evaluation_space": "inverse-normalized raw log-mel",
            "v3_condition": "published trained null speaker embedding",
            "v4_state": "EMA",
            "v4_conditions": ["null", "correct physical speaker"],
            "bootstrap": "song-grouped, 10000 resamples",
        },
        "checkpoints": {
            "v3": str(args.v3_checkpoint),
            "v4": str(args.v4_checkpoint),
        },
        "samples": metadata,
        "summaries": summaries,
        "comparisons": comparisons,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summaries": summaries}), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
