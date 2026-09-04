from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import V4Config
from .features import MelStats
from .manifest import load_manifest
from .shadow_panel import (
    _load_checkpoint,
    _optional_float,
    _per_sample_masked_mean,
    file_sha256,
    load_locked_tensors,
    load_or_create_panel_lock,
    speaker_map_sha256,
)
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official V3 and V4 on the sealed shadow panel"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mel-stats", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=(256, 512, 768))
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--catastrophe-threshold", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames_values = tuple(sorted(set(args.frames)))
    if not frames_values or min(frames_values) <= 0:
        parser.error("--frames must contain positive values")
    config = V4Config.load(args.config)
    stats = MelStats.load(args.mel_stats, config.mel.channels)
    entries = load_manifest(args.manifest)
    v4_checkpoint = _load_checkpoint(args.v4_checkpoint, config)
    speaker_to_id = v4_checkpoint["speaker_to_id"]
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
    if max(frames_values) > int(lock["samples"][0]["maximum_frames"]):
        parser.error("requested frame length exceeds the locked crop")
    tensors = load_locked_tensors(lock, entries, config, stats, speaker_to_id)
    device = torch.device(args.device)
    maximum_frames = max(frames_values)
    generator = torch.Generator().manual_seed(int(lock["protocol"]["seed"]))
    noise = torch.randn(
        len(lock["samples"]),
        maximum_frames,
        config.mel.channels,
        generator=generator,
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "protocol": {
            "panel_lock": str(args.panel_lock),
            "panel_lock_sha256": file_sha256(args.panel_lock),
            "samples": len(lock["samples"]),
            "song_units": lock["coverage"]["unique_song_units"],
            "physical_speakers": lock["coverage"]["physical_speakers"],
            "frames": list(frames_values),
            "nested_crop": "256 and 512 are prefixes of the locked 768 crop",
            "noise": "same numeric Gaussian tensor prefix in native normalized space",
            "noise_seed": int(lock["protocol"]["seed"]),
            "solver": "linear Euler",
            "intervals": args.intervals,
            "silence_threshold_rms": 1e-3,
            "catastrophe_threshold_raw_mse": args.catastrophe_threshold,
            "bootstrap": "dataset-song grouped",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "checkpoints": {
            "v3": str(args.v3_checkpoint),
            "v4": str(args.v4_checkpoint),
            "v4_step": int(v4_checkpoint["step"]),
            "v4_state": "ema",
        },
        "coverage": lock["coverage"],
        "models": {},
        "comparisons": {},
    }

    v3 = load_v3(args.v3_source, args.v3_checkpoint, device)
    payload["models"]["v3_null"] = evaluate_model(
        v3,
        "v3_null",
        tensors,
        lock["samples"],
        stats,
        noise,
        frames_values,
        args.intervals,
        args.batch_size,
        args.catastrophe_threshold,
        device,
    )
    _atomic_json(args.output, payload)
    del v3
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if speaker_map_sha256(speaker_to_id) != lock["source"]["speaker_to_id_sha256"]:
        raise ValueError("V4 speaker mapping differs from shadow lock")
    system = build_system(config, len(speaker_to_id)).to(device).eval()
    system.load_state_dict(v4_checkpoint["ema"], strict=True)
    del v4_checkpoint
    for condition in ("null", "correct"):
        name = f"v4_{condition}"
        payload["models"][name] = evaluate_model(
            system.model,
            name,
            tensors,
            lock["samples"],
            stats,
            noise,
            frames_values,
            args.intervals,
            args.batch_size,
            args.catastrophe_threshold,
            device,
        )
        _atomic_json(args.output, payload)

    for left, right in (
        ("v3_null", "v4_null"),
        ("v3_null", "v4_correct"),
        ("v4_null", "v4_correct"),
    ):
        payload["comparisons"][f"{right}_minus_{left}"] = compare_models(
            payload["models"][left],
            payload["models"][right],
            lock["samples"],
            args.bootstrap_samples,
            int(lock["protocol"]["seed"]),
        )
    payload["status"] = "complete"
    _atomic_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


def load_v3(source: Path, checkpoint_path: Path, device: torch.device):
    package = types.ModuleType("rift_svc")
    package.__path__ = [str(source.resolve() / "rift_svc")]
    sys.modules["rift_svc"] = package
    from rift_svc.dit import DiT

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model = DiT(num_speaker=1, **checkpoint["hyper_parameters"]["cfg"]["model"])
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
    return model.to(device).eval()


def evaluate_model(
    model,
    model_name: str,
    tensors: dict[str, torch.Tensor],
    metadata: list[dict[str, object]],
    stats: MelStats,
    noise: torch.Tensor,
    frames_values: tuple[int, ...],
    intervals: int,
    batch_size: int,
    catastrophe_threshold: float,
    device: torch.device,
) -> dict[str, object]:
    result = {}
    for frames in frames_values:
        rows = []
        for begin in range(0, len(metadata), batch_size):
            end = min(begin + batch_size, len(metadata))
            v4_mel = tensors["mel"][begin:end, :frames].to(device)
            raw_target = stats.denormalize(v4_mel.float())
            content = tensors["content"][begin:end, :frames].to(device)
            f0 = tensors["f0"][begin:end, :frames].to(device)
            rms = tensors["rms"][begin:end, :frames].to(device)
            state = noise[begin:end, :frames].to(device).clone()
            mask = torch.ones(end - begin, frames, dtype=torch.bool, device=device)
            times = torch.linspace(0.0, 1.0, intervals + 1, device=device)
            with torch.inference_mode(), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                for index in range(intervals):
                    timestep = times[index].expand(end - begin)
                    if model_name == "v3_null":
                        velocity = model(
                            x=state,
                            spk=torch.zeros(
                                end - begin, dtype=torch.long, device=device
                            ),
                            f0=f0.squeeze(-1),
                            rms=rms.squeeze(-1),
                            cvec=content,
                            time=timestep,
                            mask=mask,
                            drop_speaker=True,
                        )
                    else:
                        correct = tensors["speaker"][begin:end].to(device)
                        speaker = (
                            torch.full_like(correct, model.null_speaker_id)
                            if model_name == "v4_null"
                            else correct
                        )
                        velocity = model(
                            state, content, f0, rms, speaker, timestep, mask
                        )
                    state += (times[index + 1] - times[index]) * velocity.float()
                raw_prediction = (
                    (state + 1.0) * 7.0 - 12.0
                    if model_name == "v3_null"
                    else stats.denormalize(state.float())
                )
                rows.extend(
                    metric_rows(
                        raw_prediction,
                        raw_target,
                        rms,
                        metadata[begin:end],
                    )
                )
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "frames": frames,
                        "completed": end,
                        "total": len(metadata),
                    }
                ),
                flush=True,
            )
        result[str(frames)] = {
            "summary": summarize_rows(rows, metadata, catastrophe_threshold),
            "samples": rows,
        }
    return result


def metric_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rms: torch.Tensor,
    metadata: list[dict[str, object]],
) -> list[dict[str, object]]:
    residual = prediction - target
    frame_mse = residual.square().mean(-1)
    active = rms.squeeze(-1) > 1e-3
    active_mse = _per_sample_masked_mean(frame_mse, active)
    silence_mse = _per_sample_masked_mean(frame_mse, ~active)
    l1 = residual.abs().mean((1, 2))
    cosine = F.cosine_similarity(prediction.flatten(1), target.flatten(1), dim=1)
    rows = []
    for index, item in enumerate(metadata):
        rows.append(
            {
                "ordinal": int(item["ordinal"]),
                "entry_id": item["entry_id"],
                "full_raw_mse": float(frame_mse[index].mean()),
                "active_raw_mse": _optional_float(active_mse[index]),
                "silence_raw_mse": _optional_float(silence_mse[index]),
                "raw_l1": float(l1[index]),
                "raw_cosine": float(cosine[index]),
                "active_frames": int(active[index].sum()),
                "silence_frames": int((~active[index]).sum()),
            }
        )
    return rows


def summarize_rows(
    rows: list[dict[str, object]],
    metadata: list[dict[str, object]],
    catastrophe_threshold: float,
) -> dict[str, object]:
    result = {}
    for metric in (
        "full_raw_mse",
        "active_raw_mse",
        "silence_raw_mse",
        "raw_l1",
        "raw_cosine",
    ):
        selected = [
            (float(row[metric]), item)
            for row, item in zip(rows, metadata, strict=True)
            if row[metric] is not None
        ]
        values = torch.tensor([value for value, _ in selected], dtype=torch.float64)
        by_speaker: dict[str, list[float]] = defaultdict(list)
        by_dataset: dict[str, list[float]] = defaultdict(list)
        for value, item in selected:
            by_speaker[str(item["speaker"])].append(value)
            by_dataset[str(item["dataset"])].append(value)
        row = {
            "samples": len(values),
            "mean": float(values.mean()),
            "median": float(torch.quantile(values, 0.5)),
            "p90": float(torch.quantile(values, 0.9)),
            "p95": float(torch.quantile(values, 0.95)),
            "speaker_macro": _macro_mean(by_speaker),
            "dataset_macro": _macro_mean(by_dataset),
        }
        if metric.endswith("raw_mse"):
            row["catastrophe_rate"] = float(
                (values > catastrophe_threshold).double().mean()
            )
        result[metric] = row
    return result


def compare_models(
    before: dict[str, object],
    after: dict[str, object],
    metadata: list[dict[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    result = {}
    for frame_index, frames in enumerate(sorted(before, key=int)):
        result[frames] = {}
        for metric_index, metric in enumerate(
            (
                "full_raw_mse",
                "active_raw_mse",
                "silence_raw_mse",
                "raw_l1",
                "raw_cosine",
            )
        ):
            result[frames][metric] = paired_statistics(
                before[frames]["samples"],
                after[frames]["samples"],
                metadata,
                metric,
                bootstrap_samples,
                seed + frame_index * 100 + metric_index,
            )
    return result


def paired_statistics(
    before: list[dict[str, object]],
    after: list[dict[str, object]],
    metadata: list[dict[str, object]],
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    selected = [
        (float(right[metric]) - float(left[metric]), item)
        for left, right, item in zip(before, after, metadata, strict=True)
        if left[metric] is not None and right[metric] is not None
    ]
    deltas = torch.tensor([value for value, _ in selected], dtype=torch.float64)
    by_song: dict[str, list[float]] = defaultdict(list)
    by_speaker: dict[str, list[float]] = defaultdict(list)
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for value, item in selected:
        by_song[str(item["song_key"])].append(value)
        by_speaker[str(item["speaker"])].append(value)
        by_dataset[str(item["dataset"])].append(value)
    song_means = torch.tensor(
        [sum(values) / len(values) for values in by_song.values()],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        len(song_means),
        (bootstrap_samples, len(song_means)),
        generator=generator,
    )
    interval = torch.quantile(
        song_means[draws].mean(1),
        torch.tensor((0.025, 0.975), dtype=torch.float64),
    )
    higher_is_better = metric == "raw_cosine"
    wins = deltas > 0 if higher_is_better else deltas < 0
    return {
        "definition": "after minus before",
        "samples": len(deltas),
        "song_units": len(song_means),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(torch.quantile(deltas, 0.5)),
        "win_rate_after": float(wins.double().mean()),
        "song_bootstrap_mean_delta_95_ci": [float(value) for value in interval],
        "speaker_macro_delta": _macro_mean(by_speaker),
        "dataset_macro_delta": _macro_mean(by_dataset),
    }


def _macro_mean(grouped: dict[str, list[float]]) -> float:
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
