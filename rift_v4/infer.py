from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import V4Config
from .features import MelStats
from .train import build_system


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a V4 mel with Heun ODE integration"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--rms", type=Path, required=True)
    parser.add_argument(
        "--speaker", required=True, help="namespaced DATASET:SPEAKER key"
    )
    parser.add_argument("--output-mel", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=1.2)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--time-schedule", choices=("linear", "cosine"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-frames", type=int)
    parser.add_argument("--overlap-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 4:
        raise ValueError("checkpoint does not use schema 4")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("checkpoint configuration differs from the inference config")
    speaker_to_id = checkpoint["speaker_to_id"]
    if args.speaker not in speaker_to_id:
        raise ValueError(f"unknown speaker {args.speaker!r}")
    system = build_system(config, len(speaker_to_id))
    system.load_state_dict(checkpoint.get("ema", checkpoint["model"]), strict=True)
    device = torch.device(args.device)
    system.to(device).eval()
    content = _matrix(
        torch.load(args.content, map_location="cpu", weights_only=True),
        config.model.content_dim,
    )
    f0 = _vector(torch.load(args.f0, map_location="cpu", weights_only=True))
    rms = _vector(torch.load(args.rms, map_location="cpu", weights_only=True))
    frames = f0.shape[0]
    content = _resize(content, frames)[None].to(device)
    f0 = f0[None].to(device)
    rms = _resize(rms, frames)[None].to(device)
    mask = torch.ones(1, frames, dtype=torch.bool, device=device)
    speaker = torch.tensor([speaker_to_id[args.speaker]], device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    normalized = sample_chunked(
        system,
        content,
        f0,
        rms,
        speaker,
        mask,
        config.mel.channels,
        args.steps,
        args.guidance,
        args.method,
        generator,
        args.time_schedule or config.inference.time_schedule,
        args.chunk_frames or config.sampling.frame_buckets[-1],
        args.overlap_frames,
    )
    stats_payload = checkpoint.get("mel_stats")
    if stats_payload is None:
        raise ValueError("checkpoint does not embed mel statistics")
    stats = MelStats(
        tuple(stats_payload["mean"]),
        tuple(stats_payload["std"]),
        int(stats_payload["frames"]),
    )
    mel = stats.denormalize(normalized)
    _atomic_save(mel, args.output_mel)
    print(f"wrote {tuple(mel.shape)} log-mel to {args.output_mel}")


@torch.inference_mode()
def sample_chunked(
    system,
    content: torch.Tensor,
    f0: torch.Tensor,
    rms: torch.Tensor,
    speaker: torch.Tensor,
    mask: torch.Tensor,
    mel_channels: int,
    steps: int,
    guidance: float,
    method: str,
    generator: torch.Generator,
    time_schedule: str,
    chunk_frames: int,
    overlap_frames: int,
) -> torch.Tensor:
    """Sample bounded-context chunks and crossfade their normalized mels."""

    if content.shape[0] != 1:
        raise ValueError("chunked inference currently requires batch size one")
    if chunk_frames <= 0 or not 0 <= overlap_frames < chunk_frames:
        raise ValueError("overlap frames must be non-negative and smaller than chunk")
    frames = content.shape[1]
    noise = torch.randn(
        1,
        frames,
        mel_channels,
        device=content.device,
        dtype=content.dtype,
        generator=generator,
    )
    if frames <= chunk_frames:
        return (
            system.sample(
                content,
                f0,
                rms,
                speaker,
                mask,
                mel_channels,
                steps,
                guidance,
                method,
                time_schedule=time_schedule,
                initial_noise=noise,
            )[0]
            .float()
            .cpu()
        )

    stride = chunk_frames - overlap_frames
    output = torch.zeros(frames, mel_channels, device=content.device)
    weight_sum = torch.zeros(frames, 1, device=content.device)
    for start in range(0, frames, stride):
        stop = min(frames, start + chunk_frames)
        actual = stop - start
        padding = chunk_frames - actual
        chunk_mask = F.pad(mask[:, start:stop], (0, padding), value=False)
        chunk = system.sample(
            F.pad(content[:, start:stop], (0, 0, 0, padding)),
            F.pad(f0[:, start:stop], (0, 0, 0, padding)),
            F.pad(rms[:, start:stop], (0, 0, 0, padding)),
            speaker,
            chunk_mask,
            mel_channels,
            steps,
            guidance,
            method,
            time_schedule=time_schedule,
            initial_noise=F.pad(noise[:, start:stop], (0, 0, 0, padding)),
        )[0, :actual].float()
        weights = torch.ones(actual, 1, device=content.device)
        fade = min(overlap_frames, actual)
        if fade and start > 0:
            weights[:fade] = torch.linspace(0, 1, fade + 2, device=content.device)[
                1:-1, None
            ]
        if fade and stop < frames:
            weights[-fade:] = torch.minimum(
                weights[-fade:],
                torch.linspace(1, 0, fade + 2, device=content.device)[1:-1, None],
            )
        output[start:stop] += chunk * weights
        weight_sum[start:stop] += weights
        if stop == frames:
            break
    return (output / weight_sum.clamp_min(1e-8)).cpu()


def _matrix(value: torch.Tensor, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).float().squeeze()
    if tensor.ndim != 2:
        raise ValueError("content must be rank two")
    if tensor.shape[-1] == width:
        return tensor
    if tensor.shape[0] == width:
        return tensor.transpose(0, 1)
    raise ValueError("content width mismatch")


def _vector(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).float().squeeze()
    if tensor.ndim != 1:
        raise ValueError("feature must be rank one")
    return tensor[:, None]


def _resize(value: torch.Tensor, frames: int) -> torch.Tensor:
    if value.shape[0] == frames:
        return value
    return F.interpolate(
        value.transpose(0, 1)[None], size=frames, mode="linear", align_corners=False
    )[0].transpose(0, 1)


def _atomic_save(value: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
