from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

from .config import V4Config
from .manifest import ManifestEntry, load_manifest


class PitchModel(Protocol):
    def infer(self, audio: Tensor, **kwargs: object) -> Tensor: ...


@dataclass(frozen=True)
class MelStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    frames: int

    @classmethod
    def load(cls, path: str | Path, channels: int) -> MelStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(
            tuple(payload["mean"]), tuple(payload["std"]), int(payload["frames"])
        )
        if len(result.mean) != channels or len(result.std) != channels:
            raise ValueError("mel statistics do not match channel count")
        if result.frames <= 0 or any(value <= 0 for value in result.std):
            raise ValueError("invalid mel statistics")
        return result

    def normalize(self, mel: Tensor) -> Tensor:
        mean = mel.new_tensor(self.mean)
        std = mel.new_tensor(self.std)
        return (mel - mean) / std

    def denormalize(self, mel: Tensor) -> Tensor:
        mean = mel.new_tensor(self.mean)
        std = mel.new_tensor(self.std)
        return mel * std + mean


class ExactMelExtractor:
    """The single mel implementation shared by training and vocoder tooling."""

    def __init__(self, config: V4Config, device: torch.device) -> None:
        self.config = config
        mel = config.mel
        if mel.mel_scale != "slaney" or mel.mel_norm != "slaney":
            raise ValueError("only the fixed Slaney contract is supported")
        self.window = torch.hann_window(mel.win_length, device=device)
        self.filterbank = torch.from_numpy(
            librosa.filters.mel(
                sr=config.sample_rate,
                n_fft=mel.n_fft,
                n_mels=mel.channels,
                fmin=mel.fmin,
                fmax=mel.fmax,
                htk=False,
                norm="slaney",
                dtype="float32",
            ).T
        ).to(device)

    def __call__(self, waveform: Tensor) -> Tensor:
        mel = self.config.mel
        pad = (mel.n_fft - self.config.hop_length) // 2
        padded = F.pad(waveform[None, None], (pad, pad), mode=mel.pad_mode)[0, 0]
        spectrum = torch.stft(
            padded,
            n_fft=mel.n_fft,
            hop_length=self.config.hop_length,
            win_length=mel.win_length,
            window=self.window,
            center=mel.center,
            return_complex=True,
        ).abs()
        if mel.power != 1.0:
            spectrum = spectrum.pow(mel.power)
        return (
            (spectrum.transpose(0, 1) @ self.filterbank).clamp_min(mel.log_clamp).log()
        )


def extract_auxiliary_features(
    waveform: Tensor,
    config: V4Config,
    pitch_model: PitchModel,
) -> tuple[Tensor, Tensor, Tensor]:
    mel = ExactMelExtractor(config, waveform.device)(waveform)
    frames = mel.shape[0]
    window = config.mel.win_length
    pad = (window - config.hop_length) // 2
    padded = F.pad(waveform[None, None], (pad, pad), mode="reflect")[0, 0]
    rms = padded.unfold(0, window, config.hop_length).square().mean(-1).sqrt()
    rms = _resize_vector(rms, frames)
    # TorchFCPE treats excursions outside [-1, 1] as an input error. Keep the
    # exact waveform for mel/RMS, but bound only the pitch estimator input.
    f0 = pitch_model.infer(
        waveform.clamp(-1.0, 1.0)[None, :, None],
        sr=config.sample_rate,
        decoder_mode="local_argmax",
        threshold=0.006,
        f0_min=40,
        f0_max=1600,
        interp_uv=False,
        output_interp_target_length=frames,
    )
    f0 = _resize_vector(torch.as_tensor(f0).squeeze(), frames)
    return mel.float().cpu(), f0.float().cpu(), rms.float().cpu()


def compute_mel_stats(entries: list[ManifestEntry], channels: int) -> MelStats:
    count = 0
    total = torch.zeros(channels, dtype=torch.float64)
    square = torch.zeros(channels, dtype=torch.float64)
    for entry in entries:
        if entry.split != "train" or entry.quality_status != "accepted":
            continue
        mel = torch.as_tensor(
            torch.load(
                f"{entry.feature_prefix}.mel.pt", map_location="cpu", weights_only=True
            )
        ).float()
        if mel.ndim == 2 and mel.shape[0] == channels:
            mel = mel.transpose(0, 1)
        if mel.ndim != 2 or mel.shape[1] != channels:
            raise ValueError(f"{entry.id}: invalid mel shape {tuple(mel.shape)}")
        if not bool(torch.isfinite(mel).all()):
            raise ValueError(f"{entry.id}: mel contains NaN or Inf")
        count += mel.shape[0]
        total += mel.double().sum(0)
        square += mel.double().square().sum(0)
    if count == 0:
        raise ValueError("no accepted training mel features")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    return MelStats(tuple(mean.tolist()), tuple(variance.sqrt().tolist()), count)


def merge_mel_stats(paths: list[Path]) -> MelStats:
    if not paths:
        raise ValueError("at least one mel-stat shard is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    channels = len(payloads[0]["mean"])
    total_frames = 0
    total = torch.zeros(channels, dtype=torch.float64)
    square = torch.zeros(channels, dtype=torch.float64)
    for payload in payloads:
        frames = int(payload["frames"])
        mean = torch.tensor(payload["mean"], dtype=torch.float64)
        std = torch.tensor(payload["std"], dtype=torch.float64)
        if frames <= 0 or len(mean) != channels or len(std) != channels:
            raise ValueError("invalid mel-stat shard")
        total_frames += frames
        total += frames * mean
        square += frames * (std.square() + mean.square())
    mean = total / total_frames
    variance = (square / total_frames - mean.square()).clamp_min(1e-12)
    return MelStats(tuple(mean.tolist()), tuple(variance.sqrt().tolist()), total_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract exact V4 mel/F0/RMS features")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", type=Path, default=Path("config/v4.json"))
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--execute-extraction", action="store_true")
    stats = subparsers.add_parser("stats")
    stats.add_argument("--config", type=Path, default=Path("config/v4.json"))
    stats.add_argument("--manifest", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--num-shards", type=int, default=1)
    stats.add_argument("--shard-index", type=int, default=0)
    merge = subparsers.add_parser("merge-stats")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "merge-stats":
        result = merge_mel_stats(args.input)
        _atomic_json(
            {"mean": result.mean, "std": result.std, "frames": result.frames},
            args.output,
        )
        print(f"merged {len(args.input)} mel-stat shards into {args.output}")
        return
    config = V4Config.load(args.config)
    entries = load_manifest(args.manifest)
    if args.command == "stats":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        if args.num_shards > 1:
            entries = [
                entry
                for index, entry in enumerate(entries)
                if index % args.num_shards == args.shard_index
            ]
        result = compute_mel_stats(entries, config.mel.channels)
        _atomic_json(
            {"mean": result.mean, "std": result.std, "frames": result.frames},
            args.output,
        )
        print(f"wrote {args.output} from {result.frames} training frames")
        return
    if not args.execute_extraction:
        print(
            f"validated {len(entries)} recordings; pass --execute-extraction "
            "to write tensors"
        )
        return
    try:
        from torchfcpe import spawn_bundled_infer_model
    except ImportError as error:
        raise RuntimeError(
            "install the 'features' extra to use pinned TorchFCPE"
        ) from error
    device = torch.device(args.device)
    pitch_model = spawn_bundled_infer_model(device=args.device)
    total = sum(entry.quality_status == "accepted" for entry in entries)
    completed = 0
    extracted = 0
    for entry in entries:
        if entry.quality_status != "accepted":
            continue
        prefix = Path(entry.feature_prefix)
        destinations = (
            Path(f"{prefix}.mel.pt"),
            Path(f"{prefix}.f0.pt"),
            Path(f"{prefix}.rms.pt"),
        )
        if _feature_triplet_is_valid(destinations, config.mel.channels):
            completed += 1
            if completed % 100 == 0:
                _print_progress("aux_features", completed, total, extracted)
            continue
        audio, rate = sf.read(entry.audio_path, dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError(
                f"{entry.id}: feature extraction requires mono vocal audio"
            )
        waveform = torch.from_numpy(audio[:, 0]).to(device)
        if rate != config.sample_rate:
            waveform = AF.resample(waveform, rate, config.sample_rate)
        mel, f0, rms = extract_auxiliary_features(waveform, config, pitch_model)
        _atomic_tensor(mel, destinations[0])
        _atomic_tensor(f0, destinations[1])
        _atomic_tensor(rms, destinations[2])
        completed += 1
        extracted += 1
        if completed % 100 == 0:
            _print_progress("aux_features", completed, total, extracted)
    _print_progress("aux_features", completed, total, extracted)


def _print_progress(stage: str, completed: int, total: int, written: int) -> None:
    print(
        json.dumps(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "written": written,
            }
        ),
        flush=True,
    )


def _feature_triplet_is_valid(
    destinations: tuple[Path, Path, Path], mel_channels: int
) -> bool:
    if not all(destination.is_file() for destination in destinations):
        return False
    try:
        mel, f0, rms = (
            torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
            for destination in destinations
        )
    except (OSError, RuntimeError, ValueError):
        return False
    vectors_are_valid = all(
        value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1)
        for value in (f0, rms)
    )
    return bool(
        mel.ndim == 2
        and mel.shape[1] == mel_channels
        and vectors_are_valid
        and mel.shape[0] == f0.shape[0] == rms.shape[0]
        and mel.shape[0] > 0
        and bool(torch.isfinite(mel).all())
        and bool(torch.isfinite(f0).all())
        and bool(torch.isfinite(rms).all())
        and bool((f0 >= 0).all())
        and bool((f0 <= 2000).all())
        and bool((rms >= 0).all())
    )


def _resize_vector(value: Tensor, frames: int) -> Tensor:
    value = value.flatten()
    if value.shape[0] == frames:
        return value
    return F.interpolate(
        value[None, None], size=frames, mode="linear", align_corners=False
    )[0, 0]


def _atomic_tensor(value: Tensor, destination: Path) -> None:
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


def _atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
