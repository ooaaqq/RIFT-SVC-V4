from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

import soundfile as sf
import torch

from .config import V4Config
from .third_party import PCNSFLock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize with the pinned pretrained OpenVPI PC-NSF"
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mel", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    lock = PCNSFLock.load(args.lock)
    lock.validate_contract(config)
    lock.verify_checkout(args.checkout)
    lock.verify_installed_checkpoint(args.checkpoint)
    synthesize_pc_nsf(
        args.checkout,
        args.checkpoint,
        args.mel,
        args.f0,
        args.output,
        torch.device(args.device),
        config,
    )


@torch.inference_mode()
def synthesize_pc_nsf(
    checkout: Path,
    checkpoint_path: Path,
    mel_path: Path,
    f0_path: Path,
    output_path: Path,
    device: torch.device,
    config: V4Config,
) -> None:
    generator = load_pc_nsf_generator(checkout, checkpoint_path, device, config)
    mel = torch.as_tensor(
        torch.load(mel_path, map_location="cpu", weights_only=True)
    ).float()
    f0 = torch.as_tensor(
        torch.load(f0_path, map_location="cpu", weights_only=True)
    ).float()
    waveform = synthesize_pc_nsf_tensors(generator, mel, f0, device, config)
    _write_waveform(output_path, waveform, config.sample_rate)


def official_generator_config(config: V4Config) -> dict[str, object]:
    """Generator fields from the pinned upstream configs/ft_hifigan.yaml."""

    return {
        "mini_nsf": True,
        "noise_sigma": 0.0,
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock": "1",
        "sampling_rate": config.sample_rate,
        "num_mels": config.mel.channels,
        "hop_size": config.hop_length,
        "n_fft": config.mel.n_fft,
        "win_size": config.mel.win_length,
        "fmin": config.mel.fmin,
        "fmax": config.mel.fmax,
        "pc_aug": True,
    }


def official_generator_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("official PC-NSF checkpoint must contain state_dict")
    state = {
        name.removeprefix("generator."): value
        for name, value in checkpoint["state_dict"].items()
        if name.startswith("generator.")
    }
    if not state:
        raise ValueError("official PC-NSF checkpoint contains no generator weights")
    return state


def load_pc_nsf_generator(
    checkout: Path,
    checkpoint_path: Path,
    device: torch.device,
    config: V4Config,
):
    module_path = checkout / "models/nsf_HiFigan/models.py"
    specification = importlib.util.spec_from_file_location("rift_pc_nsf", module_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load official PC-NSF module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    generator = module.Generator(module.AttrDict(official_generator_config(config))).to(
        device
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(official_generator_state(checkpoint), strict=True)
    generator.eval()
    generator.remove_weight_norm()
    return generator


@torch.inference_mode()
def synthesize_pc_nsf_tensors(
    generator,
    mel: torch.Tensor,
    f0: torch.Tensor,
    device: torch.device,
    config: V4Config,
) -> torch.Tensor:
    mel = torch.as_tensor(mel).float()
    f0 = torch.as_tensor(f0).float().squeeze()
    if mel.ndim != 2:
        raise ValueError("mel must be rank two")
    if mel.shape[-1] == config.mel.channels:
        mel = mel.transpose(0, 1)
    if mel.shape[0] != config.mel.channels:
        raise ValueError("mel channel mismatch")
    if f0.ndim != 1:
        raise ValueError("F0 must be rank one")
    if mel.shape[1] != f0.shape[0]:
        raise ValueError(
            f"mel and F0 frame counts differ: {mel.shape[1]} vs {f0.shape[0]}"
        )
    if not bool(torch.isfinite(f0).all()) or (f0 < 0).any() or (f0 > 2000).any():
        raise ValueError("F0 is non-finite or outside [0, 2000] Hz")
    frames = mel.shape[1]
    if frames <= 0:
        raise ValueError("cannot synthesize an empty feature sequence")
    waveform = generator(mel[:, :frames][None].to(device), f0[:frames][None].to(device))
    return waveform[0, 0].float().cpu().clamp(-1, 1)


def _write_waveform(
    output_path: Path, waveform: torch.Tensor, sample_rate: int
) -> None:
    audio = waveform.numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".wav"
    )
    os.close(descriptor)
    try:
        sf.write(temporary, audio, sample_rate, subtype="PCM_24")
        os.replace(temporary, output_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
