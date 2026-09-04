from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import Tensor

from .config import V4Config
from .content_encoder import FrozenContentEncoder
from .manifest import ManifestEntry, load_manifest, write_manifest
from .third_party import verify_contentvec_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract immutable raw features from pinned ContentVec"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument(
        "--contentvec-lock",
        type=Path,
        default=Path("third_party/contentvec.lock.json"),
    )
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--execute-extraction", action="store_true")
    args = parser.parse_args()
    config = V4Config.load(args.config)
    entries = load_manifest(args.manifest)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if args.num_shards > 1:
        entries = [
            entry
            for index, entry in enumerate(entries)
            if index % args.num_shards == args.shard_index
        ]
    snapshot_hash = verify_contentvec_snapshot(
        args.base_model_path, args.contentvec_lock, config
    )
    print(
        f"ContentVec sha256={snapshot_hash}; recordings={len(entries)}; "
        f"output={args.output_manifest}"
    )
    if not args.execute_extraction:
        print(
            "validation only; pass --execute-extraction to load audio "
            "and write features"
        )
        return
    stamped = extract_all(
        entries,
        config,
        args.base_model_path,
        args.contentvec_lock,
        args.features_root,
        args.device,
        args.chunk_seconds,
        args.overlap_seconds,
    )
    write_manifest(stamped, args.output_manifest)


def extract_all(
    entries: list[ManifestEntry],
    config: V4Config,
    base_model_path: Path,
    contentvec_lock: Path,
    features_root: Path,
    device_name: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[ManifestEntry]:
    if overlap_seconds < 0 or chunk_seconds <= overlap_seconds:
        raise ValueError("chunk duration must be positive and greater than overlap")
    digest = verify_contentvec_snapshot(base_model_path, contentvec_lock, config)
    encoder = FrozenContentEncoder.from_local_pretrained(
        base_model_path,
        config.model.content_dim,
    )
    encoder.to(device_name).eval()
    encoder_id = f"contentvec-dualphase10ms-v1:{digest[:16]}"
    result: list[ManifestEntry] = []
    total = sum(entry.quality_status == "accepted" for entry in entries)
    completed = 0
    extracted = 0
    for entry in entries:
        if entry.quality_status != "accepted":
            result.append(entry)
            continue
        if _existing_content_is_reusable(
            entry, encoder_id, digest, config.model.content_dim
        ):
            result.append(entry)
            completed += 1
            if completed % 100 == 0:
                _print_progress(completed, total, extracted)
            continue
        destination = features_root / entry.dataset / f"{entry.id}.content.pt"
        if _cached_content_is_valid(destination, config.model.content_dim):
            result.append(
                replace(
                    entry,
                    content_feature_path=str(destination.resolve()),
                    content_encoder_id=encoder_id,
                    content_encoder_sha256=digest,
                )
            )
            completed += 1
            if completed % 100 == 0:
                _print_progress(completed, total, extracted)
            continue
        audio, source_rate = sf.read(entry.audio_path, dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError(f"{entry.id}: ContentVec extraction requires mono audio")
        waveform = torch.from_numpy(audio[:, 0])
        if source_rate != config.content_encoder.sample_rate:
            waveform = AF.resample(
                waveform, source_rate, config.content_encoder.sample_rate
            )
        content = encode_chunked(
            encoder,
            waveform,
            config.content_encoder.sample_rate,
            chunk_seconds,
            overlap_seconds,
            torch.device(device_name),
            config.content_encoder.phase_shift_seconds,
        )
        _atomic_tensor_save(content, destination)
        result.append(
            replace(
                entry,
                content_feature_path=str(destination.resolve()),
                content_encoder_id=encoder_id,
                content_encoder_sha256=digest,
            )
        )
        completed += 1
        extracted += 1
        if completed % 100 == 0:
            _print_progress(completed, total, extracted)
    _print_progress(completed, total, extracted)
    return result


def _print_progress(completed: int, total: int, written: int) -> None:
    print(
        json.dumps(
            {
                "stage": "contentvec",
                "completed": completed,
                "total": total,
                "written": written,
            }
        ),
        flush=True,
    )


def _existing_content_is_reusable(
    entry: ManifestEntry,
    encoder_id: str,
    encoder_sha256: str,
    content_dim: int,
) -> bool:
    return bool(
        entry.content_feature_path
        and entry.content_encoder_id == encoder_id
        and entry.content_encoder_sha256 == encoder_sha256
        and _cached_content_is_valid(Path(entry.content_feature_path), content_dim)
    )


def _cached_content_is_valid(destination: Path, content_dim: int) -> bool:
    if not destination.is_file():
        return False
    try:
        content = torch.load(
            destination, map_location="cpu", weights_only=True, mmap=True
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        content.ndim == 2
        and content.shape[0] > 0
        and content.shape[1] == content_dim
        and bool(torch.isfinite(content).all())
    )


@torch.inference_mode()
def encode_chunked(
    encoder: FrozenContentEncoder,
    waveform: Tensor,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    device: torch.device,
    phase_shift_seconds: float = 0.01,
) -> Tensor:
    """Extract phase-offset ContentVec states and interleave them."""
    shift = round(phase_shift_seconds * sample_rate)
    if shift <= 0:
        raise ValueError("phase shift is smaller than one input sample")
    shifted = torch.zeros_like(waveform)
    shifted[:-shift] = waveform[shift:]
    valid = torch.ones(waveform.shape[0], dtype=torch.bool)
    shifted_valid = valid.clone()
    shifted_valid[-shift:] = False
    phase0 = _encode_chunked_single(
        encoder, waveform, valid, sample_rate, chunk_seconds, overlap_seconds, device
    )
    phase1 = _encode_chunked_single(
        encoder,
        shifted,
        shifted_valid,
        sample_rate,
        chunk_seconds,
        overlap_seconds,
        device,
    )
    frames = min(phase0.shape[0], phase1.shape[0])
    if frames < 2:
        raise ValueError("phase-offset extraction produced too few frames")
    # Drop the final pair: the shifted phase sees the zero-padded tail there.
    return torch.stack((phase0[: frames - 1], phase1[: frames - 1]), dim=1).reshape(
        2 * (frames - 1), -1
    )


def _encode_chunked_single(
    encoder: FrozenContentEncoder,
    waveform: Tensor,
    valid_mask: Tensor,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    device: torch.device,
) -> Tensor:
    chunk = round(chunk_seconds * sample_rate)
    overlap = round(overlap_seconds * sample_rate)
    stride = chunk - overlap
    pieces: list[Tensor] = []
    start = 0
    while start < waveform.shape[0]:
        stop = min(waveform.shape[0], start + chunk)
        segment = waveform[start:stop].to(device)[None]
        mask = valid_mask[start:stop].to(device)[None]
        output, output_mask = encoder(segment, mask)
        content = output[0]
        valid_frames = int(output_mask[0].sum().item())
        if valid_frames <= 0:
            break
        content = content[:valid_frames]
        ratio = content.shape[0] / segment.shape[1]
        left_trim = round(overlap / 2 * ratio) if start else 0
        right_trim = round(overlap / 2 * ratio) if stop < waveform.shape[0] else 0
        right = content.shape[0] - right_trim
        if right <= left_trim:
            raise ValueError("overlap removed an entire ContentVec chunk")
        pieces.append(content[left_trim:right].float().cpu())
        if stop == waveform.shape[0]:
            break
        start += stride
    if not pieces:
        raise ValueError("cannot extract content from empty audio")
    return torch.cat(pieces)


def _atomic_tensor_save(tensor: Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}."
    )
    os.close(descriptor)
    try:
        torch.save(tensor, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
