#!/usr/bin/env python3
"""Download and stage verified public singing datasets for RIFT-SVC V4.

The script never touches an existing destination file.  HF datasets are pinned
to an explicit revision and parquet audio is decoded to ordinary WAV files in
the V4 ``sources/<dataset>/<speaker>/<song>/`` layout.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

from rift_v4.datasets import ace_opencpop_identity, stage_gtsinger

HF_DATASETS = {
    "gtsinger": {
        "repo": "AaronZ345/GTSinger",
        "revision": "dc6c01fc093514f1e8137f98437328118c937128",
        "patterns": ["**/*.wav", "**/*.WAV"],
        "role": "main-real",
    },
    "ace-opencpop": {
        "repo": "espnet/ace-opencpop-segments",
        "revision": "a66b1ab0b3bdebfece8d78fd2980dfeebbcfd67c",
        "patterns": [
            "data/train-*.parquet",
            "data/validation-*.parquet",
            "data/test-*.parquet",
        ],
        "role": "low-weight-synthetic-augmentation",
    },
}

DEFAULT_DATASETS = ("gtsinger", "ace-opencpop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="V4 data root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(HF_DATASETS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="HF cache (defaults to <root>/hf-cache)"
    )
    parser.add_argument(
        "--execute-download",
        action="store_true",
        help="download and decode; without this flag only print the plan",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir or args.root / "hf-cache"
    plan = [HF_DATASETS[name] | {"name": name} for name in args.datasets]
    print(json.dumps(plan, indent=2))
    if not args.execute_download:
        print("plan only; pass --execute-download to write data")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "install huggingface_hub before running this script"
        ) from error

    for item in plan:
        started = time.time()
        record_dir = args.root / "records"
        record_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset": item["name"],
            "repository": item["repo"],
            "revision": item["revision"],
            "patterns": item["patterns"],
            "started_unix": started,
            "status": "running",
        }
        _write_record(record_dir / f"download-{item['name']}.json", record)
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=item["repo"],
                    repo_type="dataset",
                    revision=item["revision"],
                    allow_patterns=item["patterns"],
                    cache_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                )
            )
            snapshot_files = [path for path in snapshot.rglob("*") if path.is_file()]
            record.update(
                snapshot=str(snapshot),
                snapshot_files=len(snapshot_files),
                snapshot_bytes=sum(path.stat().st_size for path in snapshot_files),
            )
            if item["name"] == "gtsinger":
                staged = stage_gtsinger(snapshot, args.root / "sources" / "GTSinger")
            else:
                staged = stage_ace_opencpop(
                    snapshot, args.root / "sources" / "ACE-Opencpop"
                )
            record.update(status="complete", staged_wavs=staged)
        except Exception as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["finished_unix"] = time.time()
            record["elapsed_seconds"] = record["finished_unix"] - started
            _write_record(record_dir / f"download-{item['name']}.json", record)


def stage_ace_opencpop(snapshot: Path, destination: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("install pyarrow before decoding parquet datasets") from error

    count = 0
    for parquet_path in sorted(snapshot.glob("data/*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        columns = set(parquet.schema_arrow.names)
        if "audio" not in columns:
            print(f"warning: skip non-audio shard {parquet_path}", file=sys.stderr)
            continue
        requested = ["audio"] + [
            column
            for column in ("id", "gender", "segment_id", "singer")
            if column in columns
        ]
        row_index = 0
        for batch in parquet.iter_batches(columns=requested, batch_size=64):
            for row in batch.to_pylist():
                audio = row["audio"]
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not audio_bytes:
                    print(
                        f"warning: {parquet_path}:{row_index} has no embedded audio",
                        file=sys.stderr,
                    )
                    row_index += 1
                    continue
                identifier = str(row.get("segment_id", ""))
                singer = str(row.get("singer", ""))
                speaker, song, segment = ace_opencpop_identity(identifier, singer)
                output = destination / speaker / song / f"{segment}.wav"
                if not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    samples, sample_rate = sf.read(
                        io.BytesIO(audio_bytes), dtype="float32", always_2d=True
                    )
                    if samples.shape[1] != 1:
                        raise ValueError(
                            f"ACE-Opencpop:{identifier}: expected mono audio"
                        )
                    if samples.shape[0] == 0:
                        print(
                            f"warning: ACE-Opencpop:{identifier}: "
                            "skip zero-frame audio",
                            file=sys.stderr,
                        )
                        row_index += 1
                        continue
                    _atomic_wav(output, samples[:, 0], sample_rate)
                    count += 1
                row_index += 1
    print(f"staged {count} ACE-Opencpop WAV files")
    return count


def _atomic_wav(path: Path, samples, sample_rate: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sf.write(temporary, samples, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(temporary, path)


def _write_record(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
