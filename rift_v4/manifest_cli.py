from __future__ import annotations

import argparse
from pathlib import Path

from .config import V4Config
from .manifest import discover_recordings, write_manifest


def _source(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use DATASET=/absolute/path") from error
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("dataset name and path must be non-empty")
    return name, Path(raw_path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index normalized singing corpora into the V4 immutable manifest"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--source", type=_source, action="append", required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = V4Config.load(args.config)
    sources = dict(args.source)
    if len(sources) != len(args.source):
        parser.error("dataset names passed to --source must be unique")
    count = write_manifest(
        discover_recordings(
            sources, args.features_root, config.sample_rate, config.hop_length
        ),
        args.output,
    )
    print(f"indexed {count} recordings into {args.output}")


if __name__ == "__main__":
    main()
