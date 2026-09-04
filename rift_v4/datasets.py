from __future__ import annotations

import argparse
import os
from pathlib import Path

GTSINGER_GROUPS = frozenset(
    {
        "Control_Group",
        "Breathy_Group",
        "Falsetto_Group",
        "Glissando_Group",
        "Mixed_Voice_Group",
        "Pharyngeal_Group",
        "Vibrato_Group",
    }
)
OPENSINGER_GENDERS = {"ManRaw": "male", "WomanRaw": "female"}


def ace_opencpop_identity(identifier: str, singer: str) -> tuple[str, str, str]:
    """Return speaker, source-song, and segment IDs from the official row key."""

    prefix, separator, segment = identifier.partition("#")
    if not separator or prefix != f"acesinger_{singer}":
        raise ValueError(f"invalid ACE-Opencpop identity: {identifier!r}/{singer!r}")
    if len(segment) != 10 or not segment.isdigit():
        raise ValueError(f"invalid Opencpop segment id: {segment!r}")
    return singer, segment[:4], segment


def stage_gtsinger(source_root: Path, destination: Path) -> int:
    """Stage all singing groups without collapsing paired filenames."""

    expected: dict[Path, Path] = {}
    for source in sorted(source_root.rglob("*")):
        if source.suffix.lower() != ".wav":
            continue
        parts = source.relative_to(source_root).parts
        # language/speaker/technique/song/group/file.wav
        if len(parts) < 6:
            raise ValueError(f"unexpected GTSinger path: {source}")
        group = parts[-2]
        if group == "Paired_Speech_Group":
            continue
        if group not in GTSINGER_GROUPS:
            raise ValueError(f"unexpected GTSinger singing group: {source}")
        language, speaker = parts[0], parts[1]
        technique = "__".join(parts[2:-3])
        song = parts[-3]
        if not technique:
            raise ValueError(f"missing GTSinger technique directory: {source}")
        target = (
            destination
            / f"{language}-{speaker}"
            / song
            / f"{technique}__{group}__{source.name}"
        )
        previous = expected.get(target)
        if previous is not None and previous.resolve() != source.resolve():
            raise ValueError(f"GTSinger source collision: {target}")
        expected[target] = source

    if not expected:
        raise ValueError(f"no GTSinger singing WAVs found under {source_root}")
    existing = (
        {path for path in destination.rglob("*") if path.is_file() or path.is_symlink()}
        if destination.exists()
        else set()
    )
    unexpected = existing - set(expected)
    if unexpected:
        sample = sorted(unexpected)[:5]
        raise FileExistsError(
            f"destination contains stale or unexpected files: {sample}"
        )
    for target, source in expected.items():
        _link_exact(source, target)
    return len(expected)


def stage_opensinger(source_root: Path, destination: Path) -> tuple[int, int]:
    """Stage the verified official OpenSinger archive as normalized symlinks."""

    expected: dict[Path, Path] = {}
    speakers: set[str] = set()
    for directory, gender in OPENSINGER_GENDERS.items():
        root = source_root / directory
        if not root.is_dir():
            raise FileNotFoundError(root)
        for song_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            try:
                identifier, _song = song_dir.name.split("_", 1)
                singer_number = int(identifier)
            except ValueError as error:
                raise ValueError(
                    f"unexpected OpenSinger directory: {song_dir}"
                ) from error
            speaker = f"{gender}-{singer_number:02d}"
            speakers.add(speaker)
            for source in sorted(song_dir.glob("*.wav")):
                target = destination / speaker / song_dir.name / source.name
                previous = expected.get(target)
                if previous is not None and previous.resolve() != source.resolve():
                    raise ValueError(f"OpenSinger source collision: {target}")
                expected[target] = source
    if len(expected) != 43_075 or len(speakers) != 76:
        raise ValueError(
            "official OpenSinger must contain 43075 WAVs from 76 speakers; "
            f"found {len(expected)} WAVs from {len(speakers)} speakers"
        )
    existing = (
        {path for path in destination.rglob("*") if path.is_file() or path.is_symlink()}
        if destination.exists()
        else set()
    )
    unexpected = existing - set(expected)
    if unexpected:
        raise FileExistsError(
            f"destination contains stale or unexpected files: {sorted(unexpected)[:5]}"
        )
    for target, source in expected.items():
        _link_exact(source, target)
    return len(expected), len(speakers)


def _link_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise FileExistsError(f"staging collision: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.symlink_to(source.resolve())
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage normalized singing datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("gtsinger", "opensinger"):
        child = subparsers.add_parser(command)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gtsinger":
        count = stage_gtsinger(args.source, args.output)
        if count != 28_628:
            raise ValueError(
                f"official GTSinger must contain 28628 singing WAVs; found {count}"
            )
        print(f"staged {count} GTSinger singing WAVs")
    else:
        count, speakers = stage_opensinger(args.source, args.output)
        print(f"staged {count} OpenSinger WAVs from {speakers} speakers")


if __name__ == "__main__":
    main()
