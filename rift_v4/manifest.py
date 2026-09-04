from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import soundfile as sf

QualityStatus = Literal["pending", "accepted", "rejected"]
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class ManifestEntry:
    """One immutable source recording and its derived feature location."""

    id: str
    dataset: str
    speaker: str
    song: str
    audio_path: str
    feature_prefix: str
    frames: int
    duration_seconds: float
    sample_rate: int
    channels: int
    audio_sha256: str
    content_feature_path: str | None = None
    content_encoder_id: str | None = None
    content_encoder_sha256: str | None = None
    split: Split = "train"
    language: str | None = None
    technique: str | None = None
    quality_status: QualityStatus = "pending"
    quality_score: float | None = None
    voiced_ratio: float | None = None
    peak: float | None = None
    clipping_ratio: float | None = None
    bandwidth_hz: float | None = None
    loudness_lufs: float | None = None
    exclusion_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def speaker_key(self) -> str:
        return f"{self.dataset}:{self.speaker}"

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ManifestEntry:
        copy = dict(payload)
        copy["exclusion_reasons"] = tuple(copy.get("exclusion_reasons", ()))
        return cls(**copy)  # type: ignore[arg-type]

    def validate(self) -> None:
        if not self.id or not self.dataset or not self.speaker or not self.song:
            raise ValueError("id, dataset, speaker, and song must be non-empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"{self.id}: invalid split {self.split!r}")
        if self.quality_status not in {"pending", "accepted", "rejected"}:
            raise ValueError(
                f"{self.id}: invalid quality status {self.quality_status!r}"
            )
        if self.frames <= 0 or self.duration_seconds <= 0:
            raise ValueError(f"{self.id}: duration and frame count must be positive")
        if not math.isfinite(self.duration_seconds):
            raise ValueError(f"{self.id}: duration must be finite")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError(f"{self.id}: invalid audio stream metadata")
        if len(self.audio_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.audio_sha256
        ):
            raise ValueError(f"{self.id}: invalid SHA-256")
        if self.quality_status == "accepted" and self.exclusion_reasons:
            raise ValueError(f"{self.id}: accepted item has exclusion reasons")
        if self.quality_status == "rejected" and not self.exclusion_reasons:
            raise ValueError(f"{self.id}: rejected item needs an exclusion reason")
        provenance = (
            self.content_feature_path,
            self.content_encoder_id,
            self.content_encoder_sha256,
        )
        if any(value is not None for value in provenance) and not all(
            value is not None for value in provenance
        ):
            raise ValueError(f"{self.id}: incomplete content feature provenance")
        if self.content_encoder_sha256 is not None and (
            len(self.content_encoder_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.content_encoder_sha256
            )
        ):
            raise ValueError(f"{self.id}: invalid content encoder SHA-256")


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = ManifestEntry.from_dict(json.loads(line))
                entry.validate()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid manifest line {line_number}: {error}"
                ) from error
            if entry.id in seen:
                raise ValueError(
                    f"duplicate manifest id on line {line_number}: {entry.id}"
                )
            seen.add(entry.id)
            entries.append(entry)
    if not entries:
        raise ValueError("manifest is empty")
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> int:
    """Atomically write canonical JSONL and return the number of entries."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted(
        entries, key=lambda item: (item.dataset, item.speaker, item.song, item.id)
    )
    seen: set[str] = set()
    for entry in materialized:
        entry.validate()
        if entry.id in seen:
            raise ValueError(f"duplicate manifest id: {entry.id}")
        seen.add(entry.id)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for entry in materialized:
                payload = asdict(entry)
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return len(materialized)


def discover_recordings(
    sources: dict[str, Path],
    features_root: Path,
    sample_rate: int,
    hop_length: int,
) -> Iterator[ManifestEntry]:
    """Discover normalized ``speaker/song/audio`` trees without copying audio."""

    extensions = {".wav", ".flac", ".ogg"}
    for dataset, root in sorted(sources.items()):
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        recordings = (
            path for path in root.rglob("*") if path.suffix.lower() in extensions
        )
        for audio_path in sorted(recordings):
            relative = audio_path.relative_to(root)
            if len(relative.parts) < 2:
                raise ValueError(
                    f"{audio_path}: expected at least <speaker>/<audio>; "
                    "normalize the source tree before indexing"
                )
            speaker = relative.parts[0]
            song = relative.parts[1] if len(relative.parts) > 2 else audio_path.stem
            info = sf.info(audio_path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"invalid or empty audio file: {audio_path}")
            digest = _sha256_file(audio_path)
            stable_id = hashlib.sha256(
                f"{dataset}\0{relative.as_posix()}\0{digest}".encode()
            ).hexdigest()[:24]
            prefix = features_root / dataset / relative.with_suffix("")
            yield ManifestEntry(
                id=stable_id,
                dataset=dataset,
                speaker=speaker,
                song=song,
                audio_path=str(audio_path.resolve()),
                feature_prefix=str(prefix.resolve()),
                frames=max(
                    1,
                    round(info.frames / info.samplerate * sample_rate / hop_length),
                ),
                duration_seconds=info.frames / info.samplerate,
                sample_rate=info.samplerate,
                channels=info.channels,
                audio_sha256=digest,
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
