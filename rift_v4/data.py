from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .config import SamplingConfig
from .features import MelStats
from .manifest import ManifestEntry


@dataclass(frozen=True)
class SampleRequest:
    index: int
    frames: int
    seed: int


def bounded_normalize(
    weights: Sequence[float], lower: float, upper: float
) -> list[float]:
    """Normalize positive weights onto a box-constrained probability simplex."""

    values = [float(value) for value in weights]
    count = len(values)
    if not count or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("bounded normalization needs finite positive weights")
    if (
        not 0 <= lower <= upper
        or count * lower > 1 + 1e-12
        or (count * upper < 1 - 1e-12)
    ):
        raise ValueError("bounded normalization constraints are infeasible")
    result: list[float | None] = [None] * count
    free = set(range(count))
    remaining = 1.0
    while free:
        scale = remaining / sum(values[index] for index in free)
        low = [index for index in free if values[index] * scale < lower]
        high = [index for index in free if values[index] * scale > upper]
        if not low and not high:
            for index in free:
                result[index] = values[index] * scale
            break
        for index in low:
            result[index] = lower
            remaining -= lower
            free.remove(index)
        for index in high:
            result[index] = upper
            remaining -= upper
            free.remove(index)
        if remaining < -1e-10:
            raise ValueError("bounded normalization exhausted probability mass")
    normalized = [float(value) for value in result]
    drift = 1.0 - sum(normalized)
    if abs(drift) > 1e-10:
        adjustable = [
            index
            for index, value in enumerate(normalized)
            if lower - 1e-12 <= value + drift <= upper + 1e-12
        ]
        if not adjustable:
            raise RuntimeError("bounded normalization could not correct rounding")
        normalized[adjustable[0]] += drift
    return normalized


def build_sampler(
    entries: Sequence[ManifestEntry], config: SamplingConfig
) -> HierarchicalBatchSampler:
    return HierarchicalBatchSampler(
        entries,
        batch_size=config.batch_size,
        batch_frame_budget=config.batch_frame_budget,
        steps_per_epoch=config.steps_per_epoch,
        frame_buckets=config.frame_buckets,
        bucket_probabilities=config.bucket_probabilities,
        dataset_probabilities=config.dataset_probabilities,
        speaker_duration_exponent=config.speaker_duration_exponent,
        speaker_probability_floor_ratio=config.speaker_probability_floor_ratio,
        speaker_probability_ceiling_ratio=config.speaker_probability_ceiling_ratio,
        song_duration_exponent=config.song_duration_exponent,
        song_probability_floor_ratio=config.song_probability_floor_ratio,
        song_probability_ceiling_ratio=config.song_probability_ceiling_ratio,
        dataset_families=config.dataset_families,
        family_probability_caps=config.family_probability_caps,
        synthetic_datasets=config.synthetic_datasets,
        max_singleton_real_speaker_median_ratio=(
            config.max_singleton_real_speaker_median_ratio
        ),
        seed=config.seed,
    )


class FeatureDataset(Dataset[dict[str, Tensor]]):
    """Load already extracted tensors; never invokes or downloads an encoder."""

    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        mel_channels: int,
        content_dim: int,
        mel_stats: MelStats | None = None,
        speaker_to_id: dict[str, int] | None = None,
        voiced_crop_probability: float = 0.7,
    ):
        self.entries = list(entries)
        self.mel_channels = mel_channels
        self.content_dim = content_dim
        self.mel_stats = mel_stats
        self.voiced_crop_probability = voiced_crop_probability
        speakers = sorted({entry.speaker_key for entry in entries})
        self.speaker_to_id = speaker_to_id or {
            speaker: index for index, speaker in enumerate(speakers)
        }
        unknown = set(speakers) - set(self.speaker_to_id)
        if unknown:
            raise ValueError(f"unknown speakers in feature dataset: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, request: int | SampleRequest) -> dict[str, Tensor]:
        if isinstance(request, int):
            request = SampleRequest(request, self.entries[request].frames, request)
        entry = self.entries[request.index]
        features = self._load_features(entry)
        available = min(tensor.shape[0] for tensor in features.values())
        wanted = min(request.frames, available)
        if wanted <= 0:
            raise ValueError(f"{entry.id}: no aligned feature frames")
        generator = random.Random(request.seed)
        starts = [generator.randrange(available - wanted + 1) for _ in range(8)]
        # Do not destructively trim source recordings. Prefer a voiced crop at
        # sample time, while retaining the best available crop for sparse files.
        voiced_start = max(
            starts,
            key=lambda offset: float(
                (features["f0"][offset : offset + wanted] > 0).float().mean()
            ),
        )
        start = (
            voiced_start
            if generator.random() < self.voiced_crop_probability
            else generator.choice(starts)
        )
        stop = start + wanted
        cropped = {name: tensor[start:stop] for name, tensor in features.items()}
        return {
            **cropped,
            "speaker": torch.tensor(
                self.speaker_to_id[entry.speaker_key], dtype=torch.long
            ),
            "length": torch.tensor(wanted, dtype=torch.long),
            "requested_length": torch.tensor(request.frames, dtype=torch.long),
        }

    def _load_features(self, entry: ManifestEntry) -> dict[str, Tensor]:
        prefix = Path(entry.feature_prefix)
        mel = _matrix(
            torch.load(f"{prefix}.mel.pt", map_location="cpu", weights_only=True),
            self.mel_channels,
            "mel",
        )
        content_path = entry.content_feature_path or f"{prefix}.content.pt"
        content = _matrix(
            torch.load(content_path, map_location="cpu", weights_only=True),
            self.content_dim,
            "content",
        )
        f0 = _vector(
            torch.load(f"{prefix}.f0.pt", map_location="cpu", weights_only=True), "f0"
        )
        rms = _vector(
            torch.load(f"{prefix}.rms.pt", map_location="cpu", weights_only=True), "rms"
        )
        target = mel.shape[0]
        if f0.shape[0] != target or rms.shape[0] != target:
            raise ValueError(
                f"{entry.id}: F0/RMS must match mel frames "
                f"({f0.shape[0]}/{rms.shape[0]} vs {target})"
            )
        mel = mel.float()
        if self.mel_stats is not None:
            mel = self.mel_stats.normalize(mel)
        return {
            "mel": mel,
            "content": _resize(content.float(), target),
            "f0": f0.float(),
            "rms": rms.float(),
        }


class HierarchicalBatchSampler(Sampler[list[SampleRequest]]):
    """Dataset -> duration-tempered speaker -> song -> recording."""

    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        batch_size: int,
        batch_frame_budget: int | None,
        steps_per_epoch: int,
        frame_buckets: Sequence[int],
        bucket_probabilities: Sequence[float],
        dataset_probabilities: dict[str, float] | None = None,
        speaker_duration_exponent: float = 0.5,
        speaker_probability_floor_ratio: float = 0.5,
        speaker_probability_ceiling_ratio: float = 2.0,
        song_duration_exponent: float = 0.5,
        song_probability_floor_ratio: float = 0.5,
        song_probability_ceiling_ratio: float = 2.0,
        dataset_families: dict[str, str] | None = None,
        family_probability_caps: dict[str, float] | None = None,
        synthetic_datasets: Sequence[str] = (),
        max_singleton_real_speaker_median_ratio: float = math.inf,
        seed: int = 0,
    ) -> None:
        self.entries = entries
        self.batch_size = batch_size
        self.batch_frame_budget = batch_frame_budget
        self.steps_per_epoch = steps_per_epoch
        self.frame_buckets = tuple(frame_buckets)
        self.bucket_probabilities = tuple(bucket_probabilities)
        self.seed = seed
        self.epoch = 0
        self.dataset_families = dict(dataset_families or {})
        self.family_probability_caps = dict(family_probability_caps or {})
        self.synthetic_datasets = frozenset(synthetic_datasets)
        if len(self.frame_buckets) != len(self.bucket_probabilities):
            raise ValueError("frame buckets and probabilities must have equal length")
        if not self.frame_buckets or any(
            int(frames) <= 0 for frames in self.frame_buckets
        ):
            raise ValueError("frame buckets must be positive")
        if any(
            not math.isfinite(float(probability)) or float(probability) < 0
            for probability in self.bucket_probabilities
        ) or not any(self.bucket_probabilities):
            raise ValueError("bucket probabilities must be finite and not all zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_frame_budget is not None and batch_frame_budget < min(
            self.frame_buckets
        ):
            raise ValueError(
                "batch_frame_budget must fit at least one requested frame bucket"
            )
        hierarchy: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for index, entry in enumerate(entries):
            if entry.split == "train" and entry.quality_status == "accepted":
                hierarchy[entry.dataset][entry.speaker][entry.song].append(index)
        if not hierarchy:
            raise ValueError("no accepted training entries in manifest")
        self.hierarchy = hierarchy
        self.datasets = sorted(hierarchy)
        requested = dataset_probabilities or {
            name: 1.0 / len(self.datasets) for name in self.datasets
        }
        unknown = set(requested) - set(self.datasets)
        if unknown:
            raise ValueError(
                f"probabilities refer to absent datasets: {sorted(unknown)}"
            )
        missing = set(self.datasets) - set(requested)
        if missing:
            raise ValueError(f"probabilities omit present datasets: {sorted(missing)}")
        if abs(sum(float(value) for value in requested.values()) - 1.0) > 1e-6:
            raise ValueError("dataset probabilities must sum to 1")
        # These are explicit draw probabilities, not duration multipliers. This
        # keeps the intended corpus mixture stable as manifests change.
        self.dataset_probabilities = [float(requested[name]) for name in self.datasets]
        if any(
            not math.isfinite(probability) or probability <= 0
            for probability in self.dataset_probabilities
        ):
            raise ValueError("dataset probabilities must be positive")
        if not 0 <= speaker_duration_exponent <= 1:
            raise ValueError("speaker duration exponent must be in [0, 1]")
        if not 0 < speaker_probability_floor_ratio <= 1:
            raise ValueError("speaker probability floor ratio must be in (0, 1]")
        if speaker_probability_ceiling_ratio < 1:
            raise ValueError("speaker probability ceiling ratio must be at least 1")
        if not 0 <= song_duration_exponent <= 1:
            raise ValueError("song duration exponent must be in [0, 1]")
        if not 0 < song_probability_floor_ratio <= 1:
            raise ValueError("song probability floor ratio must be in (0, 1]")
        if song_probability_ceiling_ratio < 1:
            raise ValueError("song probability ceiling ratio must be at least 1")
        self.speaker_duration_exponent = speaker_duration_exponent
        self.speaker_probability_floor_ratio = speaker_probability_floor_ratio
        self.speaker_probability_ceiling_ratio = speaker_probability_ceiling_ratio
        self.song_duration_exponent = song_duration_exponent
        self.song_probability_floor_ratio = song_probability_floor_ratio
        self.song_probability_ceiling_ratio = song_probability_ceiling_ratio
        self.max_singleton_real_speaker_median_ratio = (
            max_singleton_real_speaker_median_ratio
        )
        self.speaker_probabilities: dict[str, dict[str, float]] = {}
        for dataset, speakers in sorted(hierarchy.items()):
            names = sorted(speakers)
            count = len(names)
            durations = [
                sum(
                    self.entries[index].frames
                    for songs in (speakers[name],)
                    for candidates in songs.values()
                    for index in candidates
                )
                for name in names
            ]
            weights = [duration**speaker_duration_exponent for duration in durations]
            probabilities = bounded_normalize(
                weights,
                speaker_probability_floor_ratio / count,
                speaker_probability_ceiling_ratio / count,
            )
            self.speaker_probabilities[dataset] = dict(
                zip(names, probabilities, strict=True)
            )
        self.song_probabilities: dict[str, dict[str, dict[str, float]]] = {}
        for dataset, speakers in sorted(hierarchy.items()):
            self.song_probabilities[dataset] = {}
            for speaker, songs in sorted(speakers.items()):
                names = sorted(songs)
                count = len(names)
                durations = [
                    sum(self.entries[index].frames for index in songs[name])
                    for name in names
                ]
                weights = [duration**song_duration_exponent for duration in durations]
                probabilities = bounded_normalize(
                    weights,
                    song_probability_floor_ratio / count,
                    song_probability_ceiling_ratio / count,
                )
                self.song_probabilities[dataset][speaker] = dict(
                    zip(names, probabilities, strict=True)
                )
        self._validate_exposure_constraints(max_singleton_real_speaker_median_ratio)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        for step in range(self.steps_per_epoch):
            frames = rng.choices(
                self.frame_buckets, weights=self.bucket_probabilities, k=1
            )[0]
            batch_size = self.batch_size
            if self.batch_frame_budget is not None:
                batch_size = min(batch_size, self.batch_frame_budget // frames)
            if batch_size <= 0:
                raise RuntimeError("batch_frame_budget produced an empty batch")
            batch: list[SampleRequest] = []
            for position in range(batch_size):
                dataset = rng.choices(
                    self.datasets, weights=self.dataset_probabilities, k=1
                )[0]
                speakers = sorted(self.hierarchy[dataset])
                speaker = rng.choices(
                    speakers,
                    weights=[
                        self.speaker_probabilities[dataset][name] for name in speakers
                    ],
                    k=1,
                )[0]
                songs = sorted(self.hierarchy[dataset][speaker])
                song = rng.choices(
                    songs,
                    weights=[
                        self.song_probabilities[dataset][speaker][name]
                        for name in songs
                    ],
                    k=1,
                )[0]
                candidates = self.hierarchy[dataset][speaker][song]
                index = rng.choices(
                    candidates,
                    weights=[self.entries[item].frames for item in candidates],
                    k=1,
                )[0]
                crop_seed = (
                    self.seed + self.epoch * 10**9 + step * self.batch_size + position
                )
                batch.append(SampleRequest(index, frames, crop_seed))
            yield batch

    def _validate_exposure_constraints(
        self, max_singleton_real_median_ratio: float
    ) -> None:
        dataset_weights = dict(
            zip(self.datasets, self.dataset_probabilities, strict=True)
        )
        family_totals: dict[str, float] = defaultdict(float)
        for dataset, probability in dataset_weights.items():
            family_totals[self.dataset_families.get(dataset, dataset)] += probability
        exceeded = {
            family: family_totals.get(family, 0.0)
            for family, cap in self.family_probability_caps.items()
            if family_totals.get(family, 0.0) > cap + 1e-12
        }
        if exceeded:
            raise ValueError(f"dataset family probability cap exceeded: {exceeded}")
        real_probabilities = [
            dataset_weights[dataset] * probability
            for dataset, speakers in self.speaker_probabilities.items()
            if dataset not in self.synthetic_datasets
            for probability in speakers.values()
        ]
        if real_probabilities and math.isfinite(max_singleton_real_median_ratio):
            median = statistics.median(real_probabilities)
            singleton_probabilities = [
                dataset_weights[dataset]
                for dataset, speakers in self.speaker_probabilities.items()
                if dataset not in self.synthetic_datasets and len(speakers) == 1
            ]
            maximum = max(singleton_probabilities, default=0.0)
            if maximum > median * max_singleton_real_median_ratio + 1e-12:
                raise ValueError(
                    "singleton real speaker probability exceeds median cap: "
                    f"{maximum:.8f} > {median:.8f} * "
                    f"{max_singleton_real_median_ratio:g}"
                )

    def sampling_audit(
        self,
        *,
        max_steps: int,
        sample_rate: int,
        hop_length: int,
        speaker_drop_probability: float,
    ) -> dict[str, object]:
        """Return exact expected speaker/song exposure under the configured draws."""

        if max_steps <= 0 or sample_rate <= 0 or hop_length <= 0:
            raise ValueError("sampling audit dimensions must be positive")
        bucket_draws = [
            probability
            * min(
                self.batch_size,
                self.batch_frame_budget // frames
                if self.batch_frame_budget is not None
                else self.batch_size,
            )
            for frames, probability in zip(
                self.frame_buckets, self.bucket_probabilities, strict=True
            )
        ]
        draws_per_step = sum(bucket_draws)
        dataset_weights = dict(
            zip(self.datasets, self.dataset_probabilities, strict=True)
        )
        song_rows: list[dict[str, object]] = []
        speaker_rows: list[dict[str, object]] = []
        source_songs: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"probability": 0.0, "source_frames": 0.0, "exposure_frames": 0.0}
        )
        for dataset in self.datasets:
            for speaker, conditional in self.speaker_probabilities[dataset].items():
                songs = self.hierarchy[dataset][speaker]
                speaker_probability = dataset_weights[dataset] * conditional
                speaker_exposure_frames = 0.0
                speaker_source_frames = 0
                for song, candidates in sorted(songs.items()):
                    song_probability = (
                        speaker_probability
                        * self.song_probabilities[dataset][speaker][song]
                    )
                    candidate_total = sum(
                        self.entries[index].frames for index in candidates
                    )
                    expected_frames_per_step = 0.0
                    for bucket, draws in zip(
                        self.frame_buckets, bucket_draws, strict=True
                    ):
                        expected_crop = sum(
                            self.entries[index].frames
                            / candidate_total
                            * min(bucket, self.entries[index].frames)
                            for index in candidates
                        )
                        expected_frames_per_step += draws * expected_crop
                    source_frames = candidate_total
                    exposure_frames = (
                        max_steps * song_probability * expected_frames_per_step
                    )
                    expected_crops = max_steps * song_probability * draws_per_step
                    exposure_hours = _frame_hours(
                        exposure_frames, sample_rate, hop_length
                    )
                    source_hours = _frame_hours(source_frames, sample_rate, hop_length)
                    song_rows.append(
                        {
                            "dataset": dataset,
                            "speaker": speaker,
                            "song": song,
                            "probability": song_probability,
                            "expected_crops": expected_crops,
                            "expected_audio_hours": exposure_hours,
                            "source_hours": source_hours,
                            "repeat_equivalent": exposure_hours / source_hours,
                        }
                    )
                    family = self.dataset_families.get(dataset, dataset)
                    family_row = source_songs[(family, song)]
                    family_row["probability"] += song_probability
                    family_row["source_frames"] += source_frames
                    family_row["exposure_frames"] += exposure_frames
                    speaker_source_frames += source_frames
                    speaker_exposure_frames += exposure_frames
                source_hours = _frame_hours(
                    speaker_source_frames, sample_rate, hop_length
                )
                exposure_hours = _frame_hours(
                    speaker_exposure_frames, sample_rate, hop_length
                )
                speaker_rows.append(
                    {
                        "dataset": dataset,
                        "speaker": speaker,
                        "probability": speaker_probability,
                        "conditional_probability": conditional,
                        "songs": len(songs),
                        "expected_crops": max_steps
                        * speaker_probability
                        * draws_per_step,
                        "speaker_conditioned_crops": max_steps
                        * speaker_probability
                        * draws_per_step
                        * (1 - speaker_drop_probability),
                        "expected_audio_hours": exposure_hours,
                        "source_hours": source_hours,
                        "repeat_equivalent": exposure_hours / source_hours,
                    }
                )
        real = [
            float(row["probability"])
            for row in speaker_rows
            if row["dataset"] not in self.synthetic_datasets
        ]
        singleton_real = [
            float(row["probability"])
            for row in speaker_rows
            if row["dataset"] not in self.synthetic_datasets
            and len(self.speaker_probabilities[str(row["dataset"])]) == 1
        ]
        family_rows = []
        for (family, song), values in sorted(source_songs.items()):
            source_hours = _frame_hours(
                values["source_frames"], sample_rate, hop_length
            )
            exposure_hours = _frame_hours(
                values["exposure_frames"], sample_rate, hop_length
            )
            family_rows.append(
                {
                    "family": family,
                    "song": song,
                    "probability": values["probability"],
                    "expected_crops": max_steps
                    * values["probability"]
                    * draws_per_step,
                    "expected_audio_hours": exposure_hours,
                    "source_hours": source_hours,
                    "repeat_equivalent": exposure_hours / source_hours,
                }
            )
        dataset_rows = []
        for dataset in self.datasets:
            rows = [row for row in speaker_rows if row["dataset"] == dataset]
            source_hours = sum(float(row["source_hours"]) for row in rows)
            exposure_hours = sum(float(row["expected_audio_hours"]) for row in rows)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "probability": dataset_weights[dataset],
                    "speakers": len(rows),
                    "expected_crops": sum(float(row["expected_crops"]) for row in rows),
                    "expected_audio_hours": exposure_hours,
                    "source_hours": source_hours,
                    "repeat_equivalent": exposure_hours / source_hours,
                }
            )
        return {
            "protocol": 1,
            "max_steps": max_steps,
            "speaker_duration_exponent": self.speaker_duration_exponent,
            "speaker_probability_floor_ratio": (self.speaker_probability_floor_ratio),
            "speaker_probability_ceiling_ratio": (
                self.speaker_probability_ceiling_ratio
            ),
            "song_duration_exponent": self.song_duration_exponent,
            "song_probability_floor_ratio": self.song_probability_floor_ratio,
            "song_probability_ceiling_ratio": self.song_probability_ceiling_ratio,
            "max_singleton_real_speaker_median_ratio": (
                self.max_singleton_real_speaker_median_ratio
            ),
            "expected_crops_per_step": draws_per_step,
            "expected_total_crops": max_steps * draws_per_step,
            "dataset_probabilities": dataset_weights,
            "family_probabilities": {
                family: sum(
                    probability
                    for dataset, probability in dataset_weights.items()
                    if self.dataset_families.get(dataset, dataset) == family
                )
                for family in sorted(
                    {self.dataset_families.get(name, name) for name in self.datasets}
                )
            },
            "real_speaker_probability_median": statistics.median(real),
            "real_speaker_probability_max": max(real),
            "real_speaker_max_to_median": max(real) / statistics.median(real),
            "singleton_real_speaker_probability_max": max(singleton_real, default=0.0),
            "singleton_real_speaker_max_to_median": max(singleton_real, default=0.0)
            / statistics.median(real),
            "song_repeat_equivalent_quantiles": _quantiles(
                [float(row["repeat_equivalent"]) for row in song_rows],
                (0.5, 0.9, 0.95, 0.99, 1.0),
            ),
            "datasets": dataset_rows,
            "speakers": speaker_rows,
            "songs": song_rows,
            "source_family_songs": family_rows,
        }


def _frame_hours(frames: float, sample_rate: int, hop_length: int) -> float:
    return frames * hop_length / sample_rate / 3600


def _quantiles(
    values: Sequence[float], probabilities: Sequence[float]
) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantiles require at least one value")
    result = {}
    for probability in probabilities:
        if not 0 <= probability <= 1:
            raise ValueError("quantile probabilities must be in [0, 1]")
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        value = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
        result[f"p{round(probability * 100):02d}"] = value
    return result


def collate_features(items: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    maximum = max(int(item["length"]) for item in items)
    requested = torch.stack([item["requested_length"] for item in items])
    if not torch.all(requested == requested[0]):
        raise ValueError("all samples in a batch must request the same frame bucket")
    result: dict[str, Tensor] = {}
    for name in ("mel", "content", "f0", "rms"):
        result[name] = torch.stack([_pad(item[name], maximum) for item in items])
    lengths = torch.stack([item["length"] for item in items])
    result["length"] = lengths
    result["requested_length"] = requested
    result["speaker"] = torch.stack([item["speaker"] for item in items])
    result["mask"] = torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
    return result


def _matrix(tensor: Tensor, width: int, name: str) -> Tensor:
    tensor = torch.as_tensor(tensor).squeeze()
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if tensor.shape[-1] == width:
        return tensor
    if tensor.shape[0] == width:
        return tensor.transpose(0, 1)
    raise ValueError(f"{name} has no axis of width {width}: {tuple(tensor.shape)}")


def _vector(tensor: Tensor, name: str) -> Tensor:
    tensor = torch.as_tensor(tensor).squeeze()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    return tensor[:, None]


def _resize(tensor: Tensor, length: int) -> Tensor:
    if tensor.shape[0] == length:
        return tensor
    return (
        F.interpolate(
            tensor.transpose(0, 1).unsqueeze(0),
            size=length,
            mode="linear",
            align_corners=False,
        )
        .squeeze(0)
        .transpose(0, 1)
    )


def _pad(tensor: Tensor, length: int) -> Tensor:
    return F.pad(tensor, (0, 0, 0, length - tensor.shape[0]))
