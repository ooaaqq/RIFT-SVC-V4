from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MelConfig:
    channels: int
    n_fft: int
    win_length: int
    fmin: int
    fmax: int
    log_base: str
    mel_scale: str
    mel_norm: str
    power: float
    center: bool
    pad_mode: str
    log_clamp: float
    normalization: str


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    depth: int
    head_dim: int
    ff_hidden_dim: int
    kernel_size: int
    content_dim: int
    attention_scale: str


@dataclass(frozen=True)
class ContentEncoderConfig:
    repository: str
    revision: str
    sample_rate: int
    phase_shift_seconds: float


@dataclass(frozen=True)
class SamplingConfig:
    frame_buckets: tuple[int, ...]
    bucket_probabilities: tuple[float, ...]
    dataset_probabilities: dict[str, float]
    speaker_duration_exponent: float
    speaker_probability_floor_ratio: float
    speaker_probability_ceiling_ratio: float
    song_duration_exponent: float
    song_probability_floor_ratio: float
    song_probability_ceiling_ratio: float
    dataset_families: dict[str, str]
    family_probability_caps: dict[str, float]
    synthetic_datasets: tuple[str, ...]
    max_singleton_real_speaker_median_ratio: float
    batch_size: int
    batch_frame_budget: int
    steps_per_epoch: int
    seed: int
    voiced_crop_probability: float


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    learning_rate: float
    speaker_learning_rate: float
    weight_decay: float
    freeze_timestep_and_modulation: bool
    warmup_steps: int
    learning_rate_schedule: str
    grad_clip_norm: float
    speaker_drop_probability: float
    precision: str
    ema_decay: float
    gradient_accumulation_steps: int
    log_every_steps: int
    audit_checkpoint_every_steps: int
    resume_checkpoint_every_steps: int
    validation_every_steps: int
    min_learning_rate_ratio: float | None


@dataclass(frozen=True)
class PerformanceConfig:
    matmul_fp32_precision: str
    conv_fp32_precision: str
    sdpa_backend: str
    compile_model: bool
    compile_mode: str
    fused_adamw: bool
    float8_training: bool
    float8_recipe: str
    compile_warmup_buckets: bool
    compile_warmup_steps_per_bucket: int


@dataclass(frozen=True)
class InferenceConfig:
    time_schedule: str


@dataclass(frozen=True)
class EvaluationConfig:
    audio_panel_steps: tuple[int, ...]
    audio_panel_samples: int
    audio_panel_frames: int
    endpoint_panel_frames: tuple[int, ...]
    endpoint_panel_samples: int
    rf_audit_timesteps: tuple[float, ...]
    counterfactual_pairs: int
    speaker_encoder_repository: str
    speaker_encoder_revision: str
    inference_steps: int
    guidance: float
    validation_recordings_per_song: int


@dataclass(frozen=True)
class VocoderConfig:
    architecture: str
    sample_rate: int
    hop_length: int
    mel_channels: int
    fmin: int
    fmax: int
    log_base: str
    mel_scale: str
    mel_norm: str
    power: float
    center: bool
    pad_mode: str
    log_clamp: float


@dataclass(frozen=True)
class V4Config:
    schema_version: int
    sample_rate: int
    hop_length: int
    mel: MelConfig
    model: ModelConfig
    content_encoder: ContentEncoderConfig
    sampling: SamplingConfig
    training: TrainingConfig
    performance: PerformanceConfig
    inference: InferenceConfig
    evaluation: EvaluationConfig
    vocoder: VocoderConfig

    @classmethod
    def load(cls, path: str | Path) -> V4Config:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            schema_version=int(payload["schema_version"]),
            sample_rate=int(payload["sample_rate"]),
            hop_length=int(payload["hop_length"]),
            mel=MelConfig(**payload["mel"]),
            model=ModelConfig(**payload["model"]),
            content_encoder=ContentEncoderConfig(**payload["content_encoder"]),
            sampling=SamplingConfig(
                frame_buckets=tuple(payload["sampling"]["frame_buckets"]),
                bucket_probabilities=tuple(payload["sampling"]["bucket_probabilities"]),
                dataset_probabilities=dict(
                    payload["sampling"]["dataset_probabilities"]
                ),
                speaker_duration_exponent=float(
                    payload["sampling"]["speaker_duration_exponent"]
                ),
                speaker_probability_floor_ratio=float(
                    payload["sampling"]["speaker_probability_floor_ratio"]
                ),
                speaker_probability_ceiling_ratio=float(
                    payload["sampling"]["speaker_probability_ceiling_ratio"]
                ),
                song_duration_exponent=float(
                    payload["sampling"]["song_duration_exponent"]
                ),
                song_probability_floor_ratio=float(
                    payload["sampling"]["song_probability_floor_ratio"]
                ),
                song_probability_ceiling_ratio=float(
                    payload["sampling"]["song_probability_ceiling_ratio"]
                ),
                dataset_families=dict(payload["sampling"]["dataset_families"]),
                family_probability_caps=dict(
                    payload["sampling"]["family_probability_caps"]
                ),
                synthetic_datasets=tuple(payload["sampling"]["synthetic_datasets"]),
                max_singleton_real_speaker_median_ratio=float(
                    payload["sampling"]["max_singleton_real_speaker_median_ratio"]
                ),
                batch_size=int(payload["sampling"]["batch_size"]),
                batch_frame_budget=int(payload["sampling"]["batch_frame_budget"]),
                steps_per_epoch=int(payload["sampling"]["steps_per_epoch"]),
                seed=int(payload["sampling"]["seed"]),
                voiced_crop_probability=float(
                    payload["sampling"]["voiced_crop_probability"]
                ),
            ),
            training=TrainingConfig(**payload["training"]),
            performance=PerformanceConfig(**payload["performance"]),
            inference=InferenceConfig(**payload["inference"]),
            evaluation=EvaluationConfig(
                audio_panel_steps=tuple(payload["evaluation"]["audio_panel_steps"]),
                audio_panel_samples=int(payload["evaluation"]["audio_panel_samples"]),
                audio_panel_frames=int(payload["evaluation"]["audio_panel_frames"]),
                endpoint_panel_frames=tuple(
                    payload["evaluation"]["endpoint_panel_frames"]
                ),
                endpoint_panel_samples=int(
                    payload["evaluation"]["endpoint_panel_samples"]
                ),
                rf_audit_timesteps=tuple(
                    float(value)
                    for value in payload["evaluation"]["rf_audit_timesteps"]
                ),
                counterfactual_pairs=int(payload["evaluation"]["counterfactual_pairs"]),
                speaker_encoder_repository=str(
                    payload["evaluation"]["speaker_encoder_repository"]
                ),
                speaker_encoder_revision=str(
                    payload["evaluation"]["speaker_encoder_revision"]
                ),
                inference_steps=int(payload["evaluation"]["inference_steps"]),
                guidance=float(payload["evaluation"]["guidance"]),
                validation_recordings_per_song=int(
                    payload["evaluation"]["validation_recordings_per_song"]
                ),
            ),
            vocoder=VocoderConfig(**payload["vocoder"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 4:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.sample_rate <= 0 or self.hop_length <= 0:
            raise ValueError("sample rate and hop length must be positive")
        if (
            min(
                self.model.dim,
                self.model.depth,
                self.model.head_dim,
                self.model.ff_hidden_dim,
                self.model.kernel_size,
            )
            <= 0
        ):
            raise ValueError("model dimensions and depth must be positive")
        if self.model.dim % self.model.head_dim:
            raise ValueError("model dim must be divisible by head_dim")
        if self.model.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.model.ff_hidden_dim % 256:
            raise ValueError("FFN hidden dimension must be divisible by 256")
        if self.model.attention_scale != "sqrt_head":
            raise ValueError("V4 requires attention_scale='sqrt_head'")
        if len(self.sampling.frame_buckets) != len(self.sampling.bucket_probabilities):
            raise ValueError("frame bucket and probability lengths differ")
        if abs(sum(self.sampling.bucket_probabilities) - 1.0) > 1e-6:
            raise ValueError("bucket probabilities must sum to 1")
        if any(value <= 0 for value in self.sampling.bucket_probabilities):
            raise ValueError("bucket probabilities must be positive")
        if tuple(sorted(self.sampling.frame_buckets)) != self.sampling.frame_buckets:
            raise ValueError("frame buckets must be strictly ordered")
        if any(value <= 0 for value in self.sampling.frame_buckets):
            raise ValueError("frame buckets must be positive")
        if self.sampling.batch_frame_budget < max(self.sampling.frame_buckets):
            raise ValueError("batch frame budget cannot fit the longest bucket")
        if self.sampling.batch_size <= 0 or self.sampling.steps_per_epoch <= 0:
            raise ValueError("batch size and steps per epoch must be positive")
        if not self.sampling.dataset_probabilities or any(
            value <= 0 for value in self.sampling.dataset_probabilities.values()
        ):
            raise ValueError("dataset probabilities must be non-empty and positive")
        if (
            any(
                not math.isfinite(value)
                for value in self.sampling.dataset_probabilities.values()
            )
            or abs(sum(self.sampling.dataset_probabilities.values()) - 1.0) > 1e-6
        ):
            raise ValueError("dataset probabilities must sum to 1")
        if not 0 <= self.sampling.voiced_crop_probability <= 1:
            raise ValueError("voiced crop probability must be in [0, 1]")
        if not 0 <= self.sampling.speaker_duration_exponent <= 1:
            raise ValueError("speaker duration exponent must be in [0, 1]")
        floor = self.sampling.speaker_probability_floor_ratio
        ceiling = self.sampling.speaker_probability_ceiling_ratio
        if not 0 < floor <= 1 <= ceiling:
            raise ValueError("speaker probability bounds must contain 1")
        if not 0 <= self.sampling.song_duration_exponent <= 1:
            raise ValueError("song duration exponent must be in [0, 1]")
        song_floor = self.sampling.song_probability_floor_ratio
        song_ceiling = self.sampling.song_probability_ceiling_ratio
        if not 0 < song_floor <= 1 <= song_ceiling:
            raise ValueError("song probability bounds must contain 1")
        datasets = set(self.sampling.dataset_probabilities)
        if not set(self.sampling.dataset_families) <= datasets:
            raise ValueError("dataset families refer to an unknown dataset")
        if not set(self.sampling.synthetic_datasets) <= datasets:
            raise ValueError("synthetic datasets refer to an unknown dataset")
        if any(
            not math.isfinite(value) or not 0 < value <= 1
            for value in self.sampling.family_probability_caps.values()
        ):
            raise ValueError("family probability caps must be in (0, 1]")
        family_totals: dict[str, float] = {}
        for dataset, probability in self.sampling.dataset_probabilities.items():
            family = self.sampling.dataset_families.get(dataset, dataset)
            family_totals[family] = family_totals.get(family, 0.0) + probability
        exceeded = {
            family: family_totals.get(family, 0.0)
            for family, cap in self.sampling.family_probability_caps.items()
            if family_totals.get(family, 0.0) > cap + 1e-12
        }
        if exceeded:
            raise ValueError(f"dataset family probability cap exceeded: {exceeded}")
        if self.sampling.max_singleton_real_speaker_median_ratio < 1:
            raise ValueError("singleton real speaker median ratio must be at least 1")
        if len(self.content_encoder.revision) != 40:
            raise ValueError(
                "ContentVec must be pinned to a full 40-character revision"
            )
        if self.content_encoder.sample_rate != 16_000:
            raise ValueError("the pinned ContentVec input rate is 16 kHz")
        if not math.isfinite(self.content_encoder.phase_shift_seconds) or (
            self.content_encoder.phase_shift_seconds <= 0
        ):
            raise ValueError("content encoder phase shift must be positive")
        if not 0 <= self.training.speaker_drop_probability < 1:
            raise ValueError("speaker drop probability must be in [0, 1)")
        if self.training.precision not in {"fp32", "bf16"}:
            raise ValueError("training precision must be fp32 or bf16")
        if self.training.max_steps <= 0:
            raise ValueError("training steps must be positive")
        if not math.isfinite(self.training.learning_rate) or (
            self.training.learning_rate <= 0
        ):
            raise ValueError("training learning rate must be finite and positive")
        if not math.isfinite(self.training.speaker_learning_rate) or (
            self.training.speaker_learning_rate <= 0
        ):
            raise ValueError("speaker learning rate must be finite and positive")
        if not math.isfinite(self.training.weight_decay) or (
            self.training.weight_decay < 0
        ):
            raise ValueError("training weight decay must be finite and non-negative")
        if (
            self.training.freeze_timestep_and_modulation
            and self.training.speaker_drop_probability != 0
        ):
            raise ValueError(
                "single-singer frozen-conditioning fine-tuning requires zero "
                "speaker dropout"
            )
        if not math.isfinite(self.training.grad_clip_norm) or (
            self.training.grad_clip_norm <= 0
        ):
            raise ValueError("training gradient clip must be finite and positive")
        if not 0 <= self.training.warmup_steps < self.training.max_steps:
            raise ValueError("training warmup must be shorter than max steps")
        if self.training.learning_rate_schedule not in {
            "constant_after_warmup",
            "cosine",
        }:
            raise ValueError("unsupported training learning-rate schedule")
        if not math.isfinite(self.training.ema_decay) or not (
            0 < self.training.ema_decay < 1
        ):
            raise ValueError("EMA decay must be between zero and one")
        if self.training.learning_rate_schedule == "cosine":
            ratio = self.training.min_learning_rate_ratio
            if ratio is None or not math.isfinite(ratio) or not 0 <= ratio < 1:
                raise ValueError(
                    "cosine schedule requires a minimum learning-rate ratio in [0, 1)"
                )
        elif self.training.min_learning_rate_ratio is not None:
            raise ValueError(
                "constant schedule must not define a minimum learning-rate ratio"
            )
        if self.performance.matmul_fp32_precision not in {"ieee", "tf32"}:
            raise ValueError("unsupported FP32 matmul precision")
        if self.performance.conv_fp32_precision not in {"ieee", "tf32"}:
            raise ValueError("unsupported FP32 convolution precision")
        if self.performance.sdpa_backend not in {"flash", "cudnn"}:
            raise ValueError("V7 requires a fused Flash or cuDNN SDPA backend")
        if self.performance.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("unsupported torch.compile mode")
        if self.performance.float8_training and not self.performance.compile_model:
            raise ValueError("FP8 training requires torch.compile")
        if self.performance.float8_recipe not in {
            "rowwise",
            "rowwise_with_gw_hp",
            "tensorwise",
        }:
            raise ValueError("unsupported FP8 training recipe")
        if self.performance.compile_warmup_steps_per_bucket <= 0:
            raise ValueError("compile warmup steps per bucket must be positive")
        if self.inference.time_schedule not in {"linear", "cosine"}:
            raise ValueError("inference time schedule must be linear or cosine")
        if (
            not self.evaluation.audio_panel_steps
            or tuple(sorted(set(self.evaluation.audio_panel_steps)))
            != self.evaluation.audio_panel_steps
            or any(
                step <= 0 or step > self.training.max_steps
                for step in self.evaluation.audio_panel_steps
            )
        ):
            raise ValueError("audio panel steps must be unique and within training")
        if min(
            self.evaluation.audio_panel_samples,
            self.evaluation.audio_panel_frames,
            self.evaluation.endpoint_panel_samples,
            self.evaluation.counterfactual_pairs,
            self.evaluation.inference_steps,
            self.evaluation.validation_recordings_per_song,
        ) <= 0 or not (
            math.isfinite(self.evaluation.guidance) and self.evaluation.guidance > 0
        ):
            raise ValueError("audio panel sizes and guidance must be valid")
        if (
            not self.evaluation.endpoint_panel_frames
            or tuple(sorted(set(self.evaluation.endpoint_panel_frames)))
            != self.evaluation.endpoint_panel_frames
            or any(frames <= 0 for frames in self.evaluation.endpoint_panel_frames)
        ):
            raise ValueError("endpoint panel frames must be unique and positive")
        if not self.evaluation.rf_audit_timesteps or any(
            not math.isfinite(value) or not 0 < value < 1
            for value in self.evaluation.rf_audit_timesteps
        ):
            raise ValueError("RF audit timesteps must be inside (0, 1)")
        if len(self.evaluation.speaker_encoder_revision) != 40:
            raise ValueError("speaker encoder must use a full pinned revision")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient accumulation must be positive")
        if self.training.log_every_steps <= 0:
            raise ValueError("log interval must be positive")
        if (
            min(
                self.training.audit_checkpoint_every_steps,
                self.training.resume_checkpoint_every_steps,
                self.training.validation_every_steps,
            )
            <= 0
        ):
            raise ValueError(
                "checkpoint, validation, and retention values must be positive"
            )
        self._validate_vocoder_contract()

    def _validate_vocoder_contract(self) -> None:
        pairs: tuple[tuple[str, Any, Any], ...] = (
            ("sample_rate", self.sample_rate, self.vocoder.sample_rate),
            ("hop_length", self.hop_length, self.vocoder.hop_length),
            ("mel_channels", self.mel.channels, self.vocoder.mel_channels),
            ("fmin", self.mel.fmin, self.vocoder.fmin),
            ("fmax", self.mel.fmax, self.vocoder.fmax),
            ("log_base", self.mel.log_base, self.vocoder.log_base),
            ("mel_scale", self.mel.mel_scale, self.vocoder.mel_scale),
            ("mel_norm", self.mel.mel_norm, self.vocoder.mel_norm),
            ("power", self.mel.power, self.vocoder.power),
            ("center", self.mel.center, self.vocoder.center),
            ("pad_mode", self.mel.pad_mode, self.vocoder.pad_mode),
            ("log_clamp", self.mel.log_clamp, self.vocoder.log_clamp),
        )
        mismatches = [name for name, left, right in pairs if left != right]
        if mismatches:
            raise ValueError(
                "RIFT/vocoder feature contract mismatch: " + ", ".join(mismatches)
            )
