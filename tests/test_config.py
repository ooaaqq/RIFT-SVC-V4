from pathlib import Path

import pytest

from rift_v4.config import V4Config


def test_canonical_config_contract() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    assert config.model.dim == 1024
    assert config.model.depth == 16
    assert config.model.head_dim == 64
    assert config.model.ff_hidden_dim == 2816
    assert config.model.attention_scale == "sqrt_head"
    assert config.content_encoder.phase_shift_seconds == 0.01
    assert config.inference.time_schedule == "cosine"
    assert config.sampling.frame_buckets == (256, 384, 512)
    assert config.sampling.speaker_duration_exponent == 0.5
    assert config.sampling.song_duration_exponent == 0.5
    assert config.sampling.dataset_probabilities["Opencpop"] == 0.015
    assert config.sampling.family_probability_caps["Opencpop-family"] == 0.035
    assert config.evaluation.validation_recordings_per_song == 2
    assert config.evaluation.endpoint_panel_frames == (512, 768)
    assert config.evaluation.endpoint_panel_samples == 16
    assert config.evaluation.counterfactual_pairs == 32
    assert config.training.audit_checkpoint_every_steps == 2_000
    assert config.training.resume_checkpoint_every_steps == 5_000
    assert config.training.learning_rate_schedule == "constant_after_warmup"
    assert config.training.learning_rate == 1.5e-4
    assert config.training.speaker_learning_rate == 2e-4
    assert config.training.speaker_drop_probability == 0.05
    assert config.sampling.batch_frame_budget == 16_384
    assert config.performance.matmul_fp32_precision == "tf32"
    assert config.performance.conv_fp32_precision == "tf32"
    assert config.performance.sdpa_backend == "cudnn"
    assert config.performance.compile_model
    assert config.performance.compile_mode == "max-autotune"
    assert config.performance.fused_adamw
    assert not config.performance.float8_training
    assert config.performance.float8_recipe == "rowwise_with_gw_hp"
    assert config.performance.compile_warmup_buckets
    assert config.performance.compile_warmup_steps_per_bucket == 2
    assert config.vocoder.hop_length == config.hop_length


def test_vocoder_contract_rejects_mismatch(tmp_path: Path) -> None:
    source = (Path(__file__).parents[1] / "config/v4.json").read_text()
    target = tmp_path / "bad.json"
    target.write_text(source.replace('"mel_channels": 128', '"mel_channels": 80'))
    with pytest.raises(ValueError, match="mel_channels"):
        V4Config.load(target)
