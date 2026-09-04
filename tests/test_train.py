from collections import deque
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rift_v4.config import V4Config
from rift_v4.data import FeatureDataset
from rift_v4.features import MelStats
from rift_v4.manifest import ManifestEntry
from rift_v4.train import (
    _base_learning_rate_scale,
    _build_optimizer,
    _checkpoint,
    _initialize_from_ema,
    _prepare_run_directory,
    _shape_statistics,
    _use_ema_parameters,
    _validate_resume_config,
    _validation_loss,
)


def test_resume_performance_fork_only_allows_explicit_runtime_fields() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    checkpoint = asdict(config)
    current = deepcopy(checkpoint)
    current["performance"]["float8_recipe"] = "rowwise"

    changes = _validate_resume_config(checkpoint, current, frozenset({"float8_recipe"}))

    assert changes == {"float8_recipe": {"from": "rowwise_with_gw_hp", "to": "rowwise"}}
    current["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="outside"):
        _validate_resume_config(checkpoint, current, frozenset({"float8_recipe"}))


def test_shape_statistics_distinguish_valid_materialized_and_requested() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    observations = deque([(256, 256, 15_000, 64), (512, 480, 14_000, 32)], maxlen=1000)

    metrics = _shape_statistics(observations, config)

    assert metrics["canonical_shape_rate"] == 0.5
    assert metrics["unique_noncanonical_T_last_1000"] == [480]
    assert metrics["noncanonical_shape_rate_by_bucket"] == {
        "256": 0.0,
        "512": 1.0,
    }
    assert metrics["shape_window_steps"] == 2


def test_foundation_schedule_warms_up_then_stays_constant() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")

    assert _base_learning_rate_scale(config, 0) == pytest.approx(1 / 15_000)
    assert _base_learning_rate_scale(config, 14_999) == 1.0
    assert _base_learning_rate_scale(config, 15_000) == 1.0
    assert _base_learning_rate_scale(config, 499_999) == 1.0


class _FineTuneBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modulation = nn.Linear(4, 4)


class _FineTuneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.speaker = nn.Embedding(2, 4)
        self.time = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_FineTuneBlock()])
        self.final_modulation = nn.Linear(4, 4)
        self.backbone = nn.Linear(4, 4)


class _FineTuneSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FineTuneModel()


def test_foundation_optimizer_partitions_every_trainable_parameter_once() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    system = _FineTuneSystem()

    optimizer, summary = _build_optimizer(system, config)

    grouped = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable = [
        parameter for parameter in system.parameters() if parameter.requires_grad
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }
    assert summary["groups"]["backbone_decay"]["weight_decay"] == 0.01
    assert summary["groups"]["backbone_no_decay"]["weight_decay"] == 0
    assert summary["groups"]["speaker"]["weight_decay"] == 0
    assert summary["groups"]["speaker"]["parameter_names"] == ["model.speaker.weight"]
    assert summary["fused_adamw_requested"]
    assert not summary["fused_adamw_active"]


def test_finetune_optimizer_freezes_conditioning_and_preserves_null_row() -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/target-finetune.json")
    system = _FineTuneSystem()
    null_before = system.model.speaker.weight[1].detach().clone()

    optimizer, summary = _build_optimizer(system, config)

    assert not any(
        parameter.requires_grad for parameter in system.model.time.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in system.model.blocks[0].modulation.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in system.model.final_modulation.parameters()
    )
    assert summary["groups"]["speaker"]["base_learning_rate"] == 5e-5
    assert summary["groups"]["speaker"]["weight_decay"] == 0
    loss = system.model.speaker(torch.tensor([0])).sum()
    loss = loss + system.model.backbone(torch.ones(1, 4)).sum()
    loss.backward()
    optimizer.step()

    assert torch.equal(system.model.speaker.weight[1], null_before)


class _InitializationModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.speaker = nn.Embedding(2, dim)
        self.projection = nn.Linear(1, 1)


class _InitializationSystem(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.model = _InitializationModel(dim)


def test_foundation_initialization_uses_real_speaker_mean_and_null(
    tmp_path: Path,
) -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    target = _InitializationSystem(config.model.dim)
    source_embedding = torch.stack(
        (
            torch.full((config.model.dim,), 1.0),
            torch.full((config.model.dim,), 99.0),
            torch.full((config.model.dim,), 3.0),
            torch.full((config.model.dim,), -7.0),
        )
    )
    source_state = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    source_state["model.speaker.weight"] = source_embedding
    payload_config = asdict(config)
    payload_config["sampling"]["synthetic_datasets"] = ("Synthetic",)
    checkpoint = tmp_path / "foundation.pt"
    torch.save(
        {
            "schema_version": 4,
            "step": 130_000,
            "config": payload_config,
            "speaker_to_id": {"Real:a": 0, "Synthetic:b": 1, "Real:c": 2},
            "ema": source_state,
        },
        checkpoint,
    )

    details = _initialize_from_ema(target, checkpoint, {"Target:singer": 0}, config)

    assert torch.all(target.model.speaker.weight[0] == 2.0)
    assert torch.all(target.model.speaker.weight[1] == -7.0)
    assert details["source_state"] == "ema"
    assert details["source_real_speakers_averaged"] == 2


def test_ema_parameter_swap_restores_both_states() -> None:
    model = nn.Linear(2, 2)
    online = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    ema = {name: torch.full_like(value, 3) for name, value in online.items()}
    with _use_ema_parameters(model, ema):
        assert all(
            torch.equal(value, torch.full_like(value, 3))
            for value in model.state_dict().values()
        )
        assert all(torch.equal(ema[name], online[name]) for name in ema)
    assert all(torch.equal(model.state_dict()[name], online[name]) for name in online)
    assert all(torch.equal(value, torch.full_like(value, 3)) for value in ema.values())


class _ValidationSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, batch):
        flow = batch["mel"].sum() * 0 + 1
        return SimpleNamespace(
            flow=flow,
            flow_by_sample=torch.ones(batch["mel"].shape[0]),
        )


class _SpeakerLossSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, batch):
        values = batch["speaker"].float() + 1
        return SimpleNamespace(flow=values.mean(), flow_by_sample=values)


def test_validation_restores_training_mode(tmp_path: Path) -> None:
    prefix = tmp_path / "sample"
    torch.save(torch.zeros(8, 4), f"{prefix}.mel.pt")
    torch.save(torch.zeros(8, 3), f"{prefix}.content.pt")
    torch.save(torch.zeros(8), f"{prefix}.f0.pt")
    torch.save(torch.ones(8), f"{prefix}.rms.pt")
    entry = ManifestEntry(
        id="sample",
        dataset="A",
        speaker="singer",
        song="song",
        audio_path=str(tmp_path / "sample.wav"),
        feature_prefix=str(prefix),
        frames=8,
        duration_seconds=1.0,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        split="validation",
        quality_status="accepted",
    )
    dataset = FeatureDataset([entry], 4, 3, MelStats((0, 0, 0, 0), (1, 1, 1, 1), 8))
    system = _ValidationSystem()
    system.train()

    result = _validation_loss(system, dataset, torch.device("cpu"), 4, 1, {"A": 1.0})

    assert system.training
    assert system.projection.training
    assert result["real_speaker_macro_flow"] == 1.0
    assert result["by_song"] == {"A:singer:song": 1.0}
    assert result["condition_f0_voicing"]["overall"]["voiced_ratio"] == 0.0


def test_validation_selects_real_speaker_macro_and_records_songs(
    tmp_path: Path,
) -> None:
    entries = []
    for index, dataset in enumerate(("Real", "Synthetic")):
        prefix = tmp_path / f"sample-{index}"
        torch.save(torch.zeros(8, 4), f"{prefix}.mel.pt")
        torch.save(torch.zeros(8, 3), f"{prefix}.content.pt")
        torch.save(torch.zeros(8), f"{prefix}.f0.pt")
        torch.save(torch.ones(8), f"{prefix}.rms.pt")
        entries.append(
            ManifestEntry(
                id=f"sample-{index}",
                dataset=dataset,
                speaker="singer",
                song="held-out",
                audio_path=str(tmp_path / f"sample-{index}.wav"),
                feature_prefix=str(prefix),
                frames=8,
                duration_seconds=1.0,
                sample_rate=44_100,
                channels=1,
                audio_sha256="a" * 64,
                split="validation",
                quality_status="accepted",
            )
        )
    dataset = FeatureDataset(entries, 4, 3, MelStats((0, 0, 0, 0), (1, 1, 1, 1), 16))

    result = _validation_loss(
        _SpeakerLossSystem(),
        dataset,
        torch.device("cpu"),
        4,
        2,
        {"Real": 0.8, "Synthetic": 0.2},
        ("Synthetic",),
        2,
    )

    assert result["real_speaker_macro_flow"] == 1.0
    assert result["speaker_macro_flow"] == 1.5
    assert result["mixture_weighted_flow"] == pytest.approx(1.2)
    assert set(result["by_song"]) == {
        "Real:singer:held-out",
        "Synthetic:singer:held-out",
    }


def test_fresh_run_rejects_stale_identity_files(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text("{}")

    with pytest.raises(ValueError, match="fresh run directory"):
        _prepare_run_directory(tmp_path, None)


def test_audit_and_full_checkpoints_have_distinct_resume_state(
    tmp_path: Path,
) -> None:
    config = V4Config.load(Path(__file__).parents[1] / "config/v4.json")
    system = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(system.parameters())
    ema = {name: value.detach().clone() for name, value in system.state_dict().items()}
    entry = ManifestEntry(
        id="train",
        dataset="A",
        speaker="singer",
        song="song",
        audio_path="/train.wav",
        feature_prefix="/train",
        frames=8,
        duration_seconds=1.0,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        split="train",
        quality_status="accepted",
    )
    stats = MelStats((0,), (1,), 8)
    common = (
        system,
        optimizer,
        ema,
        1.0,
        2_000,
        2,
        0,
        0.1,
        config,
        {"A:singer": 0},
        [entry],
        stats,
    )

    _checkpoint(tmp_path / "step-0002000.pt", *common, kind="audit")
    _checkpoint(tmp_path / "resume-step-0002000.pt", *common, kind="full")
    audit = torch.load(
        tmp_path / "step-0002000.pt", map_location="cpu", weights_only=False
    )
    full = torch.load(
        tmp_path / "resume-step-0002000.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert audit["checkpoint_kind"] == "audit"
    assert "optimizer" not in audit
    assert "torch_rng_state" not in audit
    assert full["checkpoint_kind"] == "full"
    assert "optimizer" in full
    assert "torch_rng_state" in full
    assert full["learning_rate_multiplier"] == 1.0
