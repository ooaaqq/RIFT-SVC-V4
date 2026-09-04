import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rift_v4.content_encoder import FrozenContentEncoder
from rift_v4.extract_content import _existing_content_is_reusable, encode_chunked
from rift_v4.manifest import ManifestEntry
from rift_v4.third_party import PCNSFLock, sha256_file
from rift_v4.train import _verify_content_provenance


class FakeBackbone(nn.Module):
    def __init__(self, dim: int, layers: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(1, dim)
        self.config = SimpleNamespace(hidden_size=dim)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layers)]
        )
        self.encoder.layer_norm = nn.LayerNorm(dim)

    def forward(self, input_values, attention_mask, return_dict=True):
        del attention_mask, return_dict
        frames = max(1, input_values.shape[1] // 4)
        value = self.input(input_values[:, : frames * 4 : 4, None])
        for layer in self.encoder.layers:
            value = torch.tanh(layer(value))
        return SimpleNamespace(last_hidden_state=self.encoder.layer_norm(value))


def _encoder() -> FrozenContentEncoder:
    return FrozenContentEncoder(FakeBackbone(16), 16)


def test_contentvec_is_fully_frozen_and_stays_in_eval_mode() -> None:
    encoder = _encoder()
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert not encoder.training
    assert not encoder.backbone.training


def test_raw_chunked_content_extraction() -> None:
    encoder = _encoder()
    waveform = torch.arange(1030, dtype=torch.float32)
    content = encode_chunked(encoder, waveform, 1000, 0.1, 0.02, torch.device("cpu"))
    assert content.ndim == 2
    assert content.shape[1] == 16
    assert content.shape[0] % 2 == 0
    assert content.shape[0] >= 2
    assert not torch.equal(content[0], content[1])
    assert content.shape[0] % 2 == 0
    assert content.shape[0] >= 2


def test_rift_requires_matching_encoder_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"versioned encoder")
    digest = sha256_file(checkpoint)
    feature = tmp_path / "content.pt"
    torch.save(torch.randn(4, 16), feature)
    entry = ManifestEntry(
        id="one",
        dataset="A",
        speaker="singer",
        song="song",
        audio_path=str(tmp_path / "audio.wav"),
        feature_prefix=str(tmp_path / "features"),
        frames=4,
        duration_seconds=1,
        sample_rate=44_100,
        channels=1,
        audio_sha256="a" * 64,
        quality_status="accepted",
        content_feature_path=str(feature),
        content_encoder_id=f"contentvec-dualphase10ms-v1:{digest[:16]}",
        content_encoder_sha256=digest,
    )
    _verify_content_provenance([entry])

    legacy = ManifestEntry.from_dict(
        {**entry.__dict__, "content_encoder_id": f"contentvec-raw:{digest[:16]}"}
    )
    with pytest.raises(ValueError, match="dual-phase"):
        _verify_content_provenance([legacy])

    assert _existing_content_is_reusable(
        entry, entry.content_encoder_id or "", digest, 16
    )
    assert not _existing_content_is_reusable(
        entry, entry.content_encoder_id or "", "b" * 64, 16
    )


def test_pc_nsf_lock_matches_v4_contract() -> None:
    root = Path(__file__).parents[1]
    from rift_v4.config import V4Config

    config = V4Config.load(root / "config/v4.json")
    lock = PCNSFLock.load(root / "third_party/pc_nsf_hifigan.lock.json")
    lock.validate_contract(config)
    lock.validate_training_policy()
    assert lock.revision == "4d0889c4c180c75ad3000cc565864656344f8190"


def test_pc_nsf_extracted_checkpoint_must_match_archive(tmp_path: Path) -> None:
    checkpoint = tmp_path / "generator.ckpt"
    checkpoint.write_bytes(b"exact checkpoint")
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(checkpoint, "release/generator.ckpt")
    lock = PCNSFLock(
        repository="https://example.invalid",
        revision="abc",
        checkpoint_filename=archive.name,
        checkpoint_size=archive.stat().st_size,
        checkpoint_sha256=sha256_file(archive),
        extracted_checkpoint_sha256=sha256_file(checkpoint),
        feature_contract={},
        training_policy={},
    )
    assert lock.verify_extracted_checkpoint(archive, checkpoint) == sha256_file(
        checkpoint
    )
