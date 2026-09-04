from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import V4Config


@dataclass(frozen=True)
class PCNSFLock:
    repository: str
    revision: str
    checkpoint_filename: str
    checkpoint_size: int
    checkpoint_sha256: str | None
    extracted_checkpoint_sha256: str
    feature_contract: dict[str, int | str]
    training_policy: dict[str, bool | float | str]

    @classmethod
    def load(cls, path: str | Path) -> PCNSFLock:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["schema_version"] != 1:
            raise ValueError("unsupported PC-NSF lock schema")
        return cls(
            repository=payload["code"]["repository"],
            revision=payload["code"]["revision"],
            checkpoint_filename=payload["checkpoint"]["filename"],
            checkpoint_size=payload["checkpoint"]["size_bytes"],
            checkpoint_sha256=payload["checkpoint"]["sha256"],
            extracted_checkpoint_sha256=payload["checkpoint"]["extracted_sha256"],
            feature_contract=payload["feature_contract"],
            training_policy=payload["training_policy"],
        )

    def validate_contract(self, config: V4Config) -> None:
        expected = {
            "sample_rate": config.sample_rate,
            "hop_length": config.hop_length,
            "n_fft": config.mel.n_fft,
            "win_length": config.mel.win_length,
            "mel_channels": config.mel.channels,
            "fmin": config.mel.fmin,
            "fmax": config.mel.fmax,
            "log_base": config.mel.log_base,
            "mel_scale": config.mel.mel_scale,
            "mel_norm": config.mel.mel_norm,
            "power": config.mel.power,
            "center": config.mel.center,
            "pad_mode": config.mel.pad_mode,
            "log_clamp": config.mel.log_clamp,
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.feature_contract.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "PC-NSF feature contract mismatch: " + ", ".join(mismatches)
            )

    def validate_training_policy(self) -> None:
        expected: dict[str, bool | float | str] = {
            "mini_nsf": True,
            "pc_aug": True,
            "pc_aug_rate": 0.4,
            "precision": "32-true",
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.training_policy.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "PC-NSF training policy mismatch: " + ", ".join(mismatches)
            )

    def verify_checkout(self, checkout: str | Path) -> None:
        path = Path(checkout)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != self.revision:
            raise ValueError(f"PC-NSF checkout is {head}, expected {self.revision}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            raise ValueError("PC-NSF checkout has uncommitted modifications")

    def verify_checkpoint(self, archive: str | Path) -> str:
        path = Path(archive)
        if path.name != self.checkpoint_filename:
            raise ValueError(f"unexpected checkpoint archive name: {path.name}")
        if path.stat().st_size != self.checkpoint_size:
            raise ValueError("PC-NSF checkpoint archive size mismatch")
        digest = sha256_file(path)
        if self.checkpoint_sha256 is None:
            raise ValueError(
                "upstream published no digest; record this verified download in a "
                f"resolved local lock before training (observed sha256={digest})"
            )
        if digest != self.checkpoint_sha256:
            raise ValueError("PC-NSF checkpoint SHA-256 mismatch")
        return digest

    def verify_extracted_checkpoint(
        self, archive: str | Path, checkpoint: str | Path
    ) -> str:
        checkpoint_path = Path(checkpoint)
        local_digest = sha256_file(checkpoint_path)
        with zipfile.ZipFile(archive) as handle:
            matches = [
                name
                for name in handle.namelist()
                if Path(name).name == checkpoint_path.name and not name.endswith("/")
            ]
            if len(matches) != 1:
                raise ValueError(
                    "PC-NSF archive must contain exactly one matching checkpoint"
                )
            digest = hashlib.sha256()
            with handle.open(matches[0]) as member:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(block)
        if digest.hexdigest() != local_digest:
            raise ValueError("extracted PC-NSF checkpoint differs from archive")
        if local_digest != self.extracted_checkpoint_sha256:
            raise ValueError("installed PC-NSF checkpoint SHA-256 mismatch")
        return local_digest

    def verify_installed_checkpoint(self, checkpoint: str | Path) -> str:
        digest = sha256_file(checkpoint)
        if digest != self.extracted_checkpoint_sha256:
            raise ValueError("installed PC-NSF checkpoint SHA-256 mismatch")
        return digest


def verify_contentvec_snapshot(
    model_path: str | Path, lock_path: str | Path, config: V4Config
) -> str:
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("unsupported ContentVec lock schema")
    if payload["repository"] != config.content_encoder.repository:
        raise ValueError("ContentVec repository differs from V4 config")
    if payload["revision"] != config.content_encoder.revision:
        raise ValueError("ContentVec revision differs from V4 config")
    observed: dict[str, str] = {}
    for name in ("config", "weights"):
        expected = payload[name]
        path = Path(model_path) / expected["filename"]
        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"ContentVec {name} size mismatch")
        observed[name] = sha256_file(path)
        if observed[name] != expected["sha256"]:
            raise ValueError(f"ContentVec {name} SHA-256 mismatch")
    return observed["weights"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
