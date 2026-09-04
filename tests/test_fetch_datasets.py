from pathlib import Path

import pytest

from rift_v4.datasets import stage_gtsinger


def _gtsinger_wav(
    root: Path,
    group: str,
    name: str = "0000.wav",
    technique: str = "Breathy",
) -> Path:
    path = root / "Chinese" / "ZH-Alto-1" / technique / "song" / group / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(group.encode())
    return path


def test_stage_gtsinger_preserves_control_and_technique_groups(tmp_path: Path) -> None:
    source = tmp_path / "source"
    control = _gtsinger_wav(source, "Control_Group")
    technique = _gtsinger_wav(source, "Breathy_Group")
    second_control = _gtsinger_wav(source, "Control_Group", technique="Falsetto")
    _gtsinger_wav(source, "Paired_Speech_Group")

    destination = tmp_path / "staged"
    assert stage_gtsinger(source, destination) == 3
    song = destination / "Chinese-ZH-Alto-1" / "song"
    assert (song / "Breathy__Control_Group__0000.wav").resolve() == control.resolve()
    assert (song / "Breathy__Breathy_Group__0000.wav").resolve() == technique.resolve()
    assert (
        song / "Falsetto__Control_Group__0000.wav"
    ).resolve() == second_control.resolve()
    assert len(list(destination.rglob("*.wav"))) == 3


def test_stage_gtsinger_rejects_existing_target_from_another_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _gtsinger_wav(first, "Control_Group")
    _gtsinger_wav(second, "Control_Group")
    destination = tmp_path / "staged"

    stage_gtsinger(first, destination)
    with pytest.raises(FileExistsError, match="staging collision"):
        stage_gtsinger(second, destination)


def test_stage_gtsinger_rejects_stale_destination_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _gtsinger_wav(source, "Control_Group")
    destination = tmp_path / "staged"
    stale = destination / "speaker" / "song" / "0000.wav"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="stale or unexpected"):
        stage_gtsinger(source, destination)
