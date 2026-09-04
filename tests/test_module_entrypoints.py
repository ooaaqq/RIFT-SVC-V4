import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    (
        "rift_v4.manifest_cli",
        "rift_v4.qc",
        "rift_v4.split_cli",
        "rift_v4.features",
        "rift_v4.extract_content",
        "rift_v4.train",
        "rift_v4.sampling_cli",
        "rift_v4.infer",
        "rift_v4.assets_cli",
        "rift_v4.vocoder",
    ),
)
def test_python_module_entrypoint_exposes_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout
