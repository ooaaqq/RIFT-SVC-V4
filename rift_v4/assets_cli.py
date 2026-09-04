from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .config import V4Config
from .third_party import PCNSFLock


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and verify pinned V4 assets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-pc-nsf",
        help=(
            "verify checkout revision, clean state, checkpoint bytes, "
            "and feature contract"
        ),
    )
    verify.add_argument("--checkout", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--config", type=Path, default=Path("config/v4.json"))
    verify.add_argument(
        "--lock",
        type=Path,
        default=Path("third_party/pc_nsf_hifigan.lock.json"),
    )
    install = subparsers.add_parser(
        "install-pc-nsf",
        help="verify and extract the official pretrained checkpoint",
    )
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--output", type=Path, required=True)
    install.add_argument(
        "--lock",
        type=Path,
        default=Path("third_party/pc_nsf_hifigan.lock.json"),
    )
    args = parser.parse_args()

    if args.command == "install-pc-nsf":
        lock = PCNSFLock.load(args.lock)
        digest = lock.verify_checkpoint(args.archive)
        checkpoint_digest = _extract_checkpoint(lock, args.archive, args.output)
        print(
            f"installed pretrained PC-NSF {args.output} "
            f"archive_sha256={digest} checkpoint_sha256={checkpoint_digest}"
        )
        return

    config = V4Config.load(args.config)
    lock = PCNSFLock.load(args.lock)
    lock.validate_contract(config)
    lock.validate_training_policy()
    lock.verify_checkout(args.checkout)
    digest = lock.verify_checkpoint(args.archive)
    print(f"PC-NSF source, checkpoint, and feature contract verified: {digest}")


def _extract_checkpoint(lock: PCNSFLock, archive: Path, output: Path) -> str:
    expected = Path(lock.checkpoint_filename).with_suffix(".ckpt").name
    with zipfile.ZipFile(archive) as handle:
        matches = [
            name
            for name in handle.namelist()
            if Path(name).name == expected and not name.endswith("/")
        ]
        if len(matches) != 1:
            raise ValueError("official PC-NSF archive has no unique checkpoint")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=output.parent, prefix=f".{output.name}."
        )
        os.close(descriptor)
        try:
            with handle.open(matches[0]) as source, open(temporary, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.replace(temporary, output)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return lock.verify_extracted_checkpoint(archive, output)


if __name__ == "__main__":
    main()
