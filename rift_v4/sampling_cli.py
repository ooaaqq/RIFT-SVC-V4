from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import V4Config
from .data import build_sampler
from .manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit expected dataset, speaker, and song sampling exposure"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = V4Config.load(args.config)
    entries = [
        entry
        for entry in load_manifest(args.manifest)
        if entry.split == "train" and entry.quality_status == "accepted"
    ]
    sampler = build_sampler(entries, config.sampling)
    audit = sampler.sampling_audit(
        max_steps=config.training.max_steps,
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        speaker_drop_probability=config.training.speaker_drop_probability,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    most_repeated_song = max(
        audit["songs"], key=lambda row: float(row["repeat_equivalent"])
    )
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "speakers": len(audit["speakers"]),
                "songs": len(audit["songs"]),
                "singleton_real_speaker_max_to_median": audit[
                    "singleton_real_speaker_max_to_median"
                ],
                "max_song_repeat_equivalent": most_repeated_song["repeat_equivalent"],
                "max_song": ":".join(
                    str(most_repeated_song[name])
                    for name in ("dataset", "speaker", "song")
                ),
                "song_repeat_equivalent_quantiles": audit[
                    "song_repeat_equivalent_quantiles"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
