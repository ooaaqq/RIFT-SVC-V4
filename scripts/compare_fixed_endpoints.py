from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("full_raw_mse", "active_raw_mse", "silence_raw_mse")


def paired_summary(before: list[dict], after: list[dict], metric: str) -> dict:
    first = {item["entry_id"]: item.get(metric) for item in before}
    second = {item["entry_id"]: item.get(metric) for item in after}
    if set(first) != set(second):
        raise ValueError("fixed endpoint panels contain different samples")
    deltas = [
        second[key] - first[key]
        for key in sorted(first)
        if first[key] is not None and second[key] is not None
    ]
    if not deltas:
        return {"samples": 0, "paired_median_delta": None, "win_rate": None}
    return {
        "samples": len(deltas),
        "paired_median_delta": statistics.median(deltas),
        "paired_mean_delta": statistics.fmean(deltas),
        "win_rate": sum(delta < 0 for delta in deltas) / len(deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two fixed endpoint audits")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    result = {
        "before_step": before["checkpoint_step"],
        "after_step": after["checkpoint_step"],
        "results": {},
    }
    for state, by_frames in before["results"].items():
        result["results"][state] = {}
        for frames, by_panel in by_frames.items():
            result["results"][state][frames] = {}
            for panel, first in by_panel.items():
                second = after["results"][state][frames][panel]
                result["results"][state][frames][panel] = {
                    metric: paired_summary(first["samples"], second["samples"], metric)
                    for metric in METRICS
                }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
