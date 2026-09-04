from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

from .config import V4Config
from .counterfactual import (
    SYNTHETIC_DATASETS,
    _entry_item,
    _item_waveform,
    _speaker_embedding,
    _stable_key,
    centroid_cache_key,
    manifest_fingerprint,
    speaker_target_margin,
)
from .evaluate import _file_sha256
from .manifest import ManifestEntry, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate A-to-B WavLM speaker margins with real audio anchors"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-spec", type=Path, required=True)
    parser.add_argument("--calibration-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversion-result", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = V4Config.load(args.config)
    entries = load_manifest(args.manifest)
    pair_spec = json.loads(args.pair_spec.read_text(encoding="utf-8"))
    calibration_spec = load_or_create_calibration_spec(
        args.calibration_spec,
        entries,
        pair_spec,
        args.pair_spec,
        args.seed,
    )
    payload = calibrate_speaker_metric(
        config,
        entries,
        pair_spec,
        calibration_spec,
        args.conversion_result,
        torch.device(args.device),
    )
    _atomic_json(args.output, payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def load_or_create_calibration_spec(
    path: Path,
    entries: list[ManifestEntry],
    pair_spec: dict[str, object],
    pair_spec_path: Path,
    seed: int,
) -> dict[str, object]:
    pair_sha256 = _file_sha256(pair_spec_path)
    manifest_sha256 = manifest_fingerprint(entries)
    by_id = {entry.id: entry for entry in entries}
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("schema_version") != 1:
            raise ValueError("unsupported speaker calibration schema")
        if spec.get("pair_spec_sha256") != pair_sha256:
            raise ValueError("counterfactual pair lock changed")
        if spec.get("manifest_fingerprint_sha256") != manifest_sha256:
            raise ValueError("calibration manifest changed")
        for row in spec["pairs"]:
            item = row["target_ground_truth"]
            entry = by_id.get(item["id"])
            if entry is None or entry.audio_sha256 != item["audio_sha256"]:
                raise ValueError(f"calibration audio changed: {item['id']}")
        return spec

    excluded = set(pair_spec.get("excluded_song_keys", []))
    grouped: dict[str, dict[str, list[ManifestEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        if (
            entry.quality_status == "accepted"
            and entry.split in {"validation", "test"}
            and entry.dataset not in SYNTHETIC_DATASETS
            and entry.frames >= int(pair_spec["frames"])
            and f"{entry.dataset}:{entry.song}" not in excluded
        ):
            grouped[entry.speaker_key][entry.song].append(entry)

    rows = []
    for index, pair in enumerate(pair_spec["pairs"]):
        target_speaker = pair["target_speaker"]
        reference_songs = {item["song"] for item in pair["target_references"]}
        candidates = [
            max(items, key=lambda entry: entry.frames)
            for song, items in grouped[target_speaker].items()
            if song not in reference_songs
        ]
        candidates.sort(
            key=lambda entry: _stable_key(
                seed, str(index), target_speaker, entry.song, entry.id
            )
        )
        if not candidates:
            raise ValueError(f"no independent target anchor for pair {index}")
        rows.append(
            {
                "pair": index,
                "target_ground_truth": _entry_item(
                    candidates[0],
                    int(pair_spec["frames"]),
                    seed + index,
                ),
            }
        )
    spec = {
        "schema_version": 1,
        "pair_spec_sha256": pair_sha256,
        "manifest_fingerprint_sha256": manifest_sha256,
        "seed": seed,
        "selection": "independent target song excluded from target reference centroid",
        "pairs": rows,
    }
    _atomic_json(path, spec)
    return spec


def normalized_transfer_progress(
    converted_margin: float, source_margin: float, target_margin: float
) -> float | None:
    separation = target_margin - source_margin
    if separation <= 0:
        return None
    return (converted_margin - source_margin) / separation


@torch.inference_mode()
def calibrate_speaker_metric(
    config: V4Config,
    entries: list[ManifestEntry],
    pair_spec: dict[str, object],
    calibration_spec: dict[str, object],
    conversion_results: list[Path],
    device: torch.device,
) -> dict[str, object]:
    by_id = {entry.id: entry for entry in entries}
    target_items = {
        int(row["pair"]): row["target_ground_truth"]
        for row in calibration_spec["pairs"]
    }
    processor = AutoFeatureExtractor.from_pretrained(
        config.evaluation.speaker_encoder_repository,
        revision=config.evaluation.speaker_encoder_revision,
    )
    encoder = WavLMForXVector.from_pretrained(
        config.evaluation.speaker_encoder_repository,
        revision=config.evaluation.speaker_encoder_revision,
    ).to(device).eval()
    embeddings: dict[tuple[object, ...], torch.Tensor] = {}

    def embedding(item: dict[str, object]) -> torch.Tensor:
        key = (item["id"], int(item["start_frame"]), int(item["frames"]))
        if key not in embeddings:
            embeddings[key] = _speaker_embedding(
                _item_waveform(by_id[item["id"]], item, config),
                config.sample_rate,
                processor,
                encoder,
                device,
            )
        return embeddings[key]

    def centroid(items: list[dict[str, object]]) -> torch.Tensor:
        key = ("centroid", *centroid_cache_key(items))
        if key not in embeddings:
            embeddings[key] = F.normalize(
                torch.stack([embedding(item) for item in items]).mean(0), dim=0
            )
        return embeddings[key]

    anchors = []
    for index, pair in enumerate(pair_spec["pairs"]):
        source_centroid = centroid(pair["source_references"])
        target_centroid = centroid(pair["target_references"])
        source_embedding = embedding(pair["source"])
        target_item = target_items[index]
        target_embedding = embedding(target_item)

        def anchor_row(
            value: torch.Tensor,
            source_reference: torch.Tensor,
            target_reference: torch.Tensor,
        ) -> dict[str, float]:
            to_target = float(value @ target_reference)
            to_source = float(value @ source_reference)
            return {
                "similarity_to_target": to_target,
                "similarity_to_source": to_source,
                "target_margin": speaker_target_margin(to_target, to_source),
            }

        source = anchor_row(source_embedding, source_centroid, target_centroid)
        target = anchor_row(target_embedding, source_centroid, target_centroid)
        anchors.append(
            {
                "pair": index,
                "source_speaker": pair["source_speaker"],
                "target_speaker": pair["target_speaker"],
                "source_song": pair["source"]["song"],
                "target_song": target_item["song"],
                "source_ground_truth": source,
                "target_ground_truth": target,
                "anchor_separation": (
                    target["target_margin"] - source["target_margin"]
                ),
            }
        )
        print(json.dumps({"completed_pairs": index + 1, "total_pairs": 32}), flush=True)

    aggregate: dict[str, object] = {
        "source_ground_truth": _mean_anchor(anchors, "source_ground_truth"),
        "target_ground_truth": _mean_anchor(anchors, "target_ground_truth"),
    }
    separations = [float(row["anchor_separation"]) for row in anchors]
    aggregate["anchor_separation"] = _summary(separations)
    aggregate["positive_anchor_pairs"] = sum(value > 0 for value in separations)

    conversions = {}
    anchors_by_pair = {int(row["pair"]): row for row in anchors}
    for result_path in conversion_results:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        target_rows = {
            int(row["pair"]): row
            for row in payload["samples"]["ema"]
            if row["condition"] == "target"
        }
        rows = []
        for pair, converted in sorted(target_rows.items()):
            anchor = anchors_by_pair[pair]
            source_margin = anchor["source_ground_truth"]["target_margin"]
            target_margin = anchor["target_ground_truth"]["target_margin"]
            progress = normalized_transfer_progress(
                float(converted["target_margin"]), source_margin, target_margin
            )
            rows.append(
                {
                    "pair": pair,
                    "converted_margin": converted["target_margin"],
                    "progress": progress,
                }
            )
        valid = [float(row["progress"]) for row in rows if row["progress"] is not None]
        source_mean = aggregate["source_ground_truth"]["target_margin"]
        target_mean = aggregate["target_ground_truth"]["target_margin"]
        converted_mean = sum(float(row["converted_margin"]) for row in rows) / len(rows)
        conversions[str(payload["checkpoint"])] = {
            "converted_margin_mean": converted_mean,
            "progress_ratio_of_means": normalized_transfer_progress(
                converted_mean, source_mean, target_mean
            ),
            "per_pair_progress": _summary(valid),
            "valid_pairs": len(valid),
            "invalid_nonpositive_anchor_pairs": len(rows) - len(valid),
            "samples": rows,
        }
    aggregate["conversions"] = conversions
    return {
        "schema_version": 1,
        "protocol": {
            "pairs": len(anchors),
            "speaker_encoder": {
                "repository": config.evaluation.speaker_encoder_repository,
                "revision": config.evaluation.speaker_encoder_revision,
            },
            "progress_definition": (
                "(converted_margin-source_ground_truth_margin)/"
                "(target_ground_truth_margin-source_ground_truth_margin)"
            ),
        },
        "aggregate": aggregate,
        "anchors": anchors,
    }


def _mean_anchor(rows: list[dict[str, object]], name: str) -> dict[str, float]:
    return {
        metric: sum(float(row[name][metric]) for row in rows) / len(rows)
        for metric in (
            "similarity_to_target",
            "similarity_to_source",
            "target_margin",
        )
    }


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
