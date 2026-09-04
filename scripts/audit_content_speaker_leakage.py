from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rift_v4.manifest import load_manifest


SPEAKERS = (
    "GTSinger:French-FR-Soprano-1",
    "GTSinger:Japanese-JA-Soprano-1",
    "GTSinger:Korean-KO-Soprano-2",
    "GTSinger:Spanish-ES-Bass-1",
    "M4Singer:Alto-5",
    "M4Singer:Alto-6",
    "M4Singer:Bass-1",
    "M4Singer:Tenor-5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-entries-per-speaker", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def pooled_feature(entry) -> np.ndarray:
    value = torch.as_tensor(torch.load(entry.content_feature_path, map_location="cpu", weights_only=True)).float()
    if value.ndim != 2 or value.shape[1] != 768:
        raise ValueError(f"{entry.id}: invalid ContentVec shape {tuple(value.shape)}")
    return torch.cat((value.mean(0), value.std(0, unbiased=False))).numpy()


def build_rows(entries, max_train: int, seed: int):
    grouped = defaultdict(list)
    for entry in entries:
        if entry.quality_status == "accepted" and entry.speaker_key in SPEAKERS and entry.content_feature_path:
            grouped[(entry.speaker_key, entry.split)].append(entry)
    rng = np.random.default_rng(seed)
    selected = []
    for speaker in SPEAKERS:
        train = sorted(grouped[(speaker, "train")], key=lambda item: item.id)
        if len(train) > max_train:
            indices = sorted(rng.choice(len(train), max_train, replace=False).tolist())
            train = [train[index] for index in indices]
        heldout = sorted(grouped[(speaker, "validation")] + grouped[(speaker, "test")], key=lambda item: item.id)
        if not train or not heldout:
            raise RuntimeError(f"{speaker}: train={len(train)} heldout={len(heldout)}")
        selected.extend(("train", entry) for entry in train)
        selected.extend(("heldout", entry) for entry in heldout)
    rows = []
    for index, (split, entry) in enumerate(selected, 1):
        rows.append({"split": split, "speaker": entry.speaker_key, "dataset": entry.dataset, "song": entry.song, "entry_id": entry.id, "feature": pooled_feature(entry)})
        if index % 200 == 0:
            print(f"loaded {index}/{len(selected)}", flush=True)
    return rows


def probe(rows, speakers):
    index = {speaker: value for value, speaker in enumerate(speakers)}
    train = [row for row in rows if row["split"] == "train" and row["speaker"] in index]
    test = [row for row in rows if row["split"] == "heldout" and row["speaker"] in index]
    x_train = np.stack([row["feature"] for row in train])
    y_train = np.array([index[row["speaker"]] for row in train])
    x_test = np.stack([row["feature"] for row in test])
    y_test = np.array([index[row["speaker"]] for row in test])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=20260902),
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)
    prediction = probability.argmax(1)
    groups = defaultdict(list)
    for row, prob, target in zip(test, probability, y_test, strict=True):
        groups[(row["speaker"], row["song"])].append((prob, target))
    song_probability = np.stack([np.mean([item[0] for item in values], axis=0) for values in groups.values()])
    song_target = np.array([values[0][1] for values in groups.values()])

    centroids = np.stack([x_train[y_train == label].mean(0) for label in range(len(speakers))])
    scale = x_train.std(0).clip(1e-6)
    normalized_test = (x_test - x_train.mean(0)) / scale
    normalized_centroids = (centroids - x_train.mean(0)) / scale
    normalized_test /= np.linalg.norm(normalized_test, axis=1, keepdims=True).clip(1e-12)
    normalized_centroids /= np.linalg.norm(normalized_centroids, axis=1, keepdims=True).clip(1e-12)
    centroid_prediction = (normalized_test @ normalized_centroids.T).argmax(1)

    return {
        "speakers": list(speakers),
        "chance_accuracy": 1 / len(speakers),
        "train_entries": len(train),
        "heldout_entries": len(test),
        "train_songs": len({(row["speaker"], row["song"]) for row in train}),
        "heldout_songs": len(groups),
        "entry_accuracy": float(accuracy_score(y_test, prediction)),
        "entry_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "entry_top3_accuracy": float(top_k_accuracy_score(y_test, probability, k=min(3, len(speakers)), labels=np.arange(len(speakers)))),
        "song_accuracy": float(accuracy_score(song_target, song_probability.argmax(1))),
        "song_balanced_accuracy": float(balanced_accuracy_score(song_target, song_probability.argmax(1))),
        "nearest_centroid_accuracy": float(accuracy_score(y_test, centroid_prediction)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=np.arange(len(speakers))).tolist(),
        "counts": {
            speaker: {
                "train_entries": sum(row["split"] == "train" and row["speaker"] == speaker for row in rows),
                "heldout_entries": sum(row["split"] == "heldout" and row["speaker"] == speaker for row in rows),
                "train_songs": len({row["song"] for row in rows if row["split"] == "train" and row["speaker"] == speaker}),
                "heldout_songs": len({row["song"] for row in rows if row["split"] == "heldout" and row["speaker"] == speaker}),
            }
            for speaker in speakers
        },
    }


def main() -> None:
    args = parse_args()
    rows = build_rows(load_manifest(args.manifest), args.max_train_entries_per_speaker, args.seed)
    payload = {
        "schema_version": 1,
        "feature": "ContentVec temporal mean + standard deviation (1536 dimensions)",
        "split": "train songs to V4 validation/test songs",
        "probe": "standardized L2 logistic regression, C=0.1; plus nearest centroid",
        "all_eight": probe(rows, SPEAKERS),
        "within_dataset": {
            dataset: probe(rows, tuple(speaker for speaker in SPEAKERS if speaker.startswith(dataset + ":")))
            for dataset in ("GTSinger", "M4Singer")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "all_eight": {key: value for key, value in payload["all_eight"].items() if key.endswith("accuracy") or key.endswith("entries") or key.endswith("songs")},
        "within_dataset": {dataset: {key: value for key, value in result.items() if key.endswith("accuracy")} for dataset, result in payload["within_dataset"].items()},
    }, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
