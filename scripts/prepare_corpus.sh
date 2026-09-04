#!/usr/bin/env bash
set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 DATASET SOURCE_DIR MANIFEST_STEM" >&2
  exit 2
fi

dataset="$1"
source="$2"
stem="$3"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${RIFT_DATA_ROOT:?set RIFT_DATA_ROOT to the training data directory}"
contentvec="${CONTENTVEC_MODEL:?set CONTENTVEC_MODEL to the pinned local snapshot}"
python="${RIFT_PYTHON:-python}"
manifest="$data/manifests/$stem"

mkdir -p "$data/manifests" "$data/records"
cd "$repo"

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_cli \
  --config config/v4.json \
  --source "$dataset=$source" \
  --features-root "$data/features" \
  --output "$manifest.pending.jsonl"

PYTHONPATH="$repo" "$python" -m rift_v4.qc \
  --config config/v4.json \
  --manifest "$manifest.pending.jsonl" \
  --output "$manifest.qc.jsonl"

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools audit \
  --manifest "$manifest.qc.jsonl" \
  --catalog config/datasets.json 2>&1 | tee "$data/records/$stem-qc.json"

PYTHONPATH="$repo" "$python" -m rift_v4.split_cli \
  --manifest "$manifest.qc.jsonl" \
  --output "$manifest.split.jsonl" \
  --validation-ratio 0.05 --test-ratio 0.05 --seed 2026

PYTHONPATH="$repo" "$python" -m rift_v4.features extract \
  --config config/v4.json \
  --manifest "$manifest.split.jsonl" \
  --device cuda --execute-extraction 2>&1 | tee "$data/records/$stem-features.log"

PYTHONPATH="$repo" "$python" -m rift_v4.extract_content \
  --config config/v4.json \
  --manifest "$manifest.split.jsonl" \
  --base-model-path "$contentvec" \
  --contentvec-lock third_party/contentvec.lock.json \
  --features-root "$data/raw-content" \
  --output-manifest "$manifest.content.jsonl" \
  --device cuda --execute-extraction 2>&1 | tee "$data/records/$stem-content.log"

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools audit \
  --manifest "$manifest.content.jsonl" \
  --catalog config/datasets.json \
  --require-features --require-content 2>&1 \
  | tee "$data/records/$stem-final.json"
