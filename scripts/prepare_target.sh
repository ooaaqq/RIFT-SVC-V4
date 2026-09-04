#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 SOURCE_DIR" >&2
  exit 2
fi

source_dir="$(realpath "$1")"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${RIFT_DATA_ROOT:?set RIFT_DATA_ROOT to the target-singer data directory}"
contentvec="${CONTENTVEC_MODEL:?set CONTENTVEC_MODEL to the pinned local snapshot}"
python="${RIFT_PYTHON:-python}"
speaker="${RIFT_TARGET_SPEAKER:-target}"
validation_file="${RIFT_VALIDATION_SONGS_FILE:-}"
staged="$data/sources/Target/$speaker"
manifest="$data/manifests/training"

mkdir -p "$staged" "$data/manifests" "$data/records"
shopt -s nullglob
audio_files=("$source_dir"/*.wav "$source_dir"/*.flac "$source_dir"/*.ogg)
if ((${#audio_files[@]} < 5)); then
  echo "target preparation requires at least five top-level recordings" >&2
  exit 1
fi
for audio in "${audio_files[@]}"; do
  name="$(basename "$audio")"
  song="${name%.*}"
  destination="$staged/$song/$name"
  mkdir -p "$(dirname "$destination")"
  if [[ -L "$destination" ]]; then
    [[ "$(realpath "$destination")" == "$(realpath "$audio")" ]] || {
      echo "staging link points to a different source: $destination" >&2
      exit 1
    }
  elif [[ -e "$destination" ]]; then
    echo "staging path is not a symlink: $destination" >&2
    exit 1
  else
    ln -s "$audio" "$destination"
  fi
done

cd "$repo"
PYTHONPATH="$repo" "$python" -m rift_v4.manifest_cli \
  --config config/target-finetune.json \
  --source "Target=$data/sources/Target" \
  --features-root "$data/features" \
  --output "$manifest.pending.jsonl"

PYTHONPATH="$repo" "$python" -m rift_v4.qc \
  --config config/target-finetune.json \
  --manifest "$manifest.pending.jsonl" \
  --output "$manifest.qc.jsonl"

validation_args=()
if [[ -n "$validation_file" ]]; then
  while IFS= read -r song; do
    [[ -z "$song" || "$song" == \#* ]] || validation_args+=(--validation-song "$song")
  done < "$validation_file"
  ((${#validation_args[@]} > 0)) || {
    echo "validation song file contains no songs: $validation_file" >&2
    exit 1
  }
fi
PYTHONPATH="$repo" "$python" -m rift_v4.split_cli \
  --manifest "$manifest.qc.jsonl" \
  --output "$manifest.split.jsonl" \
  --validation-ratio 0.20 --test-ratio 0 --seed 2026 \
  "${validation_args[@]}"

PYTHONPATH="$repo" "$python" -m rift_v4.features extract \
  --config config/target-finetune.json \
  --manifest "$manifest.split.jsonl" \
  --device cuda --execute-extraction 2>&1 | tee "$data/records/target-features.log"

PYTHONPATH="$repo" "$python" -m rift_v4.extract_content \
  --config config/target-finetune.json \
  --manifest "$manifest.split.jsonl" \
  --base-model-path "$contentvec" \
  --contentvec-lock third_party/contentvec.lock.json \
  --features-root "$data/raw-content" \
  --output-manifest "$manifest.content.jsonl" \
  --device cuda --execute-extraction 2>&1 | tee "$data/records/target-content.log"

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools reconcile-frames \
  --manifest "$manifest.content.jsonl" \
  --output "$data/manifests/training.content.jsonl" \
  --mel-channels 128 --require-content

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools audit \
  --config config/target-finetune.json \
  --manifest "$data/manifests/training.content.jsonl" \
  --require-features --require-content --verify-audio --workers 8 2>&1 \
  | tee "$data/records/target-final.json"

PYTHONPATH="$repo" "$python" -m rift_v4.sampling_cli \
  --config config/target-finetune.json \
  --manifest "$data/manifests/training.content.jsonl" \
  --output "$data/records/sampling-audit.json"

echo "target preparation complete: $data/manifests/training.content.jsonl"
