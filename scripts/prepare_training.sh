#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${RIFT_DATA_ROOT:?set RIFT_DATA_ROOT to the training data directory}"
python="${RIFT_PYTHON:-python}"
manifest_dir="$data/manifests"
output="$manifest_dir/training.content.jsonl"
audit_workers="${RIFT_AUDIT_WORKERS:-8}"
mkdir -p "$manifest_dir" "$data/records"

if ! [[ "$audit_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "RIFT_AUDIT_WORKERS must be a positive integer" >&2
  exit 2
fi

cd "$repo"
PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools merge \
  --catalog config/datasets.json \
  --manifest-dir "$manifest_dir" \
  --output "$output"

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools reconcile-frames \
  --manifest "$output" \
  --output "$output" \
  --mel-channels 128 \
  --require-content

PYTHONPATH="$repo" "$python" -m rift_v4.manifest_tools audit \
  --config config/v4.json \
  --catalog config/datasets.json \
  --manifest "$output" \
  --require-features \
  --require-content \
  --workers "$audit_workers" \
  --verify-audio 2>&1 | tee "$data/records/training-audit.json"

PYTHONPATH="$repo" "$python" -m rift_v4.sampling_cli \
  --config config/v4.json \
  --manifest "$output" \
  --output "$data/records/sampling-audit.json"

stats_shards="${RIFT_STATS_SHARDS:-4}"
if ! [[ "$stats_shards" =~ ^[1-9][0-9]*$ ]]; then
  echo "RIFT_STATS_SHARDS must be a positive integer" >&2
  exit 2
fi
stats_pids=()
for shard in $(seq 0 $((stats_shards - 1))); do
  PYTHONPATH="$repo" "$python" -m rift_v4.features stats \
    --config config/v4.json \
    --manifest "$output" \
    --output "$manifest_dir/mel-stats.part.$shard.json" \
    --num-shards "$stats_shards" \
    --shard-index "$shard" \
    >"$data/records/mel-stats.part.$shard.log" 2>&1 &
  stats_pids+=("$!")
done
stats_failed=0
for pid in "${stats_pids[@]}"; do
  if ! wait "$pid"; then
    stats_failed=1
  fi
done
((stats_failed == 0)) || {
  echo "one or more mel stats shards failed; inspect $data/records/mel-stats.part.*.log" >&2
  exit 1
}
inputs=()
for shard in $(seq 0 $((stats_shards - 1))); do
  part="$manifest_dir/mel-stats.part.$shard.json"
  [[ -s "$part" ]] || { echo "missing mel stats shard: $part" >&2; exit 1; }
  inputs+=(--input "$part")
done
PYTHONPATH="$repo" "$python" -m rift_v4.features merge-stats \
  "${inputs[@]}" \
  --output "$manifest_dir/mel-stats.json" \
  2>&1 | tee "$data/records/mel-stats.log"

echo "training preparation complete: $output" | tee "$data/records/training-complete.log"
