#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${RIFT_DATA_ROOT:?set RIFT_DATA_ROOT to the training data directory}"
python="${RIFT_PYTHON:-${RIFT_V4_PYTHON:-python}}"
manifest="${RIFT_MANIFEST:-$data/manifests/training.content.jsonl}"
mel_stats="${RIFT_MEL_STATS:-$data/manifests/mel-stats.json}"
run_dir="${RIFT_RUN_DIR:-$data/runs/rift}"
rift_log="${RIFT_TRAIN_LOG:-$data/records/rift.log}"
mkdir -p "$data/records" "$run_dir"

usage() {
  echo "usage: $0 smoke | perf | rift [STOP_AT_STEP] | resume CHECKPOINT [STOP_AT_STEP] | decay CHECKPOINT [STOP_AT_STEP] | finetune FOUNDATION_CHECKPOINT [STOP_AT_STEP] | panel CHECKPOINT" >&2
  exit 2
}

command="${1:-}"
cd "$repo"
case "$command" in
  smoke)
    PYTHONPATH="$repo" exec "$python" scripts/gpu_smoke_test.py \
      --config config/v4.json --frames 12288 16384 --warmup 2 --steps 3 \
      --model-mode compile --float8-mode on
    ;;
  perf)
    PYTHONPATH="$repo" "$python" scripts/gpu_smoke_test.py \
      --config config/v4.json --canonical-buckets --warmup 5 --steps 50 \
      --model-mode compile --compile-mode default --float8-mode off
    PYTHONPATH="$repo" "$python" scripts/gpu_smoke_test.py \
      --config config/v4.json --canonical-buckets --warmup 5 --steps 50 \
      --model-mode compile --compile-mode default --float8-mode on \
      --float8-recipe rowwise_with_gw_hp
    PYTHONPATH="$repo" exec "$python" scripts/gpu_smoke_test.py \
      --config config/v4.json --canonical-buckets --warmup 5 --steps 50 \
      --model-mode compile --compile-mode default --float8-mode on \
      --float8-recipe rowwise
    ;;
  rift)
    if find "$run_dir" -maxdepth 1 -type f -name '*.pt' -print -quit \
      2>/dev/null | grep -q .; then
      echo "run directory already contains checkpoints: $run_dir" >&2
      echo "use the explicit resume command or choose an empty run directory" >&2
      exit 1
    fi
    stop_at="${2:-${RIFT_STOP_AT_STEP:-}}"
    stop_args=()
    [[ -z "$stop_at" ]] || stop_args+=(--stop-at-step "$stop_at")
    PYTHONUNBUFFERED=1 PYTHONPATH="$repo" "$python" -m rift_v4.train \
      --config config/v4.json \
      --manifest "$manifest" \
      --mel-stats "$mel_stats" \
      --output "$run_dir" \
      --device cuda \
      --num-workers 8 \
      "${stop_args[@]}" \
      --execute-training 2>&1 | tee "$rift_log"
    ;;
  resume)
    checkpoint="${2:-}"
    [[ -n "$checkpoint" ]] || usage
    stop_at="${3:-${RIFT_STOP_AT_STEP:-}}"
    stop_args=()
    [[ -z "$stop_at" ]] || stop_args+=(--stop-at-step "$stop_at")
    PYTHONUNBUFFERED=1 PYTHONPATH="$repo" "$python" -m rift_v4.train \
      --config config/v4.json \
      --manifest "$manifest" \
      --mel-stats "$mel_stats" \
      --output "$run_dir" \
      --resume "$checkpoint" \
      --device cuda \
      --num-workers 8 \
      "${stop_args[@]}" \
      --execute-training 2>&1 | tee -a "$rift_log"
    ;;
  decay)
    checkpoint="${2:-}"
    [[ -n "$checkpoint" ]] || usage
    stop_at="${3:-${RIFT_STOP_AT_STEP:-}}"
    stop_args=()
    [[ -z "$stop_at" ]] || stop_args+=(--stop-at-step "$stop_at")
    PYTHONUNBUFFERED=1 PYTHONPATH="$repo" "$python" -m rift_v4.train \
      --config config/v4.json \
      --manifest "$manifest" \
      --mel-stats "$mel_stats" \
      --output "$run_dir" \
      --resume "$checkpoint" \
      --resume-lr-scale 0.5 \
      --device cuda \
      --num-workers 8 \
      "${stop_args[@]}" \
      --execute-training 2>&1 | tee -a "$rift_log"
    ;;
  finetune)
    checkpoint="${2:-}"
    [[ -n "$checkpoint" ]] || usage
    stop_at="${3:-${RIFT_STOP_AT_STEP:-}}"
    stop_args=()
    [[ -z "$stop_at" ]] || stop_args+=(--stop-at-step "$stop_at")
    finetune_config="${RIFT_FINETUNE_CONFIG:-config/target-finetune.json}"
    PYTHONUNBUFFERED=1 PYTHONPATH="$repo" "$python" -m rift_v4.train \
      --config "$finetune_config" \
      --manifest "$manifest" \
      --mel-stats "$mel_stats" \
      --output "$run_dir" \
      --initialize-from "$checkpoint" \
      --device cuda \
      --num-workers 8 \
      "${stop_args[@]}" \
      --execute-training 2>&1 | tee "$rift_log"
    ;;
  panel)
    checkpoint="${2:-}"
    [[ -n "$checkpoint" ]] || usage
    pc_nsf_checkout="${RIFT_PC_NSF_CHECKOUT:?set RIFT_PC_NSF_CHECKOUT}"
    pc_nsf_checkpoint="${RIFT_PC_NSF_CHECKPOINT:?set RIFT_PC_NSF_CHECKPOINT}"
    PYTHONUNBUFFERED=1 PYTHONPATH="$repo" exec "$python" -m rift_v4.evaluate \
      --config config/v4.json \
      --manifest "$manifest" \
      --checkpoint "$checkpoint" \
      --panel-spec "$data/records/audio-panel.json" \
      --output "$run_dir/audio-panel" \
      --pc-nsf-checkout "$pc_nsf_checkout" \
      --pc-nsf-lock third_party/pc_nsf_hifigan.lock.json \
      --vocoder-checkpoint "$pc_nsf_checkpoint" \
      --device cuda
    ;;
  *) usage ;;
esac
