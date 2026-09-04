#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="${RIFT_DATA_ROOT:?set RIFT_DATA_ROOT to the training data directory}"
source="${GTSINGER_SOURCE:-$data/gtsinger-git}"
python="${RIFT_PYTHON:-python}"

cd "$repo"
mkdir -p "$data/sources"

PYTHONPATH="$repo" "$python" -m rift_v4.datasets gtsinger \
  --source "$source" \
  --output "$data/sources/GTSinger"

exec "$repo/scripts/prepare_corpus.sh" \
  GTSinger "$data/sources/GTSinger" gtsinger
