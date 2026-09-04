#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${RIFT_PYTHON:-${RIFT_V4_PYTHON:-python}}"

cd "$repo"
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$python" -m rift_v4.monitor "$@"
