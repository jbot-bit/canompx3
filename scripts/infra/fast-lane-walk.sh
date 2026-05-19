#!/usr/bin/env bash
# fast-lane-walk.sh — front-door wrapper for scripts/tools/fast_lane_walk.py
#
# Stage 4 of docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md.
# Composes the existing fast-lane writers + renders the awareness Markdown
# report. Read-only over capital-class state.
#
# Venv discovery: same pattern as scripts/infra/prereg-loop.sh
#   1. $CANOMPX3_PYTHON override
#   2. <repo>/.venv-wsl/bin/python
#   3. python3 on PATH
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-wsl"
PYTHON="${CANOMPX3_PYTHON:-}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
  else
    PYTHON="python3"
  fi
fi

cd "$ROOT"
exec "$PYTHON" "$ROOT/scripts/tools/fast_lane_walk.py" "$@"
