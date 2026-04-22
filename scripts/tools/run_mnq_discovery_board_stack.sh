#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PRIMARY_ROOT="$(git -C "$PROJECT_ROOT" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"

ensure_shared_venv_link() {
  if [[ -e "$PROJECT_ROOT/.venv-wsl" ]]; then
    return
  fi
  if [[ -n "${PRIMARY_ROOT:-}" && -d "$PRIMARY_ROOT/.venv-wsl" ]]; then
    ln -s "$PRIMARY_ROOT/.venv-wsl" "$PROJECT_ROOT/.venv-wsl"
  fi
}

resolve_python() {
  ensure_shared_venv_link
  if [[ -x "$PROJECT_ROOT/.venv-wsl/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv-wsl/bin/python"
    return
  fi

  if [[ -n "${PRIMARY_ROOT:-}" && -x "$PRIMARY_ROOT/.venv-wsl/bin/python" ]]; then
    printf '%s\n' "$PRIMARY_ROOT/.venv-wsl/bin/python"
    return
  fi

  command -v python3 2>/dev/null || command -v python 2>/dev/null || true
}

PY="$(resolve_python)"

if [[ -z "${PY:-}" ]]; then
  echo "ERROR: python interpreter not found" >&2
  exit 2
fi

steps=(
  "research/mnq_unfiltered_baseline_cross_family_v1.py"
  "research/mnq_live_context_overlays_v1.py"
  "research/mnq_layered_candidate_board_v1.py"
  "research/mnq_prior_day_family_board_v1.py"
  "research/mnq_geometry_transfer_board_v1.py"
)

echo "=== MNQ DISCOVERY BOARD STACK ==="
echo "Project root: $PROJECT_ROOT"
echo "Python:       $PY"

for step in "${steps[@]}"; do
  echo ""
  echo "--- Running $step ---"
  "$PY" "$PROJECT_ROOT/$step"
done

echo ""
echo "Board outputs refreshed:"
echo "- docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md"
echo "- docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md"
echo "- docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md"
echo "- docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md"
echo "- docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md"
