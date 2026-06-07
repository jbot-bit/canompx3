#!/usr/bin/env bash
# run_mutation.sh — portable, scoped mutation testing for the capital-critical core.
#
# WHY cosmic-ray, not mutmut: mutmut 3.x hard-refuses native Windows on EVERY
# subcommand ("To run mutmut on Windows, please use the WSL" — boxed/mutmut#397),
# which would pin mutation testing to a WSL/.venv-wsl/Python-3.12 island that no
# clean checkout or GitHub-Actions runner could reproduce. cosmic-ray runs
# natively on Windows + Linux + CI on this repo's Python 3.11.9 and mutates
# in-place (no `mutants/` source-copy tree → no collision with check_drift.py's
# rglob("*.py") scans).
#
# PORTABILITY: every path here is repo-relative; the only machine assumption is a
# working `python -m pytest` (resolved via PATH / the active venv). No absolute
# paths, no WSL, no /mnt/c. A fresh clone runs this unchanged.
#
# Usage (invoke via `uv run bash` so the venv's cosmic-ray/python are on PATH):
#   uv run bash scripts/tools/run_mutation.sh <module_src> "<test_command>" [timeout_s]
# Example:
#   uv run bash scripts/tools/run_mutation.sh pipeline/cost_model.py \
#     "python -m pytest tests/test_pipeline/test_cost_model.py -q" 30
#
# (cosmic-ray cannot read a process-substitution `<(...)` config on Windows —
# this script writes a REAL temp .toml under .mutation/, which works everywhere.)
#
# Output: a survival-rate report + the session sqlite under .mutation/ (gitignored).
set -euo pipefail

MODULE_SRC="${1:?usage: run_mutation.sh <module_src> <test_command> [timeout_s]}"
TEST_CMD="${2:?need a test command, e.g. 'python -m pytest tests/...'}"
TIMEOUT="${3:-30}"

if [[ ! -f "$MODULE_SRC" ]]; then
  echo "ERROR: module source not found: $MODULE_SRC" >&2
  exit 1
fi

# Ephemeral session dir — gitignored, removed on exit so it never reaches a peer's
# working tree or a commit (multi-terminal-shared-file-hygiene).
SESS_DIR=".mutation"
mkdir -p "$SESS_DIR"
slug="$(echo "$MODULE_SRC" | tr '/.' '__')"
CFG="$SESS_DIR/${slug}.toml"
SESSION="$SESS_DIR/${slug}.sqlite"

cleanup() { rm -f "$CFG" "$SESSION" 2>/dev/null || true; }
trap cleanup EXIT

# 1. Write the config deterministically (avoids cosmic-ray's interactive new-config).
cat > "$CFG" <<EOF
[cosmic-ray]
module-path = "$MODULE_SRC"
timeout = ${TIMEOUT}.0
excluded-modules = []
test-command = "$TEST_CMD"

[cosmic-ray.distributor]
name = "local"
EOF

echo "=== baseline (unmutated tests must pass) ==="
cosmic-ray --verbosity INFO baseline "$CFG"

echo "=== init session (plan mutations) ==="
cosmic-ray init "$CFG" "$SESSION" --force

echo "=== exec (run every mutant) ==="
cosmic-ray exec "$CFG" "$SESSION"

echo "=== survival rate ==="
cr-rate --estimate --confidence 95.0 "$SESSION"

echo "=== survivors (killed mutants omitted) ==="
cr-report --no-show-output "$SESSION"
