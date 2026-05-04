#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec env CANOMPX3_CODEX_ENABLE_GOLD_DB=1 "$ROOT/scripts/infra/codex-project.sh" "$@"
