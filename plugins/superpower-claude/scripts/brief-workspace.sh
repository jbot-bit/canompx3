#!/usr/bin/env bash
set -euo pipefail

plugin_root="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
repo_root="$(cd "${plugin_root}/../.." && pwd)"
mode="${1:-interactive}"

if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/python" ]]; then
  python_bin="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/Scripts/python.exe" ]]; then
  python_bin="${VIRTUAL_ENV}/Scripts/python.exe"
elif [[ -f "${repo_root}/.venv-wsl/bin/python" ]]; then
  python_bin="${repo_root}/.venv-wsl/bin/python"
elif [[ -f "${repo_root}/.venv/Scripts/python.exe" ]]; then
  python_bin="${repo_root}/.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "python runtime not found" >&2
  exit 127
fi

cd "${repo_root}"
"${python_bin}" "scripts/tools/claude_superpower_brief.py" --root "${repo_root}" --mode "${mode}"
