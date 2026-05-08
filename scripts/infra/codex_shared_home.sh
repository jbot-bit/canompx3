#!/usr/bin/env bash

resolve_shared_codex_home() {
  if [[ -n "${CANOMPX3_SHARED_CODEX_HOME:-}" ]]; then
    printf '%s\n' "$CANOMPX3_SHARED_CODEX_HOME"
    return 0
  fi
  if [[ -n "${USER:-}" ]]; then
    printf '%s\n' "/mnt/c/Users/$USER/.codex"
    return 0
  fi
  return 1
}

setup_shared_codex_home() {
  if [[ -n "${CODEX_HOME:-}" ]]; then
    return 0
  fi
  local candidate=""
  candidate="$(resolve_shared_codex_home || true)"
  if [[ -n "$candidate" && -d "$candidate" ]]; then
    export CODEX_HOME="$candidate"
  fi
}
