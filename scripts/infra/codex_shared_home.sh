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

resolve_local_codex_home() {
  if [[ -n "${HOME:-}" ]]; then
    printf '%s\n' "$HOME/.codex"
    return 0
  fi
  return 1
}

resolve_effective_codex_config() {
  if [[ -n "${CODEX_HOME:-}" && -f "${CODEX_HOME}/config.toml" ]]; then
    printf '%s\n' "${CODEX_HOME}/config.toml"
    return 0
  fi
  if [[ -f "$HOME/.codex/config.toml" ]]; then
    printf '%s\n' "$HOME/.codex/config.toml"
    return 0
  fi
  return 1
}

codex_profile_exists() {
  local profile="${1:-}"
  local config=""
  if [[ -z "$profile" ]]; then
    return 1
  fi
  config="$(resolve_effective_codex_config || true)"
  if [[ -z "$config" || ! -f "$config" ]]; then
    return 1
  fi
  grep -Fqx "[profiles.${profile}]" "$config"
}

append_codex_profile_arg() {
  local profile="${1:-}"
  local array_name="${2:-}"
  local config=""
  if [[ -z "$profile" || -z "$array_name" ]]; then
    return 0
  fi
  local -n codex_args_ref="$array_name"
  if codex_profile_exists "$profile"; then
    codex_args_ref+=(-p "$profile")
    return 0
  fi
  config="$(resolve_effective_codex_config || true)"
  if [[ -n "$config" ]]; then
    echo "WARN: Codex profile '$profile' not found in $config; launching without -p." >&2
  else
    echo "WARN: No Codex config with profile '$profile' found; launching without -p." >&2
  fi
}

setup_shared_codex_home() {
  if [[ -n "${CODEX_HOME:-}" ]]; then
    return 0
  fi
  if [[ -n "${CANOMPX3_SHARED_CODEX_HOME:-}" ]]; then
    local forced_candidate=""
    forced_candidate="$(resolve_shared_codex_home || true)"
    if [[ -n "$forced_candidate" && -d "$forced_candidate" ]]; then
      export CODEX_HOME="$forced_candidate"
    fi
    return 0
  fi
  local local_candidate=""
  local_candidate="$(resolve_local_codex_home || true)"
  if [[ -n "$local_candidate" && -d "$local_candidate" ]]; then
    return 0
  fi
  local candidate=""
  candidate="$(resolve_shared_codex_home || true)"
  if [[ -n "$candidate" && -d "$candidate" ]]; then
    export CODEX_HOME="$candidate"
  fi
}
