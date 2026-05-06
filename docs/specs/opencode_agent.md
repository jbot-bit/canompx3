# OpenCode Coding Agent — Spec

The DeepSeek Coding Agent v4 backup runtime. When Claude tokens run out,
launch OpenCode against OpenRouter/DeepSeek with the same canonical
boundaries: read-only MCP servers, fail-closed model resolution, and
(Phase 3) Claude-side review gating before commits land.

This document is **pointers, not restated rules** (per integrity-guardian
§1). Canonical authorities live elsewhere — this spec only links to them.

## Overview

- Launcher: `scripts/tools/opencode-agent.ps1` (PowerShell; npm-global resolves `opencode` automatically).
- Model resolver: `scripts/tools/opencode_resolve_model.py` — single source of truth via `trading_app.ai.provider_registry.get_profile("deepseek_coding")`.
- MCP wiring: `opencode.json` at repo root (auto-discovered via cwd-walk-up per https://opencode.ai/docs/config/).
- Doctrine: `AGENTS.md` + `CLAUDE.md` auto-load from project root per https://opencode.ai/docs/rules/.
- Profile contract: `trading_app.ai.provider_registry.PROFILE_REGISTRY["deepseek_coding"]` (`runtime_class="interactive_editor"`, `mutation_allowed=True`, ZDR router).

## Auth

OpenCode reads `OPENROUTER_API_KEY` from the environment. The launcher
resolves the key from a 5-source chain (env → `~/.canompx-ask/.env` →
`<repo>/.env` → `~/.canompx-ask/config.toml`) and exports it before
spawning OpenCode.

If OpenCode prompts for auth interactively despite the env var, run once
per machine:
```
opencode auth login --provider openrouter
```
This persists the key in OpenCode's auth store. Verify with:
```
opencode auth list
```

## Model Resolution

Source of truth: the `deepseek_coding` profile in `provider_registry.py`.
Set `CANOMPX3_AI_DEEPSEEK_CODING_MODEL=openrouter/<vendor>/<model>` to
configure. The resolver:

- Returns the model on stdout (exit 0) when the env var is set + the
  router config validates.
- Exits 1 with stderr diagnostics otherwise.

The launcher calls the resolver before spawning OpenCode:
- success → uses resolver's stdout, banner shows `(canonical profile)`.
- failure → emits stderr WARN, falls back to the launcher's
  annotated default, banner shows `(launcher default)`.

The launcher's default is annotated with `# canonical-default-fallback:`
in `param()`. Drift check `check_hardcoded_openrouter_model_in_launcher`
fails if any second hardcoded `openrouter/<vendor>/<model>` literal is
introduced without that annotation.

## MCP

`opencode.json` wires four read-only servers:
- `gold-db` — `trading_app/mcp_server.py` (canonical strategy/fitness/outcomes lookups).
- `repo-state` — `scripts/tools/repo_state_mcp_server.py` (task routing, system context).
- `research-catalog` — `scripts/tools/research_catalog_mcp_server.py` (literature, hypotheses, audit results).
- `strategy-lab` — `scripts/tools/strategy_lab_mcp_server.py` (validation, fitness, lane allocation).

Source of truth for what each server exposes is `.mcp.json` (read-only mirror).
`code-review-graph` is intentionally excluded — it ships via `uvx` with a
separate auth lifecycle and will be added in a follow-up.

## Review Gate

`scripts/tools/claude_review_deepseek.py` reads `git diff --cached`,
asks Claude (seven-sins rubric inlined as a hermetic constant for
`CLAUDE_REASONING_MODEL`), parses a strict JSON verdict, and exits:

- `0` — APPROVE, or no diff to review (empty / doc-only / < 5-line).
- `1` — BLOCK; findings printed to stderr; commit aborted.
- `2` — REVIEW_UNAVAILABLE (network or parse error); never silent. User
  may `--no-verify` once with explicit acknowledgement; the drift check
  `check_deepseek_review_gate_intact` still fires if the marker is
  present but the canonical invocation is missing.

The launcher exports `OPENCODE_AGENT_ACTIVE=1` before spawning OpenCode.
Pre-commit step `# 0d.` only fires when that env var is set, so normal
Claude-side / manual commits skip the gate.

Test surface: `tests/test_scripts/test_claude_review_deepseek.py` covers
mock APPROVE / BLOCK / inactive-env / empty-diff / doc-only / threshold
helpers. Live Claude calls are not exercised in CI.

## Credits (Phase 4 — not yet wired)

Phase 4 lands `scripts/tools/check_or_credits.py`. Opt-in via
`OPENCODE_AGENT_CHECK_CREDITS=1`. The endpoint URL must be falsified
against the live OpenRouter API before the script ships (Phase 4 step 1).

## Fallback Path

If OpenCode is ever discontinued or its `--model openrouter/...` interface
changes, the long-term backstop is to extend `scripts/tools/ask.py` (which
already has OpenRouter plumbing via `trading_app.ai.openrouter_runtime`).
The trade-off was recorded during planning: OpenCode wins on UX, MCP, and
auto-loaded doctrine; `ask.py` wins on full local control. Build the
python-native path only if OpenCode fails.

For one-shot review questions when a full TUI isn't warranted, use
`python scripts/tools/ask.py --code <prompt>` directly.

## Known Issues

- OpenCode v1.14.39 is the verified version. CLI flags may shift on
  upgrades; re-verify Stage A criterion 9 (live smoke) on bumps.
- The local-MCP `command` array uses `.venv/Scripts/python.exe` (Windows
  venv layout). Linux/macOS users need to adjust to `.venv/bin/python`
  or absolutize the path. Falsify on first launch via `/mcp` in the TUI.
- The `instructions` array lists `AGENTS.md` and `CLAUDE.md` for
  self-documentation even though OpenCode auto-loads them from the
  project root. The third entry, `docs/governance/document_authority.md`,
  is a non-root file and is the only one that strictly needs the
  `instructions` field.

## Maintenance

- Bumping the OpenCode CLI version: re-verify the `opencode --model`
  flag still accepts `openrouter/<vendor>/<model>` format. The `--model`
  flag is documented at https://opencode.ai/docs/cli/.
- Adding a new MCP server: edit `opencode.json`. Match the OpenCode
  schema (top-level `mcp`, type=`local`, command array). Verify in the
  TUI via `/mcp`.
- Changing the canonical model: set
  `CANOMPX3_AI_DEEPSEEK_CODING_MODEL=<new>` in your shell profile or
  `~/.canompx-ask/.env`. The launcher picks it up automatically; no
  code change needed.
- Updating the launcher default: edit the `param()` block AND keep the
  `# canonical-default-fallback:` annotation. The drift check enforces
  the contract.
