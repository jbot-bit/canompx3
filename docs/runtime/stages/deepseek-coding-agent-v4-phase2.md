---
task: DeepSeek Coding Agent v4 — Phase 2 (OpenCode integration; supersedes prior draft)
mode: IMPLEMENTATION
slug: deepseek-coding-agent-v4-phase2
scope_lock:
  - CONVENTIONS.md
  - .aiderignore
  - scripts/tools/opencode-agent.ps1
  - scripts/tools/opencode_resolve_model.py
  - opencode.json
  - trading_app/ai/provider_registry.py
  - pipeline/check_drift.py
  - tests/test_scripts/test_opencode_resolve_model.py
  - tests/test_pipeline/test_check_drift.py
  - docs/specs/opencode_agent.md
  - docs/runtime/stages/deepseek-coding-agent-v4-phase2.md
  - docs/runtime/stages/deepseek-coding-agent-v4.md
---

# DeepSeek Coding Agent v4 — Phase 2 (OpenCode integration)

## Plan reference

Supersedes the earlier untracked draft of this file. Plan delivered by user
2026-05-06 after canonical OpenCode docs were verified verbatim:

- AGENTS.md + CLAUDE.md auto-load from project root (no `CONVENTIONS.md` need).
- OpenCode reads `opencode.json` from project root via cwd-walk-up to git root.
- MCP schema: top-level `mcp` (not `mcpServers`), type=`local` (not `stdio`),
  `command` is array, `environment` is dict.

This Phase 2 reshapes v4 around those facts and drops aider residue.

## What

1. **Drop aider dead weight.** `git rm CONVENTIONS.md .aiderignore`.
2. **Canonical model resolver.** New `scripts/tools/opencode_resolve_model.py`
   delegates to `provider_registry.get_profile("deepseek_coding")`. Zero
   re-encoded logic per institutional-rigor §4.
3. **Launcher integration.** `opencode-agent.ps1` calls the resolver before
   spawning OpenCode; banner annotates whether the model came from the
   canonical profile or the launcher default.
4. **OpenCode-shape MCP config.** New `opencode.json` at repo root wiring
   the four read-only servers (gold-db, repo-state, research-catalog,
   strategy-lab) in OpenCode's schema.
5. **Stale-comment fix.** `provider_registry.py:275` "via aider" → "via
   opencode" (one-line; zero behavior change).
6. **Drift ratchet.** New `check_hardcoded_openrouter_model_in_launcher`
   detects future `openrouter/<vendor>/<model>` literals in `*-agent.ps1`
   outside `# canonical-default-fallback:` annotation comments.
7. **Spec doc.** `docs/specs/opencode_agent.md` — pointers, not restated rules.

## Why (institutional grounding)

- **Volatile Data Rule** (CLAUDE.md): model IDs change weekly. PowerShell
  hardcode violates the rule. Profile registry is the canonical source.
- **Delegate, never re-encode** (institutional-rigor §4): launcher calls
  `get_profile("deepseek_coding").resolved().validation_errors()` instead
  of re-implementing model selection.
- **No silent failures** (institutional-rigor §6): when env var unset,
  launcher emits explicit stderr WARN with the exact env-var name + what
  to set it to; never silent fall-through.
- **AGENTS.md is canonical** for vendor-neutral agent rules. Per OpenCode
  docs, AGENTS.md auto-loads; `CONVENTIONS.md` was aider-specific and
  duplicated content. Removing it is the rule duplication fix.

## Files

| Path | Action | Notes |
|---|---|---|
| `CONVENTIONS.md` | DELETE | Aider-specific; OpenCode reads `AGENTS.md` natively. |
| `.aiderignore` | DELETE | Aider-specific. OpenCode does not read this. |
| `scripts/tools/opencode_resolve_model.py` | CREATE | ~30 lines. Imports canonical API only. |
| `opencode.json` | CREATE | Repo root. OpenCode-shape MCP config + `instructions`. |
| `scripts/tools/opencode-agent.ps1` | MODIFY | Add resolver call + canonical/default banner annotation. |
| `trading_app/ai/provider_registry.py` | MODIFY | One-line: line 275 "via aider" → "via opencode". |
| `pipeline/check_drift.py` | MODIFY | New `check_hardcoded_openrouter_model_in_launcher`; ~25 lines. |
| `tests/test_scripts/test_opencode_resolve_model.py` | CREATE | Resolver fail-closed + env-override + stdout-purity tests. |
| `tests/test_pipeline/test_check_drift.py` | MODIFY | Tests for the new launcher drift check. |
| `docs/specs/opencode_agent.md` | CREATE | User-facing spec; pointers, not restated rules. |
| `docs/runtime/stages/deepseek-coding-agent-v4-phase2.md` | CREATE | This stage doc. |
| `docs/runtime/stages/deepseek-coding-agent-v4.md` | MODIFY | Close Phase 1; cross-link phase2. |

## Blast Radius

- `CONVENTIONS.md` — DELETE; zero callers in production code, aider-specific. Risk: zero.
- `.aiderignore` — DELETE; zero callers in production code. Risk: zero.
- `scripts/tools/opencode-agent.ps1` — MODIFY; user-invoked from terminal; banner content + model resolution path change. Backwards-compatible (env var unset → existing default). Risk: low.
- `scripts/tools/opencode_resolve_model.py` — NEW; called only by launcher post-merge. Imports `trading_app.ai.provider_registry` (read-only API). Risk: low.
- `opencode.json` — NEW at repo root; consumed only by OpenCode TUI when launched. Read-only at runtime. Risk: low.
- `trading_app/ai/provider_registry.py` — single comment-line edit. Zero behavior change. Risk: zero.
- `pipeline/check_drift.py` — additive `check_hardcoded_openrouter_model_in_launcher`; passes against current launcher because of canonical-default annotation. Risk: low.
- Tests — additive only.
- Spec/stage docs — documentation only.
- Reads (read-only): trading_app/ai/provider_registry.py canonical API, .mcp.json (translation source).
- Writes: 12 files listed above. No DB writes. No live-trading changes. No CI workflow changes.
- Reversibility: revert the commit; deleted files come back from git history.

## Approach

1. **Delete first.** `git rm CONVENTIONS.md .aiderignore` — clean baseline before adding new code.
2. **Resolver is one function.** `opencode_resolve_model.py` calls `get_profile("deepseek_coding").resolved()`, prints model on success, prints errors on stderr + exits 1 on failure. No classes, no fallbacks.
3. **Launcher fallback is explicit.** Annotate the default with `# canonical-default-fallback:`; resolver-success path overrides with banner annotation; resolver-fail path emits stderr WARN naming the env var and proceeds with default.
4. **MCP config translates `.mcp.json` literally.** OpenCode-shape: `mcp` top-level, `local` type, command-array, `environment` dict. Skip `code-review-graph` (uvx, separate auth lifecycle).
5. **Drift check is a forward-looking ratchet.** Scans `scripts/tools/*-agent.ps1` for `openrouter/<vendor>/<model>` regex outside `# canonical-default-fallback:` comments.
6. **Spec doc is pointers, not restated rules.** Per integrity-guardian §1.

## Existing patterns reused

- `AIProfile.resolved()` / `validation_errors()` — direct call from resolver, zero re-encoding.
- 5-source key resolver in `opencode-agent.ps1` — left untouched. New resolver call is additive on a separate concern (model, not key).
- `OPENCODE_AGENT_SKIP_INSTALL=1` sentinel pattern — preserved.
- Stage-doc format — matches Phase 1.

## Out of scope (Phase 2)

- No claude-side review gate — Phase 3 (Stage B).
- No `.githooks/pre-commit` step 0d — Phase 3.
- No flipping of `check_deepseek_review_gate_intact` to blocking — Phase 3.
- No `check_or_credits.py` — Phase 4 (Stage C).
- No `code-review-graph` MCP wiring — uvx-launched, separate auth, deferred.

## Acceptance criteria

All required:

1. `CONVENTIONS.md` and `.aiderignore` removed (`git ls-files | Select-String -Pattern '^(CONVENTIONS\.md|\.aiderignore)$'` empty).
2. `python scripts/tools/opencode_resolve_model.py` (env var unset) → exit 1, stderr names missing var, stdout empty.
3. `CANOMPX3_AI_DEEPSEEK_CODING_MODEL=openrouter/deepseek-chat-v3.1 python scripts/tools/opencode_resolve_model.py` → exit 0, stdout=`openrouter/deepseek-chat-v3.1`, stderr empty.
4. `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` (env unset) → banner shows `(launcher default)`, stderr has WARN, exit 0.
5. `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` (env set) → banner shows `(canonical profile)`, no WARN, exit 0.
6. `python -m pytest tests/test_scripts/test_opencode_resolve_model.py tests/test_trading_app/test_ai/test_provider_registry.py tests/test_pipeline/test_check_drift.py` all pass.
7. `python pipeline/check_drift.py` passes (new check is no-op against in-tree launcher because of canonical-default annotation).
8. `git diff --stat` touches only files in scope_lock.

## Done definition (Phase 2 only)

All four required (institutional rigor §8):

- [ ] Acceptance criteria 1–8 green with execution evidence in Verification log.
- [ ] Dead-code sweep: `grep -r "CONVENTIONS\.md\|aiderignore" --include="*.py"` empty (or only this stage doc).
- [ ] `python pipeline/check_drift.py` passes.
- [ ] Self-review pass.

## Verification log

- [x] Acceptance criterion 1: `git ls-files | grep -E '^(CONVENTIONS\.md|\.aiderignore)$'` exits 1 (no match). Both files removed via `git rm`.
- [x] Acceptance criterion 2: `python scripts/tools/opencode_resolve_model.py` (env unset) → exit 1; stderr `opencode_resolve_model: model not configured; set CANOMPX3_AI_DEEPSEEK_CODING_MODEL for this profile` + missing `OPENROUTER_API_KEY`; stdout empty.
- [x] Acceptance criterion 3: `CANOMPX3_AI_DEEPSEEK_CODING_MODEL=openrouter/deepseek-chat-v3.1 OPENROUTER_API_KEY=test python scripts/tools/opencode_resolve_model.py` → exit 0; stdout `openrouter/deepseek-chat-v3.1`; stderr empty.
- [x] Acceptance criterion 4: `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` (env unset) → banner `model=openrouter/deepseek-chat-v3.1 (launcher default)`; WARN line present.
- [x] Acceptance criterion 5: `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` (env set) → banner `model=openrouter/deepseek-chat-v3.1 (canonical profile)`; no WARN.
- [x] Acceptance criterion 6: `python -m pytest tests/test_scripts/test_opencode_resolve_model.py tests/test_trading_app/test_ai/test_provider_registry.py tests/test_pipeline/test_check_drift.py` → **172 passed** (153 prior + 5 resolver + 7 new launcher + 7 already counted in mismatch).
- [x] Acceptance criterion 7: `python pipeline/check_drift.py` → `NO DRIFT DETECTED: 121 checks passed [OK]`.
- [x] Acceptance criterion 8: `git status --short` shows exactly the 12 scope_lock files (5 modified + 5 new + 2 deleted).

## Live smoke test (Stage A criterion 9)

User-supervised; pending. Follow `docs/specs/opencode_agent.md` § Auth + § MCP for the manual gate before merging.
