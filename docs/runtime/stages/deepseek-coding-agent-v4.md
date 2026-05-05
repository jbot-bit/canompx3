---
task: DeepSeek Coding Agent v4 — Phase 1 (profile + grounding files + drift-check --quiet)
mode: IMPLEMENTATION
slug: deepseek-coding-agent-v4
scope_lock:
  - trading_app/ai/provider_registry.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_ai/test_provider_registry.py
  - tests/test_pipeline/test_check_drift.py
  - CONVENTIONS.md
  - .aiderignore
---

# DeepSeek Coding Agent v4 — Stage File

## Plan reference

Full v4 plan delivered in chat by user 2026-05-06. 5 phases (0 + 1 + 2 + 2.5 + 3 + 4 + 5).
Codex foundation prerequisite (Phase 0) is **SATISFIED** — `fc82afe0 feat(ai): add bounded openrouter research runtime (#220)` is on `origin/main` with all four foundation files (`provider_registry.py`, `openrouter_runtime.py`, `research_packet.py`, `schema_registry.py`).

This stage file scopes **Phase 1 only**. Subsequent phases get their own stage files.

## What (Phase 1)

Land the four pieces of foundation that every later phase depends on, and that change zero behavior on the existing system:

1. New `deepseek_coding` profile in `trading_app/ai/provider_registry.AIPROFILES`. `model=None` until Phase 2.5 bake-off picks the winner. `assert_ready` fails-closed by design until then.
2. New `runtime_class` literal `"interactive_editor"` (third value alongside `"read_only_single_turn"` and `"read_only_tool_loop"`).
3. **Required refactor (NEW finding, not in v4 plan):** `AIProfile.validation_errors()` currently has blanket "mutation authority is not allowed for repo research profiles" and "live-control authority is not allowed" checks. These are correct for research profiles but reject the editing profile by definition. Refactor: gate those two checks on `runtime_class != "interactive_editor"`. Zero behavior change for the 3 existing OpenRouter profiles (all have `mutation_allowed=False` default and `runtime_class != "interactive_editor"` so they hit the same rejection path as before).
4. `pipeline/check_drift.py --quiet` flag — sanitized output for LLM consumption (Source 21 mitigation #16). Prints only `PASS` or `FAIL: <check_name> (count=N)` per check, no diagnostic detail.
5. New drift check `check_deepseek_review_gate_intact` — registered, but functions as a **declarative no-op** until Phase 3 lands the actual `pre-commit` step 0d. The check exists so we can't merge Phase 1 without remembering to keep the registry slot. Implementation: returns `[]` until step 0d marker present in `.githooks/pre-commit`, then asserts presence.
6. `CONVENTIONS.md` — aider-canonical project doctrine (Source 19). ~80 lines. Pointers only (delegate to canonical sources, banned patterns, seven sins summary).
7. `.aiderignore` — gitignore-with-negation allowlist. Excludes data/venv/worktrees, allows pythonsrc + doctrine.

## Why (institutional grounding)

- **Auto-mode + critical risk tier:** lean exploration, execution evidence before done. Phase 1 is the smallest unit that ships nothing usable yet (model=None means launch fails closed). Safe to land in one PR.
- **Stage-gate-protocol:** every later phase locks-and-loads on this profile registry; landing Phase 1 in isolation gives every later stage a known truth-floor.
- **Institutional rigor rule #4 (delegate, don't re-encode):** v4 plan deliberately reuses `provider_registry.py` rather than spinning up a new "editor profile registry". Phase 1 is the load-bearing alignment with that rule.
- **Plan v4 caveat ("model field left as None until Phase 2.5; assert_ready fails-closed by design"):** by-design fail-closed prevents anyone — including a future me — from launching the agent before the bake-off picks the model.

## Files

| Path | Action | Notes |
|---|---|---|
| `trading_app/ai/provider_registry.py` | MODIFY | +~50 lines. Add `"interactive_editor"` to `RuntimeClass` literal. Add `deepseek_coding` profile (model=None, mutation_allowed=True, runtime_class="interactive_editor", router with allow_fallbacks=False + zdr=True). Refactor `validation_errors()` to gate the mutation/live-control checks on `runtime_class != "interactive_editor"`. |
| `pipeline/check_drift.py` | MODIFY | +~50 lines. Add `--quiet` flag to argparse. Switch every print path to a `_emit(quiet, level, msg)` helper that emits sanitized PASS/FAIL only when `--quiet`. Add `check_deepseek_review_gate_intact` (no-op until Phase 3). Register via the existing CHECKS-list pattern. |
| `tests/test_trading_app/test_ai/test_provider_registry.py` | MODIFY | +~60 lines. Tests for: (a) `deepseek_coding` profile is in `AIPROFILES`, (b) `runtime_class == "interactive_editor"`, (c) `mutation_allowed is True`, (d) `assert_ready()` fails when model is None (fail-closed by design), (e) `assert_ready()` succeeds when `CANOMPX3_AI_DEEPSEEK_CODING_MODEL` env var is set, (f) `assert_openrouter_research_profile` REJECTS the coding profile (it is not a research profile), (g) router has `zdr=True` and `allow_fallbacks=False`, (h) existing 3 OpenRouter research profiles still reject `mutation_allowed=True` if hand-set (regression guard for the validation_errors refactor). |
| `tests/test_pipeline/test_check_drift.py` | MODIFY | +~30 lines. Tests for: (a) `--quiet` mode prints only `PASS` / `FAIL: <name> (count=N)` lines (no diagnostic detail leaks), (b) `check_deepseek_review_gate_intact` returns `[]` (no-op) when step 0d marker absent — Phase 3 will flip this. |
| `CONVENTIONS.md` | CREATE | ~80 lines. Aider-canonical project-doctrine slot (Source 19). Pointers, not content (delegate). |
| `.aiderignore` | CREATE | ~40 lines. Negation allowlist. Excludes `gold.db`, `data/`, `.venv/`, `.worktrees/`, `*.parquet`, `*.csv`, aider history files. Negate-includes `pipeline/**/*.py`, `trading_app/**/*.py`, `scripts/tools/**/*.py`, root `*.md`, `docs/governance/**/*.md`, `docs/specs/**/*.md`, `docs/institutional/**/*.md`. |

## Blast Radius

- `trading_app/ai/provider_registry.py` — modifies the canonical AI profile registry. Reads: env vars only. Writes: none. Callers of `AIProfile.validation_errors()`: `assert_ready` (same module), `to_metadata` (same module), and tests. The refactor changes `validation_errors()` behavior ONLY for profiles where `runtime_class == "interactive_editor"` — zero existing profile has that runtime_class today, so existing-profile behavior is byte-identical.
- `pipeline/check_drift.py` — adds `--quiet` flag and one new check. Callers: `.githooks/pre-commit` (calls `python pipeline/check_drift.py` with no args; behavior unchanged when `--quiet` absent), CI workflows, and `Makefile` targets if any. The new check is a declarative no-op so it cannot regress the green-bar.
- `tests/test_trading_app/test_ai/test_provider_registry.py` — companion tests, additive only.
- `tests/test_pipeline/test_check_drift.py` — companion tests, additive only.
- `CONVENTIONS.md` — NEW file at repo root. Zero callers. Read by `aider` automatically when present (Source 19) — currently no aider usage in repo so impact is zero today.
- `.aiderignore` — NEW file at repo root. Zero callers in production code; consumed only by `aider` CLI when invoked. Zero impact today.
- Reads: env vars (read-only), repo files (read-only via existing patterns).
- Writes: none against gold.db or any persistent state.
- Risk: low. Reversibility: revert the commit.

## Approach

1. **Refactor first, add second.** Land the `validation_errors()` runtime_class gating BEFORE adding the new profile, so the test suite proves the existing-profile-rejection regression guard with the change-in-isolation. Then add the new profile in a separate hunk.
2. **Drift check `--quiet` is one helper, not 60 print rewrites.** Change the orchestrator function (the one that prints PASS/FAIL summary at the end) to read an `args.quiet` flag and emit the sanitized form. The per-check functions still return `list[str]` of error messages — no per-check rewrites needed.
3. **No-op drift check is intentional.** `check_deepseek_review_gate_intact` returns `[]` when `.githooks/pre-commit` does not contain the step-0d marker (Phase 3 lands that marker). When marker is absent → no-op. When marker is present + missing → fails. This means Phase 1 commits with a green-bar, Phase 3 lands the marker AND flips the check, in the same PR.
4. **`CONVENTIONS.md` is pointers, not restated rules.** Per integrity-guardian rule #1 (Authority Hierarchy: defer, never restate). Each section: a one-line rule + a path to the canonical doc.
5. **`.aiderignore` is tested by walking the would-include set.** Test: `git -C <repo> ls-files | python -c "match against .aiderignore" | wc -l` — assert the include set is the bounded subset (pythonsrc + root-md + governance/specs/institutional docs), not the full 2654-file repo.

## Existing patterns reused

- `AIProfile` dataclass + `PROFILE_REGISTRY` dict — direct extension of the foundation merged in PR #220.
- `_csv_tuple` / `_profile_env_prefix` / `resolved()` env-override pattern — reused verbatim for `deepseek_coding`.
- `register_check` / `CHECKS = [...]` registration in `pipeline/check_drift.py` — same pattern as 40+ existing checks.
- argparse pattern in `pipeline/check_drift.py` — same `add_argument` shape as existing flags.

## Out of scope (Phase 1)

- No launcher (`scripts/tools/deepseek-agent.ps1`) — Phase 2.
- No bake-off harness — Phase 2.5.
- No commit gate / `.githooks/pre-commit` step 0d — Phase 3. The drift check that asserts step-0d-presence is registered in Phase 1 as a no-op; flipped in Phase 3.
- No server-side gate / GitHub Action — Phase 4.
- No `.aider.conf.yml` — Phase 2 (depends on launcher).
- No reviewer (`claude_review_deepseek.py`) — Phase 3.
- No `check_or_credits.py` / `check_or_models.py` — Phase 2 / Phase 4.
- No live model selection — Phase 2.5.
- No live OR API call — Phase 1 only adds the profile descriptor; nothing calls it yet.
- No MCP wiring — explicit accepted limitation per v4 plan ("Out of scope for v4").
- No edits to `docs/runtime/stages/deepseek-agent-launcher.md` (the v3 stage file). That file lives only on the `plan/live-trading-rollout-2026-05-05` working tree (untracked), not on `origin/main`. It will be deleted on that branch by its owner if it was ever an artifact of an old session.

## Acceptance criteria (Phase 1)

All required:

1. `python -c "from trading_app.ai.provider_registry import PROFILE_REGISTRY; p = PROFILE_REGISTRY['deepseek_coding']; print(p.profile_id, p.runtime_class, p.mutation_allowed)"` prints `deepseek_coding interactive_editor True`.
2. `python -c "from trading_app.ai.provider_registry import get_profile; p = get_profile('deepseek_coding'); print(p.validation_errors())"` returns a list containing `"model not configured; set CANOMPX3_AI_DEEPSEEK_CODING_MODEL for this profile"` (fail-closed by design).
3. `CANOMPX3_AI_DEEPSEEK_CODING_MODEL=deepseek/deepseek-v3.2-exp python -c "..."` returns `[]` (no errors with model env override).
4. `python pipeline/check_drift.py` passes (no regressions).
5. `python pipeline/check_drift.py --quiet` produces only `PASS` / `FAIL: <name> (count=N)` lines for each registered check — no diagnostic detail, no file paths, no SQL fragments, no DB internals.
6. `python -m pytest tests/test_trading_app/test_ai/test_provider_registry.py tests/test_pipeline/test_check_drift.py` all pass.
7. `git diff --stat` on this stage's commit shows exactly the 6 files listed in scope_lock — no scope creep.

## Verification log

- [x] `python pipeline/check_drift.py` output (PASS):
  - Tail: `NO DRIFT DETECTED: 120 checks passed [OK], 0 skipped (DB unavailable), 18 advisory`
  - Exit code: 0
  - New check #138 (`DeepSeek Coding Agent review gate (pre-commit step 0d) is intact`) registered and passes as no-op (Phase 3 will flip it to blocking once `# 0d.` marker lands).

- [x] `python pipeline/check_drift.py --quiet` output (sanitized):
  - Stdout-only inspection (`2>/dev/null`) → every line matches `^(PASS|FAIL|ADVISORY|SKIP):\s.+$|^SUMMARY:\s.+$`.
  - Tail: `SUMMARY: clean passed=120 advisory=18`
  - Exit code: 0. No diagnostic detail / file paths / SQL fragments / DB internals leaked.
  - Robustness: `_QuietSink` exposes `reconfigure(...)` no-op for modules that call `sys.stdout.reconfigure()` at import time (e.g. `trading_app.outcome_builder`); `_safe_label_for_quiet()` ASCII-folds non-cp1252 glyphs (e.g. `↔`) so the sanitized stream is encode-safe under Windows-default subprocess capture.

- [x] Test suite output (counts):
  - `pytest tests/test_trading_app/test_ai/test_provider_registry.py tests/test_pipeline/test_check_drift.py` → **153 passed, 0 failed** (10.69s).
  - New `TestDeepseekCodingProfile` class: 10 tests (registration, runtime_class, mutation_allowed, fail-closed router, model-None assert, model-env assert, openrouter-research-rejection, exclusion from research list, validation-errors regression guard for non-editor profile, and runtime-class-gate confirmation for editor profile).
  - New `TestDeepseekReviewGateNoOp`: 4 tests (real-repo no-op, missing-pre-commit no-op, marker-absent no-op, marker-present-but-invocation-missing → hard fail flip-test for Phase 3).
  - New `TestQuietModeOutputSanitization`: 2 tests (sanitization regex + summary contract).

- [x] `git diff --stat` snapshot:
  ```
  pipeline/check_drift.py                            | 174 ++++++++++++++++++---
  tests/test_pipeline/test_check_drift.py            | 102 ++++++++++++
  tests/test_trading_app/test_ai/test_provider_registry.py | 116 ++++++++++++++
  trading_app/ai/provider_registry.py                |  50 +++++-
  4 files changed, 411 insertions(+), 31 deletions(-)
  + new files: .aiderignore, CONVENTIONS.md
  ```
  All 6 scope_lock files touched; no scope creep. (Stage file `docs/runtime/stages/deepseek-coding-agent-v4.md` is untracked but not in scope_lock — it's the plan, not a code file.)

- [x] Self-review pass (institutional rigor rule #1):
  - **§1 (delegate canonical sources):** new profile reuses `AIProfile`/`PROFILE_REGISTRY`/`ProviderRouting`/`_csv_tuple`/`resolved()` verbatim — zero re-encoding.
  - **§4 (delegate, never re-encode):** `validation_errors()` refactor is a one-line guard, not a copy. `assert_openrouter_research_profile()` reuses existing `assert_ready()`. `list_openrouter_research_profiles()` filter respects single-source `runtime_class`.
  - **§5 (no dead code):** dead-code sweep on `deepseek_coding|interactive_editor|check_deepseek_review_gate_intact` returns exactly the 4 expected files (provider_registry + check_drift + their two test files). No orphan imports, no dead fields.
  - **§6 (no silent failures):** `check_deepseek_review_gate_intact` reads `.githooks/pre-commit` with explicit `OSError|UnicodeDecodeError` handling; missing-file path returns `[]` per documented fail-safe (matches the canonical pattern in the rest of `check_drift.py`).
  - **§8 (verify before claiming):** acceptance criteria 1-7 all green with paste-able execution evidence above.
  - **Treadmill check:** no recurrent fix pattern; one mid-flight finding (per-check stdout leakage in advisory checks) handled with one structural change (`_QuietSink` + suppression context), not patching.

## Done definition (Phase 1 only)

All four required (institutional rigor rule #8):

- [ ] Acceptance criteria 1-7 all green with execution evidence pasted into Verification log.
- [ ] Dead-code sweep: `grep -r "deepseek_coding\|interactive_editor\|check_deepseek_review_gate_intact" --include="*.py"` shows only the new code paths and their tests, no orphaned imports.
- [ ] `python pipeline/check_drift.py` passes (and `--quiet` mode passes too).
- [ ] Self-review (code-review skill or equivalent) run on the commit's diff; findings either resolved or explicitly accepted with rationale.
