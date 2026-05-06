---
task: Close capital-class governance hole — _diff_is_doc_only and small-diff skip in claude_review_deepseek.py bypass canonical-source files (notably docs/runtime/lane_allocation.json)
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/claude_review_deepseek.py
  - tests/test_tools/test_claude_review_deepseek.py
  - pipeline/check_drift.py
---

## Blast Radius

- `scripts/tools/claude_review_deepseek.py` — adds canonical-source allowlist; tightens `_diff_is_doc_only()` and small-diff (`<5 lines`) bypass so they NEVER fire on capital-class files.
- `tests/test_tools/test_claude_review_deepseek.py` — new file. Companion tests prove every canonical-source path triggers review even when the diff is doc-only or small.
- `pipeline/check_drift.py` — new drift check (numbered after #140) that asserts the allowlist constant exists and contains every entry the canonical-source table requires. Fail-closed: drift check FAILS if `lane_allocation.json` is missing from the allowlist.
- Reads: nothing new. The reviewer already reads staged diff via `git diff --cached`.
- Writes: nothing.
- Live order-route impact: NONE — this strengthens the gate, never weakens it. A commit that PREVIOUSLY passed will still pass UNLESS it touches a canonical-source file (in which case it will now correctly trigger review).

## Why this is IMPLEMENTATION not DESIGN

The gap was identified by an independent code-review pass. The fix is mechanically obvious: the `_diff_is_doc_only` and `<5 lines` short-circuits at `claude_review_deepseek.py:223-226` must check a canonical-source allowlist FIRST. No design-space ambiguity. Companion tests are required (per institutional-rigor § 1) but the structure is determined by the existing test layout.

## Source-grounded canonical allowlist

Files added to allowlist with consumer-source evidence (no metadata, no training memory):

| Path | Consumer (file:line) | Capital impact |
|---|---|---|
| `docs/runtime/lane_allocation.json` | `trading_app/lane_allocator.py:54`, `trading_app/live/session_orchestrator.py:392`, `trading_app/pre_session_check.py:578`, `trading_app/prop_profiles.py:1119` | Drives live lane selection / regime gate |
| `docs/runtime/chordia_audit_log.yaml` | `trading_app/chordia.py:165` | Chordia gate truth for validation |
| `pipeline/cost_model.py` | imported by every research scan + live engine | Friction inputs for every order |
| `pipeline/dst.py` | session times, holdout, ORB UTC window | Look-ahead bias surface |
| `pipeline/asset_configs.py` | active instruments + tick math | Determines what trades |
| `pipeline/paths.py` | `GOLD_DB_PATH`, `LIVE_JOURNAL_DB_PATH` | DB locations |
| `pipeline/holdout_policy.py` (via `trading_app/holdout_policy.py`) | sacred-window constants | IS/OOS classification |
| `trading_app/prop_profiles.py` | `ACCOUNT_PROFILES` (live profiles) | Account / risk caps |
| `trading_app/lane_ctl.py` | `get_paused_strategy_ids` | Live pause state |
| `trading_app/config.py` | `ALL_FILTERS`, `E2_EXCLUDED_FILTER_PREFIXES` | Filter logic + E2 look-ahead gate |
| `trading_app/cost_model.py` | (alias / wrapper if any) | Friction |
| `trading_app/holdout_policy.py` | `HOLDOUT_SACRED_FROM`, `enforce_holdout_date` | IS/OOS |
| `trading_app/eligibility/builder.py` | `parse_strategy_id` (canonical parser) | Strategy ID semantics |
| `trading_app/entry_rules.py` | `detect_break_touch` (E2 entry math) | Live entry math |
| `pipeline/cost_model.py` (already listed) | — | — |

If a future canonical source is added, the drift check forces the allowlist update.

## Implementation

1. Add module-level constant `_CANONICAL_SOURCE_ALLOWLIST: frozenset[str]` populated from the table above.
2. Add helper `_diff_touches_canonical_source(diff: str) -> bool` that parses the same `+++ b/<path>` / `--- a/<path>` markers as `_diff_is_doc_only` and returns True if any path matches the allowlist.
3. In `main()`: REORDER the early-return logic so `_diff_touches_canonical_source(diff)` is checked BEFORE `_diff_is_doc_only(diff)` and BEFORE the `len(diff.splitlines()) < 5` check. If a canonical source is touched, the size + doc-only short-circuits are bypassed and the diff goes to Claude regardless.
4. Add drift check `check_canonical_source_review_allowlist_complete` to `pipeline/check_drift.py`. The check:
   - Imports `_CANONICAL_SOURCE_ALLOWLIST` from the reviewer module.
   - Asserts the literal set `{"docs/runtime/lane_allocation.json", "docs/runtime/chordia_audit_log.yaml", "pipeline/cost_model.py", "pipeline/dst.py", "pipeline/asset_configs.py", "pipeline/paths.py", "trading_app/prop_profiles.py", "trading_app/lane_ctl.py", "trading_app/config.py", "trading_app/holdout_policy.py", "trading_app/eligibility/builder.py", "trading_app/entry_rules.py"}` is a subset of the constant. (Subset, not equality, so the allowlist can grow without breaking the check.)
   - FAILS if any canonical entry is missing.

## Companion tests (new file `tests/test_tools/test_claude_review_deepseek.py`)

Every test runs with `OPENCODE_AGENT_ACTIVE=1` set via `monkeypatch.setenv`.

1. **`test_diff_touches_canonical_source_lane_allocation`** — synthesize a 3-line diff that modifies only `docs/runtime/lane_allocation.json`. Assert `_diff_touches_canonical_source(diff) is True`.
2. **`test_canonical_source_overrides_doc_only_skip`** — synthesize a doc-only diff that includes `docs/runtime/lane_allocation.json`. Patch `_call_claude` to return `("BLOCK", {...})`. Run `main(--mock --rubric-fail)` — assert exit code 1 (BLOCKED), proving the doc-only short-circuit did NOT fire.
3. **`test_canonical_source_overrides_small_diff_skip`** — synthesize a 2-line diff modifying only `pipeline/cost_model.py` (well under 5 lines). Run with `--mock --rubric-fail`. Assert exit code 1.
4. **`test_doc_only_md_still_skips_when_no_canonical_source`** — synthesize a diff modifying only `docs/audit/results/2026-05-07-foo.md`. Run normally (no Claude call); assert exit code 0 (skipped via doc-only path — preserves existing behavior).
5. **`test_small_pure_md_diff_still_skips`** — 3-line diff to `README.md`. Assert exit code 0.
6. **`test_canonical_source_path_table`** — assert every entry in the design-table set is present in `_CANONICAL_SOURCE_ALLOWLIST`. Lock the contract at the test layer too (defense-in-depth alongside the drift check).
7. **`test_diff_parser_handles_renames`** — `+++ b/docs/runtime/lane_allocation.json` after `--- a/docs/runtime/lane_allocation.json` (no rename); also test `+++ b/<path>` only (new file). Both must register the canonical hit.
8. **`test_diff_parser_handles_dev_null`** — `--- a/docs/runtime/lane_allocation.json` followed by `+++ /dev/null` (deletion). Asserts canonical hit (deleting a canonical file is even higher-impact).

## Acceptance criteria — ALL required before claiming done

1. `pytest tests/test_tools/test_claude_review_deepseek.py -v` → 8/8 PASS (show output).
2. `python pipeline/check_drift.py` → 122+ PASS, 0 FAIL, 0 NEW SKIPPED.
3. `grep -n "_diff_touches_canonical_source" scripts/tools/claude_review_deepseek.py` → exactly one definition + one call site.
4. `grep -n "_CANONICAL_SOURCE_ALLOWLIST" scripts/tools/claude_review_deepseek.py` → constant defined + used in helper + (transitively) referenced by drift check.
5. **Pressure-test (per backtesting-methodology RULE 13 / institutional-rigor § 1):** manually stage a 1-line edit to `docs/runtime/lane_allocation.json`, run `OPENCODE_AGENT_ACTIVE=1 python scripts/tools/claude_review_deepseek.py --mock --rubric-fail` — confirm exit code 1 (BLOCKED). Without this fix the same scenario would exit 0 (silently approved). Show stdout/stderr in commit message.
6. **Verify the pre-existing pass-paths still work:** stage a `README.md`-only edit, run the same command — exit 0. A 100-line `pipeline/check_drift.py` non-canonical edit (e.g. comment-only) — exit 0 only if Claude APPROVE. Demonstrate the gate hasn't over-tightened.
7. Self-review pass via `code-review` skill (institutional-rigor § 1) — produce line citations not narrative. Then re-run all gates if any fix required.

## Out of scope

- Changing the rubric or the model used.
- Restructuring the reviewer's exit-code semantics.
- Adding new canonical sources beyond what current production code reads. The list is grounded in actual `import` / file-load sites; speculative future additions are deferred.

## Risk register

- **Risk:** Allowlist drifts as new canonical sources are added. **Mitigation:** drift check `check_canonical_source_review_allowlist_complete` enforces the minimum subset; ratchet adds future entries.
- **Risk:** Diff parser misses a path-with-spaces edge case. **Mitigation:** test 7 covers new-file (`+++` only) and test 8 covers deletion (`/dev/null`); standard `git diff --cached` does not put spaces in paths in this repo (verified by `find . -name "* *" | head` returning 0 hits in scope-locked dirs).
- **Risk:** False-positive — a genuinely doc-only commit that happens to touch `docs/runtime/lane_allocation.json` (e.g. comment-only inside the JSON). **Mitigation:** this is BY DESIGN. The fix's whole point is that any change to `lane_allocation.json`, including "trivial" ones, requires Claude review. Comment-only changes to JSON files don't exist anyway (JSON has no comments).
- **Risk:** Performance — the allowlist check fires on every commit. **Mitigation:** O(N_files_in_diff) set lookup, microseconds; no concern.
