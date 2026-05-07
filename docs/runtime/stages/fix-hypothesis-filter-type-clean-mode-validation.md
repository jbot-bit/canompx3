---
task: fail-closed validation of hypothesis filter_type against ALL_FILTERS in clean mode
mode: IMPLEMENTATION
scope_lock:
  - trading_app/hypothesis_loader.py
  - trading_app/strategy_discovery.py
  - tests/test_trading_app/test_hypothesis_loader.py
  - tests/test_trading_app/test_strategy_discovery.py
acceptance:
  - hypothesis_loader raises HypothesisLoaderError on unregistered filter_type in BOTH clean and proxy mode
  - strategy_discovery._inject_hypothesis_filters raises (defense-in-depth) instead of silent `continue` on unknown filter_type
  - regression tests cover both gates (clean-mode loader + injection-bypass mutation)
  - all existing tests pass; check_drift.py 122 PASS; behavioral audit 7/7
---

## Blast Radius

- trading_app/hypothesis_loader.py — lift `ALL_FILTERS.get(filter_type) is None -> raise` out of the `if proxy_mode:` block at line ~750. Now applies in every mode. The `requires_micro_data` check stays inside the proxy-mode branch (genuinely proxy-specific).
- trading_app/strategy_discovery.py — replace `_inject_hypothesis_filters` line 1131 silent `continue` (when filter is not in ALL_FILTERS) with `raise HypothesisLoaderError`. Defense-in-depth — the upstream fix should make this unreachable, but the explicit raise documents the invariant.
- tests/test_trading_app/test_hypothesis_loader.py — add `test_clean_mode_rejects_unknown_filter` (mirrors existing proxy-mode test pattern).
- tests/test_trading_app/test_strategy_discovery.py — replace `test_hypothesis_filter_injection_skips_unknown_filter_type` with `test_hypothesis_filter_injection_raises_on_unknown_filter_type` (the old test pinned the prior silent-skip behavior — the new test pins the fail-closed behavior). Class-level docstring also updated.
- Reads: gold.db (read-only via existing tests). Writes: none.
- Behavioral change: invalid pre-reg files (under `--hypothesis-file`) now hard-fail at load time instead of running to completion with zero combos and a logger.warning the operator could miss.

## Pre-existing pre-reg compatibility (verified)

- Empirical scan (script in plan § Verification): 9 occurrences of unregistered filter_types across 9 pre-reg YAMLs.
- Of those: 5 are theoretical/archived (closed verdicts or framework docs, not loaded by any active workflow), 3 are PR #48 family (REL_VOL_STATE) deferred to Stage A, 1 is F5_BELOW_PDL deferred to Stage B.
- Verified that `load_hypothesis_metadata()` does NOT call `extract_scope_predicate()`. The 3 active scripts that load these YAMLs (`scripts/research/cross_session_context_audit.py`, `scripts/research/external_context_role_audit.py`, `research/mnq_parent_structure_shadow_buckets_v1.py`) call `load_hypothesis_metadata` only — none invoke `strategy_discovery.run_discovery(--hypothesis-file=...)`. The new gate has zero impact on currently-running scripts.

## Doctrine cited

- integrity-guardian.md § 3 — Fail-closed mindset; never silently `continue` in audit/health paths
- institutional-rigor.md § 6 — No silent failures
- RESEARCH_RULES.md Rule 11 — Never delete prior results (drives the deferred-archive decision)

## Doctrine NOT cited (corrected from initial plan draft)

- pre_registered_criteria.md Criterion 2 — initial plan draft cited this; bias-corrected after re-reading the canonical doc. MinBTL N comes from `metadata.total_expected_trials` (researcher-declared), NOT from `total_declared_trials` (predicate-computed). The two are independent. The bug being fixed is a silent zero-combos discovery surface, not a MinBTL inflation.

## Audit context

Found during Round 3 code review pass on trading_app/strategy_discovery.py (2026-05-07). The pre-fix comment at line 1131 ("scope predicate will reject anyway") was performative — the predicate has nothing to reject when the typo'd filter is silently dropped from the grid. Verified empirically pre-fix vs post-fix via stash/pop on a real PR #48 pre-reg (`2026-04-22-pr48-promotion-shortlist-v1.yaml` declares filter `REL_VOL_STATE`):
- Pre-fix: `extract_scope_predicate()` returns `total_declared_trials=1, allowed_filter_types={'REL_VOL_STATE'}` — does NOT raise.
- Post-fix: same call raises `HypothesisLoaderError("uses filter.type 'REL_VOL_STATE', but the filter is not registered in trading_app.config.ALL_FILTERS")`.

## Out of scope (deferred to follow-up stages)

- **Stage A** — REL_VOL_STATE register-or-archive decision (1-page mini-audit; 3 PR #48 pre-regs).
- **Stage B** — F5_BELOW_PDL register + retroactive validation against the 2026-04-15 evidence.
- **Stage C** — Audit `2026-04-15-mega-deployed-lanes-only.md` for hand-coded-filter contamination (broader research-truth-protocol scope per `feedback_aperture_overlay_canonical_parser.md`).
- **Stage D** — Archive 5 theoretical-construct pre-regs to `docs/audit/hypotheses/archive/2026-04/`. Originally planned in this PR but deferred after empirical grep showed 9+ docs and 3+ scripts reference these by exact path. Archive requires either updating every reference (significant scope creep) or accepting silent link rot. New stage will plan this with a verified file list and reference-update plan.
