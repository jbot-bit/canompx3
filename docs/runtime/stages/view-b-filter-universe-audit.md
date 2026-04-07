---
mode: IMPLEMENTATION
slug: view-b-filter-universe-audit
task: View B — Filter Universe Audit page. Add a collapsible Filter Universe Audit section to the trade-sheet HTML listing every filter in trading_app.config.ALL_FILTERS plus the ATR velocity overlay, cross-referenced against active validated_setups (routed count) and prop_profiles.ACCOUNT_PROFILES (deployed count), with confidence tier and last-revalidated metadata where available. Pure read-only addition. Closes View B of Phase 2 of the parent eligibility-context design.
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 1
scope_lock:
  - scripts/tools/generate_trade_sheet.py
  - tests/tools/test_generate_trade_sheet.py
  - docs/plans/2026-04-07-filter-universe-audit-design.md
  - docs/plans/2026-04-07-eligibility-context-design.md
  - docs/runtime/stages/view-b-filter-universe-audit.md
  - HANDOFF.md

blast_radius: Pure read-only addition to the trade sheet HTML. Adds one collapsible filter universe audit section below the existing session cards plus two new helper functions and new CSS rules. Touches two code files plus four doc files. NO changes to trading_app/config.py, trading_app/eligibility/*, trading_app/prop_profiles.py, pipeline/*, or any file in the e2-canonical-window-fix scope_lock. Zero schema changes, zero data rebuilds, zero entry model changes, zero trading logic changes, zero live bot changes. One-way dependency direction preserved (trade sheet reads from trading_app.config registry). Rollback via single revert of commit 2 (the wiring step) restores pre-stage behaviour because commits 1 and 3 are dead code without the wiring. See ## Blast Radius section for full detail.

## Blast Radius

Adds one collapsible `<details>` section to the trade-sheet HTML below the
session cards. Contents of the section are produced by two new helper
functions inside `scripts/tools/generate_trade_sheet.py`:

  `_build_filter_universe_rows(db_path, trading_day)`
    - opens ONE read-only duckdb connection
    - runs ONE aggregate query for routed counts by filter_type
    - walks `prop_profiles.ACCOUNT_PROFILES` for deployed counts
    - iterates `trading_app.config.ALL_FILTERS` + `ATR_VELOCITY_OVERLAY`
    - reads `VALIDATED_FOR`, `LAST_REVALIDATED`, `CONFIDENCE_TIER` ClassVars
    - returns a list of row dicts sorted by (deployed DESC, routed DESC)

  `_render_filter_universe_section(rows)`
    - pure function, no I/O, no DB
    - returns an HTML fragment with a `<details>`/`<summary>` block and a
      table of rows

Both helpers are wired into `generate_html` at a point after the existing
session card rendering. CSS additions live inside the existing `<style>`
f-string and follow the existing dark-theme palette: three new row-class
rules (`row-live`, `row-routed`, `row-dead`), one new badge class
(`badge-filter-live`), and one new section-wrapper class
(`filter-universe`).

Dependencies READ (not written):

  - `trading_app.config.ALL_FILTERS` — 82-entry filter registry
  - `trading_app.config.ATR_VELOCITY_OVERLAY` — separate overlay instance
  - `trading_app.config.StrategyFilter.VALIDATED_FOR` ClassVar (optional)
  - `trading_app.config.StrategyFilter.LAST_REVALIDATED` ClassVar (optional)
  - `trading_app.config.StrategyFilter.CONFIDENCE_TIER` ClassVar (optional)
  - `trading_app.prop_profiles.ACCOUNT_PROFILES` — deployed lane lookup
  - `trading_app.eligibility.builder.parse_strategy_id` — for deployed
    lane filter_type extraction (already imported in commit 1 of the
    predecessor stage)
  - `gold.db` → `validated_setups` table (read-only SELECT)

Consumer chain: nothing downstream of `generate_trade_sheet.py` consumes
the new helpers. The HTML fragment is embedded in the standalone
`trade_sheet.html` output file which is opened in the user's browser.

Rollback: four-commit structure. Commit 2 (wiring) is the only
behavior-changing commit. `git revert <commit 2 sha>` restores
pre-stage behaviour because commits 1 and 3 are dead code without
the wiring.

acceptance:
  G1_section_present:
    test: "grep -c 'Filter Universe' trade_sheet.html"
    expect: "at least 1"
  G2_row_per_filter:
    test: "new pytest test walks the helper and asserts len(rows) == len(ALL_FILTERS) + 1"
    expect: "83 rows (82 + overlay)"
  G3_deployed_count_matches:
    test: "sum of row['deployed'] == count of active profile daily lanes"
    expect: "5 (current deployed lane count)"
  G4_live_rows_have_deployed_gt_zero:
    test: "all rows with status=LIVE have deployed > 0"
    expect: "True for every row"
  G5_routed_rows_have_routed_gt_zero_deployed_zero:
    test: "all rows with status=ROUTED have routed > 0 and deployed == 0"
    expect: "True for every row"
  G6_dead_rows_have_both_zero:
    test: "all rows with status=DEAD have routed == 0 and deployed == 0"
    expect: "True for every row"
  G7_smoke_test:
    test: "PYTHONPATH=. python scripts/tools/generate_trade_sheet.py --no-open"
    expect: "exit 0, HTML file written, 'Filter Universe' present in HTML"
  G8_drift_unchanged:
    test: "PYTHONPATH=. python pipeline/check_drift.py 2>&1 | grep -c 'FAILED:'"
    expect: "1 (pre-existing #57 only)"
  G9_all_tests_pass:
    test: "pytest tests/tools/test_generate_trade_sheet.py tests/test_trading_app/test_eligibility_builder.py -q"
    expect: "all tests pass (19 + 65 existing + 3 new = 87)"
  G10_no_e2_scope_collision:
    test: "git diff --name-only (stage commits) | grep -E e2 scope_lock files"
    expect: "empty"

pre_commit_bypass_note: |
  All commits in this stage use `git commit --no-verify` because pre-existing
  drift check #57 (MGC 2026-04-06 partial daily_features row, root cause =
  bars_1m ingestion gap for 2026-04-06) is unrelated to this refactor and
  cannot be fixed without re-downloading Databento data (out of scope per
  HANDOFF 'track separately' note). This matches the pattern used by the
  predecessor trade-book canonicalization stage (commits d23d8c3 through
  5a4653b) and by the earlier institutional docs commits (1d15b35, 81d38dc,
  b70e56a).

phases:
  phase_1a_design:
    status: complete
    file: docs/plans/2026-04-07-filter-universe-audit-design.md
  phase_1b_stage:
    status: in_progress
    file: docs/runtime/stages/view-b-filter-universe-audit.md
  phase_2_commit_1_helpers:
    status: pending
    note: Add _build_filter_universe_rows + _render_filter_universe_section + CSS, all dead code
  phase_3_commit_2_wire:
    status: pending
    note: Wire audit section into generate_html via _render_filter_universe_section call
  phase_4_gate_1_review:
    status: pending
    note: code-review skill over commits 1+2
  phase_5_commit_3_tests:
    status: pending
    note: 3 new pytest tests for the helpers
  phase_6_gate_2_smoke:
    status: pending
    note: smoke test + final verification
  phase_7_commit_4_docs:
    status: pending
    note: HANDOFF update + parent design update + stage file delete
  phase_8_push:
    status: pending
    note: git push to origin/main

---

# Stage: View B — Filter Universe Audit

## Purpose

Surface every filter in the registry (`ALL_FILTERS`) alongside a count
of active validated strategies using it and a count of profile lanes
currently deploying it. Implements the second half of the
eligibility-context design's View A + View B split. View A (deployed
lane signal column) landed earlier today in commits `d23d8c3` through
`5a4653b`. This stage completes Phase 2.

## Why this is not gold-plating

The parent design (`2026-04-07-eligibility-context-design.md` § "Design
Principles") explicitly names View B as a counterweight to View A:
*"hiding the test universe inflates apparent edge (confirmation bias)"*.
View A answers "what am I trading today?". View B answers "what filters
exist at all, and which ones aren't working?". Both are needed to
debug the filter routing without confirmation bias.

Concrete use cases for View B (from today's smoke test):
- "I see 9 OVNRNG_50 strategies routed but none deployed — is that because
  they failed validation or because I haven't plugged them into a profile
  yet?" → View B shows ROUTED badge + 9/0 routed/deployed counts.
- "FAST5 is in the registry but I don't recall deploying it anywhere." →
  View B shows DEAD badge if unrouted, or ROUTED badge with a count if
  routed but not deployed.
- "How stale is the research behind X_MES_ATR70?" → View B shows
  last_revalidated + STALE pill if > 180 days.

Without View B, the trader has to open the code, read `ALL_FILTERS`, and
cross-reference by hand. With View B, it's one collapsible HTML section.

## Out of scope

- Filter annotation hardening (adding `VALIDATED_FOR`, `LAST_REVALIDATED`,
  `CONFIDENCE_TIER` to the 70+ unannotated filter variants) — needs research
  and is a separate stage.
- Dead filter cleanup (deleting unrouted registry entries) — stakeholder
  decision.
- Drift check for metadata coverage — locked by E2 worktree scope.
- Dashboard live integration (Phase 3 of parent plan) — separate stage.

## Files NOT touched

- `trading_app/config.py` — filter registry, READ-ONLY access only.
- `trading_app/eligibility/*` — View B does not call the eligibility builder.
- `trading_app/prop_profiles.py` — READ-ONLY walk.
- `pipeline/*` — unchanged.
- `trading_app/live/*` — unchanged.
- Every file in the `e2-canonical-window-fix` scope lock — untouched.

## Process compliance

- [x] Design doc saved
- [x] Stage file written
- [x] User approval received ("go")
- [ ] Commits 1-4 executed with inter-phase code reviews
- [ ] Gate G1-G10 verified
- [ ] HANDOFF updated
- [ ] Stage file deleted on close
