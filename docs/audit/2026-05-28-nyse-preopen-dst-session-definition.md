# Adversarial-Audit Gate Artifact — NYSE_PREOPEN dst.py session definition (Lane B Stage 1)

**Date:** 2026-05-28
**Gate:** `.claude/rules/adversarial-audit-gate.md` (CRIT-path canonical edit in `pipeline/dst.py`)
**Actor:** independent-context `evidence-auditor` pass (separate conversation; commit-message and code-comment claims treated as claims requiring proof, falsified by executed output not by reading)
**Commit under review:**

- `62c51a14` — `feat(sessions): add NYSE_PREOPEN dynamic session (09:00 ET order-imbalance) + DOW-classification hardening`

**Why this artifact exists:** `pipeline/dst.py` is the canonical session-timing source (every outcome / feature / execution path reads window timing from it). The Stage 1 stage file (`docs/runtime/stages/2026-05-27-nyse-preopen-session-build-stage1.md` § Acceptance, lines 126-128) mandates an independent-context adversarial-audit gate BEFORE any Stage 2 dispatch. The audit was deferred from the implementing session (context budget) and run at the start of the 2026-05-28 session for greater independence. This file is the required searchable record.

---

## Verdict: CONDITIONAL — zero critical issues. Stage 2 UNBLOCKED.

All six claims verified by executed output. The single CONDITIONAL condition is the recommendation to close a forward-looking silent gap (`session_guard._SESSION_ORDER`) before Stage 3 builds features — not a defect in the commit under review. No capital-impact or data-integrity defect found.

## Per-claim findings (PREMISE → TRACE → EVIDENCE → CONCLUSION)

1. **Resolver correctness, both DST regimes — CONFIRMED by execution.**
   `nyse_preopen_brisbane(date(2024,6,15)) = (23,0)` (summer EDT, 09:00 ET = 13:00 UTC = 23:00 AEST same cal day); `nyse_preopen_brisbane(date(2024,12,15)) = (0,0)` (winter EST, = 14:00 UTC = 00:00 AEST next cal day). Matches the docstring exactly. The "mirrors `us_equity_open_brisbane` tz arithmetic exactly, no hardcoded offsets" claim is true — the resolver builds a 09:00 `America/New_York` aware datetime and converts to Brisbane, identical structure to the sibling at `dst.py:326`.

2. **`orb_utc_window` resolves both regimes, exactly 30 min before NYSE_OPEN — CONFIRMED.**
   Summer 5m: NYSE_PREOPEN `13:00→13:05 UTC` vs NYSE_OPEN `13:30→13:35` — 30 min gap. Winter (trading_day 2024-12-16): NYSE_PREOPEN `14:00→14:05 UTC` vs NYSE_OPEN `14:30→14:35`. Winter midnight-crossing maps to the correct Brisbane trading_day via the `hour < TRADING_DAY_START_HOUR_LOCAL` cal-date bump, identical to NYSE_OPEN. No raise on 5/15/30m. No silent day-shift.

3. **DOW `-1` offset is conservative-correct, not a latent bug — CONFIRMED.**
   `validate_dow_filter_alignment` (`dst.py:245-252`) fires for ANY session registered in `DOW_MISALIGNED_SESSIONS` whenever a DOW filter (`skip_days`) is active, regardless of DST regime. In summer EDT, NYSE_PREOPEN resolves to 23:00 Brisbane same calendar day = DOW-*aligned*, so the static `-1` registration causes the guard to OVER-block a (valid) summer DOW filter. Critically, it never PERMITS a wrong DOW filter — the failure direction is fail-closed. This is the identical convention NYSE_OPEN already carries (`dst.py` DOW comment block lines 198-217), and no live strategy uses a DOW filter on either session (grep for `DayOfWeek`/`skip_days` × `NYSE_OPEN`/`NYSE_PREOPEN` → 0 hits). The commit's "fail-closes on any DOW filter for this session" claim is accurate.

4. **New drift check `check_dow_classification_complete` is genuinely fail-closed — CONFIRMED by injection.**
   `python pipeline/check_drift.py` passes with the check registered (`check_drift.py:14016`). Injection probe: a fabricated unclassified dynamic session fires `"Dynamic sessions not DOW-classified..."`; a session placed in both DOW sets fires `"Sessions in BOTH..."`. Both branches confirmed live; injection reverted, clean catalog returns `[]`.

5. **Five-list atomicity complete for Stage 1; session_guard gap genuinely deferred — CONFIRMED.**
   NYSE_PREOPEN present in all five canonical lists (resolver, SESSION_CATALOG, DST_CLEAN_SESSIONS, DOW_MISALIGNED_SESSIONS, ORB_LABELS_DYNAMIC). `trading_app/config.py`, `cascade_table.py`, `conditional_overlays.py`: zero NYSE_PREOPEN hits required at this stage. `session_guard._SESSION_ORDER` (`session_guard.py:38`) does NOT list NYSE_PREOPEN — but no production caller routes NYSE_PREOPEN through `_session_index` until features exist (Stage 3), so the gap is inert now and correctly deferred.

6. **Tests pass at the claimed counts — CONFIRMED.**
   `tests/test_pipeline/test_dst.py` + `tests/test_app_sync.py` = 162 (105 + 57). `tests/test_pipeline/test_check_drift.py` = 222 collected; DOW subset 8 passed. One pre-existing environmental timeout on `test_quiet_mode_lines_are_sanitized` (subprocess + gold.db read-lock contention on Windows — NOT introduced by this commit).

## Critical issues
NONE. No capital-impact or data-integrity defect.

## Silent gaps
- `pipeline/session_guard.py:_SESSION_ORDER` (line 38) omits NYSE_PREOPEN. The hazard is QUIETER than a `ValueError`: `_SESSION_ORDER` builds the regex `_SESSION_COL_RE` (`session_guard.py:110`), so an absent session is invisible to that regex and its `orb_NYSE_PREOPEN_*` columns fall through `is_feature_safe`'s "Unknown column → fail CLOSED" branch (lines 158-159) and get masked for EVERY target session — a silent look-ahead-mask exclusion. Inert until Stage 3 builds features. Closed by Stage 3 (insert between US_DATA_830 and NYSE_OPEN; add `_WINDOW_FEATURES` safe-after entries per `backtesting-methodology.md` RULE 1.2).

## Unsupported assumptions
- Comment at `dst.py:199` marks NYSE_PREOPEN DOW with "✗" — accurate for winter EST only; summer EDT is aligned. Imprecise documentation, not a code defect (the guard behavior is correct per claim 3).
- Stale source-of-truth pointer at `session_guard.py:36` ("trading_app/ml/config.py SESSION_CHRONOLOGICAL_ORDER") — the ML subsystem was removed in the V3 sprint; `check_session_guard_sync` (`check_drift.py:5366`) is now a registry-stable no-op and `_SESSION_ORDER` stands alone as canonical. A Stage-3 parity check should reconcile `ORB_LABELS ⊆ _SESSION_ORDER`, not against the dead ml/config constant.

## Tests missing
- No test asserts the summer DOW-aligned behavioral difference (hour=23, same Brisbane calendar day). Low severity given the guard fails closed.
- No test for the `_SESSION_ORDER` deferred gap (a session in ORB_LABELS absent from `_SESSION_ORDER`). Added by Stage 3's parity drift check.

## Do-not-touch (audit-verified correct)
- `pipeline/dst.py` `orb_utc_window` midnight-crossing branch (~lines 578-656) — verified correct for the NYSE_PREOPEN winter case; do not refactor.
- The `-1` DOW classification for NYSE_PREOPEN — verified conservative-correct, matching the proven NYSE_OPEN convention.
- `nyse_preopen_brisbane` resolver tz arithmetic — verified to mirror `us_equity_open_brisbane`.

## Highest-priority fix
Close the `session_guard._SESSION_ORDER` silent gap before Stage 3 builds features: insert `NYSE_PREOPEN` between `US_DATA_830` and `NYSE_OPEN` (chronological position verified by resolver: US_DATA_830 22:30/23:30, NYSE_PREOPEN 23:00/00:00, NYSE_OPEN 23:30/00:30 Brisbane summer/winter), add its `_WINDOW_FEATURES` safe-after entries, and add a drift check asserting `ORB_LABELS ⊆ _SESSION_ORDER`. Scheduled as Stage 3.
