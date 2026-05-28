---
task: |
  Lane B Stage 2 — NYSE-holiday contamination source for NYSE_PREOPEN.
  NYSE_PREOPEN is defined by the NYSE 09:00-ET cash-market order-imbalance
  publication. On NYSE cash holidays (July 4, Thanksgiving, MLK, Good Friday,
  etc.) that event does NOT occur, so any ORB built from thin holiday Globex
  bars is contaminated. The prereg names holiday contamination an explicit kill
  criterion. Stage 2 adds a CANONICAL fail-closed NYSE-holiday source and wires
  it into the NYSE_PREOPEN ORB build path so holiday days produce NO ORB.

  KEY FINDING (this session): the correct source is the XNYS (NYSE cash)
  calendar, NOT the existing is_cme_holiday (CMES). Verified by execution:
  is_cme_holiday(2024-07-04)=False (CME equity-index futures trade a shortened
  Globex session on July 4), but XNYS.is_session(2024-07-04)=False (NYSE cash
  IS closed — no order-imbalance event). Using is_cme_holiday would let
  contaminated July-4 rows into the backtest. exchange_calendars exposes XNYS;
  coverage 2006-05-30 -> 2027-05-28 (fully covers our data horizon).

  POLICY SPLIT (deliberate): is_cme_holiday is fail-OPEN (trade rather than
  miss, correct for live). is_nyse_holiday for BACKTEST builds is fail-CLOSED
  (strict=True): a date beyond XNYS coverage RAISES rather than silently
  assuming "open", because a contaminated holiday ORB feeding the prereg is a
  kill criterion. strict=False preserves the fail-open contract for any future
  live use. Grounded: backtesting-methodology.md RULE 1 (silent contamination)
  + the prereg kill criterion.
mode: IMPLEMENTATION
updated: 2026-05-28T00:00Z
agent: claude (opus 4.7)
supersedes: none

scope_lock:
  - pipeline/market_calendar.py
  - pipeline/build_daily_features.py
  - tests/test_pipeline/test_market_calendar.py
  - tests/test_pipeline/test_build_daily_features.py

## Blast Radius
- pipeline/market_calendar.py — adds is_nyse_holiday(d, *, strict=True) using a new _XNYS = xcals.get_calendar("XNYS"), via is_session() (NOT sessions_in_range — that gave a false negative on MLK 2024-01-15 in testing; is_session is authoritative). NEW function, ZERO existing callers. Deterministic, no DB/network. strict=True raises on beyond-coverage dates (fail-closed for backtests); strict=False returns False + WARNING (fail-open, mirrors is_cme_holiday for future live use).
- pipeline/build_daily_features.py — compute_orb_range gains a scoped guard: `if orb_label == "NYSE_PREOPEN" and is_nyse_holiday(trading_day): return {high/low/size/volume: None}`. Routes NYSE holidays into the EXISTING empty-ORB path (line 232 all-None return). Downstream detect_break (:311) and compute_outcome (:371) already short-circuit on orb_high is None — NO new downstream handling needed (canonical empty-ORB contract reused). Guard is scoped to NYSE_PREOPEN ONLY — cannot affect any of the 12 existing sessions.
- tests/test_pipeline/test_market_calendar.py — new TestIsNyseHoliday class: NYSE-closed holidays (July4/Thanksgiving/MLK/GoodFriday/Christmas/weekends) return True; normal days False; CME-vs-NYSE divergence (July4: cme=False, nyse=True) asserted explicitly; strict=True raises beyond coverage, strict=False returns False.
- tests/test_pipeline/test_build_daily_features.py — assert compute_orb_range returns all-None for NYSE_PREOPEN on a known NYSE holiday, and builds normally for a non-NYSE_PREOPEN session on the same date (scoping proof).
- Reads: XNYS calendar (deterministic). Writes: NONE (no DB write, no migration). No schema change. NYSE_PREOPEN has no feature data yet (Stage 3), so ZERO current behavioral change to live/stored data.
- No allocator/config/deployment change. No write to validated_setups.

## Acceptance (all required before deleting this stage file)
- tests/test_pipeline/test_market_calendar.py + test_build_daily_features.py PASS — show output.
- python pipeline/check_drift.py PASSES.
- dead-code sweep: grep confirms is_nyse_holiday is wired (not orphaned) and the guard is scoped to NYSE_PREOPEN.
- self-review (line citations).

## NOT done by this stage (deferred to Stage 3)
- session_guard._SESSION_ORDER insertion + _WINDOW_FEATURES safe-after entries.
- ORB_LABELS subset _SESSION_ORDER parity drift check.
- run_rebuild_with_sync.sh MNQ — populate orb_outcomes + daily_features for NYSE_PREOPEN. IS/OOS day counts + DST-imbalance kill floor (N_EST>=30 AND N_EDT>=30).
- Stage 4: promote draft, run K=27 strict Chordia (t>=3.79, NO_THEORY_GRANT).
