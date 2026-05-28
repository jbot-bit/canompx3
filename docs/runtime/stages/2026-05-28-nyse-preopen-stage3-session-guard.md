---
task: |
  Lane B Stage 3 (part A — guard fix) — close the session_guard._SESSION_ORDER
  silent look-ahead-mask gap for NYSE_PREOPEN, surfaced by the 2026-05-28
  adversarial-audit gate (docs/audit/2026-05-28-nyse-preopen-dst-session-definition.md
  § Highest-priority fix).

  THE HAZARD (audit § Silent gaps): _SESSION_ORDER builds the regex
  _SESSION_COL_RE (session_guard.py:110). A session ABSENT from _SESSION_ORDER
  is invisible to that regex, so its orb_NYSE_PREOPEN_* feature columns fall
  through is_feature_safe's "Unknown column -> fail CLOSED" branch (:158-159)
  and get MASKED for EVERY target session — a silent look-ahead-mask exclusion,
  NOT just the ValueError the prior memory framing implied. Inert today (no
  caller routes NYSE_PREOPEN through _session_index yet), but WILL silently
  corrupt feature visibility the moment Stage-3 part-B builds features.

  THE FIX: insert "NYSE_PREOPEN" into _SESSION_ORDER between US_DATA_830 and
  NYSE_OPEN. Chronological position VERIFIED by resolver execution: US_DATA_830
  22:30/23:30, NYSE_PREOPEN 23:00/00:00, NYSE_OPEN 23:30/00:30 Brisbane
  summer/winter. Position is load-bearing for look-ahead correctness
  (backtesting-methodology.md RULE 1.2): NYSE_PREOPEN fires AFTER the
  session_london_* window closes (23:00) and DURING the session_ny_* window
  (23:00-02:00), so session_ny_* must remain look-ahead-UNSAFE for NYSE_PREOPEN
  — the chronological index enforces this automatically once positioned right.

  HARDENING (harden-as-you-go, real n>=1 gap the audit confirmed): add a
  fail-closed drift check asserting every ORB_LABELS session is present in
  _SESSION_ORDER. _SESSION_ORDER is now the standalone canonical chronological
  source (ML subsystem removed V3 sprint; check_session_guard_sync is a no-op,
  the ml/config.py pointer at :36 is dead) — so a half-registered session
  (in ORB_LABELS but absent from _SESSION_ORDER) is exactly the class of gap
  that bit us here. The check makes it mechanically impossible to repeat.
mode: IMPLEMENTATION
updated: 2026-05-28T00:00Z
agent: claude (opus 4.7)
supersedes: none

scope_lock:
  - pipeline/session_guard.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_session_guard.py
  - tests/test_pipeline/test_check_drift.py

## Blast Radius
- pipeline/session_guard.py — inserts "NYSE_PREOPEN" into _SESSION_ORDER (line 38) between US_DATA_830 and NYSE_OPEN. _SESSION_ORDER feeds _SESSION_COL_RE regex (:110) + every look-ahead index comparison (_session_index, get_prior_sessions, is_feature_safe). Insertion is ADDITIVE — it can only make a currently-mask-everywhere session correctly visible; it cannot un-mask any of the 12 existing sessions (their relative order is unchanged, NYSE_PREOPEN slots between two adjacent existing entries). Also fix the stale source-of-truth comment at :36 (ml/config.py pointer is dead).
- pipeline/check_drift.py — NEW fail-closed check check_session_order_covers_orb_labels: assert set(ORB_LABELS) subset of set(_SESSION_ORDER). Mirrors check_dow_classification_complete (#new) + check_orb_labels_session_catalog_sync (#32) pattern. Register in CHECKS tuple.
- tests/test_pipeline/test_session_guard.py — assert NYSE_PREOPEN in _SESSION_ORDER at the correct index (between US_DATA_830 and NYSE_OPEN); assert orb_NYSE_PREOPEN_* columns now MATCH _SESSION_COL_RE; assert is_feature_safe correctness: session_ny_* is look-ahead-UNSAFE for NYSE_PREOPEN (NY window not closed), session_london_* is SAFE (london closed before NYSE_PREOPEN), orb_NYSE_PREOPEN_size is safe for NYSE_OPEN (earlier->later) but orb_NYSE_OPEN_size is UNSAFE for NYSE_PREOPEN (later->earlier look-ahead).
- tests/test_pipeline/test_check_drift.py — known-violation injection for the new parity check (Rule 11).
- Reads: none. Writes: none (NO DB write, NO migration). NYSE_PREOPEN has no feature data yet, so ZERO current behavioral change to live/stored data.

## Acceptance (all required before deleting this stage file)
- tests/test_pipeline/test_session_guard.py + test_check_drift.py PASS — show output.
- python pipeline/check_drift.py PASSES (new parity check + existing checks green).
- dead-code sweep.
- self-review (line citations).
- THEN independent code review (user mandate — downside of getting it wrong is big) before declaring Stage 3 done.

## NOT done by this stage (part B, deferred)
- run_rebuild_with_sync.sh MNQ — populate orb_outcomes + daily_features for NYSE_PREOPEN. IS/OOS day counts; DST-imbalance kill floor (N_EST>=30 AND N_EDT>=30); the draft's N=27 budget. This is a DATA-WRITE operation (separate stage, separate review — touches gold.db).
- Stage 4: promote draft, run K=27 strict Chordia (t>=3.79, NO_THEORY_GRANT).
