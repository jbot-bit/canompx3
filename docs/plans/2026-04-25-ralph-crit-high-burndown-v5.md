# Ralph CRIT/HIGH Burndown Plan v5.2

**Status:** active (2026-04-25)
**Canonical checkpoint for this burndown.** Conversation memory supplements this file; the file is authoritative.

---

## Context

Ralph's Priority 1-4 auto-targeting was finding only LOW/MED severity for iterations 167-171. Priority 1 (unscanned CRIT/HIGH files) was largely exhausted after 251 prior files. Priority 2-4 was grinding thin-surface medium-centrality files. Meanwhile `HANDOFF.md` Next Steps and `docs/ralph-loop/deferred-findings.md` already listed known CRIT/HIGH items that nobody was burning down.

Iteration 172 validated the fix: a Priority 0 rule (commit `8ca4e1c6`) forces ralph to check the known-CRIT backlog before auto-targeting. On first run it caught a CRIT (B6 F-1 seed) and verified the pre-landed fix. See `ralph-ledger.json` iter 172, `consecutive_low_only` reset to zero, `last_high_finding_iter=172`.

This plan burns down the remaining known CRIT/HIGH work. **Success:** every CRITICAL item from `e02c529d` and `HANDOFF.md` is either ralph-verified as institutionally sound or has a fresh fix committed with stage doc plus tests green plus drift at full count.

---

## Institutional grounding

1. `.claude/rules/institutional-rigor.md` § 2 — After any fix, review the fix. The adversarial-audit gate (see `adversarial-audit-gate.md` once written in Pass two) formalizes this.
2. `.claude/rules/institutional-rigor.md` § 3 — Refactor when a pattern of bugs appears. The shutdown-helper extraction in Pass one is this rule in action.
3. `.claude/rules/institutional-rigor.md` § 6 — No silent failures. Every `except Exception` records, propagates, or loudly notifies.
4. `.claude/rules/integrity-guardian.md` § 3 — Fail-closed. Halt paths must not return success after swallowing an exception.
5. `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — change-point detection reference (not scope here).

---

## Backlog inventory

### Pre-landed fixes verified (iter 173)

| ID | Commit | Severity | Scope | Verified |
|----|--------|----------|-------|----------|
| F8 | `e02c529d` | CRITICAL | Orphan bracket cleanup halts instead of log.warning | iter 173 |
| R2 | `e02c529d` | CRITICAL | Telegram dispatch via to_thread when event loop running | iter 173 |
| F2 | `e02c529d` | HIGH | F-1 None equity at startup notifies | iter 173 |
| F5 | `e02c529d` | HIGH | HWM poll exception propagates as None to trigger 3-strike halt | iter 173 |
| F6 | `e02c529d` | MED | Journal unhealthy in demo now notifies | iter 173 |

### Fresh CRIT/HIGH (HANDOFF-deferred)

| ID | Severity | Status | Commit | Audit verdict |
|----|----------|--------|--------|---------------|
| F4 | CRITICAL | RESOLVED iter 174 | `87dffa38` | CONDITIONAL — C1 kill-switch race found in audit |
| R1 | CRITICAL | RESOLVED iter 175 | `6dafda10` | CONDITIONAL — S2, S3, S4 gaps flagged |
| R3 | HIGH | RESOLVED iter 176 | `64d0952d` | pending audit in iter 178 |
| C1 | CRITICAL (audit insert) | IN FLIGHT iter 177 | — | — |
| F7 | HIGH | queued | — | — |
| R4 | HIGH | queued | — | — |
| R5 | HIGH | queued | — | — |

### Open in deferred-findings.md

- **SR-L6** (MEDIUM): re-check trigger at N at or above 100 monitored trades. Not actionable by code.

### Out of scope (named)

- **O-SR** — change-point detection upgrade. Multi-stage, separate plan.
- **F4 spec complexity** — if F4 re-opened, may exceed single-iteration threshold; stage and defer.

---

## Iteration ledger

| Iter | Classification | Item | Commit | Verdict | Audit verdict |
|------|----------------|------|--------|---------|---------------|
| 173 | judgment audit-only | Verify `e02c529d` five fixes | `2f45a3e8` | ACCEPT | n/a (audit-only) |
| 174 | judgment fix | F4 bracket submit failure triggers emergency flatten | `87dffa38` | ACCEPT | CONDITIONAL (C1) |
| 174-m | mechanical | Ledger update | `28ee1a9b` | n/a | n/a |
| 175 | judgment fix | R1 wall-clock rollover task | `6dafda10` | ACCEPT | CONDITIONAL (S2/S3/S4) |
| 175-m | mechanical | Ledger update | `ac8e0169` | n/a | n/a |
| 176 | judgment fix | R3 reconnect ceiling plus stable-run reset | `64d0952d` | ACCEPT | pending iter 178 |
| 176-m | mechanical | Ledger update | `f40ac4dd` | n/a | n/a |
| 177 | judgment fix | C1 kill-switch event-loop race plus T1, T2, T4 tests | f8f993b7 | ACCEPT | PASS (iter 178) |
| 178 | judgment audit-only | Adversarial audit of iters 176 and 177 | da2c4dfb | ACCEPT | PASS — no CRIT/HIGH, 4 LOW all ACCEPTABLE |
| 179 | judgment fix | Hardening Pass one: shutdown helper plus kill-switch drift check | — | — | — |
| 180 | judgment docs | Hardening Pass two: durable plan file plus audit-gate rule | — | — | — |
| 181 | judgment fix | R4 live signals log rotation | — | — | — |
| 182 | judgment audit-only | Audit iter 181 | — | — | — |
| 183 | judgment fix | R5 heartbeat re-notify | — | — | — |
| 184 | judgment audit-only | Audit iter 183 | — | — | — |
| 185 | judgment fix | F7 fill poller PENDING timeout | — | — | — |
| 186 | judgment audit-only | Audit iter 185 | — | — | — |
| 187 | judgment fix | Hardening Pass three: magic-number rationale drift check | — | — | — |
| 188 | judgment fix | Silent-gap cleanup (S2, ledger semantics) | — | — | — |

---

## Hardening passes

### Pass one — structural (iter 179)

**Scope:** shutdown helper plus new drift check on kill-switch guard.

**Files touched.**
- `trading_app/live/session_orchestrator.py` — add a shutdown helper method near the shutdown block. Retrofit the four existing cancel callsites at lines near 2984.
- `pipeline/check_drift.py` — add drift check 114. Scans files under the live orchestrator folder for any method body that contains the broker-submission pattern and flags the method if the kill-switch guard word is absent from the same method body. Add to main run list. Bump expected total from 113 to 114.
- `tests/test_trading_app/test_session_orchestrator.py` — add three test cases for the helper: normal cancel, timeout path, already-completed no-op.
- `tests/test_pipeline/test_check_drift.py` — add positive and negative cases for check 114.

**Acceptance.** All tests pass. Drift 114 of 114. Pre-commit eight of eight. Stage doc closed.

### Pass two — meta (iter 180, can start immediately, docs only)

**Scope:** durable plan file plus adversarial-audit gate rule file.

**Files touched.**
- `docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md` — this file.
- `.claude/rules/adversarial-audit-gate.md` — new rule file describing the gate: trigger is any judgment-classified commit touching truth-layer folders, actor is the evidence-auditor subagent, artifact is a structured audit report with verdict plus critical issues plus tests missing, scope is the commit under review plus any untouched commit it depends on.
- `.claude/rules/institutional-rigor.md` — amend section two to cross-reference the new rule file.

**Acceptance.** Plan file present and status markers match current git log. Rule file referenced from at least one canonical doc. No code changes.

### Pass three — magic-number rationale (iter 187, after all HIGH items land and are audited)

**Scope:** drift check on unexplained numeric literals in live trading folders.

**Files touched.**
- `pipeline/check_drift.py` — add drift check 115. Scans files in `trading_app/live/` and `trading_app/risk/` for numeric literals at or above ten that lack a rationale comment in a five-line window above.
- The two target folders — sweep for existing violations and add rationale comments.
- `tests/test_pipeline/test_check_drift.py` — positive, negative, boundary cases.

**Acceptance.** Drift 115 of 115. Sweep complete, no open violations.

---

## Adversarial-audit gate (permanent)

After every judgment-classified commit that touches `trading_app/live/`, `trading_app/risk/`, or `pipeline/` truth-layer paths, dispatch the evidence-auditor subagent for an independent-context adversarial pass BEFORE the next phase dispatches. The gate enforces `institutional-rigor.md` § 2. Single-agent ralph iterations are necessary but not sufficient — the C1 kill-switch race (audit insert on iter 174) is the proof case.

Mechanical and ledger commits are exempt.

---

## Critical files

| Path | Role |
|------|------|
| `.claude/agents/ralph-loop.md` | Priority 0 rule lives here (lines 83-101 as of `8ca4e1c6`) |
| `HANDOFF.md` Next Steps plus Blockers | Source of Priority 0 items |
| `docs/ralph-loop/deferred-findings.md` | Structured ledger; every deferred item is a row |
| `docs/ralph-loop/ralph-ledger.json` | Per-iteration severity stats |
| `trading_app/live/session_orchestrator.py` | Touched by F4, R1, C1, F7, R3, R5, Pass one |
| `trading_app/live/execution_engine.py` | F4 bracket path |
| `trading_app/live/account_hwm_tracker.py` | F5 three-strike halt canonical source |
| `pipeline/dst.py` | R1 canonical 09:00 Brisbane source |
| `scripts/infra/telegram_feed.py` | R2 async-safe dispatch |
| `pipeline/check_drift.py` | Pass one adds check 114, Pass three adds check 115 |

---

## Do-not-touch areas

| Path or Object | Reason |
|----------------|--------|
| SR-L6 threshold or monitor code | Gate is N at or above 100 monitored trades, not code. |
| O-SR change-point upgrade | Multi-stage, needs its own plan plus literature re-grounding. |
| Parallel Codex dirty working-tree files | Six recurring CRLF-drift files pinned by gitattributes; leave alone until Codex commits. |
| Any change to Priority 0 rule without stage-gate | Prompt is canonical; edits go through stage-gate protocol. |
| `_check_trading_day_rollover` idempotency guard | Audit-verified correct; do not refactor. |
| `_fire_kill_switch` persistence path | Audit-verified correct; do not make conditional. |
| `compute_trading_day_utc_range` usage in wall-clock task | Audit-verified correct; do not inline hour constant. |

---

## Verification summary

| Gate | Proof |
|------|-------|
| Iter 173 | Audit-only; ACCEPT; five findings resolved |
| Iter 174 | F4 fix landed; 4 of 4 mutation-proof tests PASS; drift 107 of 107; audit CONDITIONAL (C1) |
| Iter 175 | R1 fix landed; mutation-proof tests PASS; drift 107 of 107; audit CONDITIONAL (S2/S3/S4) |
| Iter 176 | R3 fix landed; 4 tests PASS; drift 107 of 107; audit pending iter 178 |
| Iter 177 | C1 fix plus T1/T2/T4 tests; 166/166; drift 107/107; commit f8f993b7 |
| Iter 178 | Adversarial audit iters 176+177; PASS; 4 LOW ACCEPTABLE; commit da2c4dfb |
| Iter 179 | Pass one complete; drift 114 of 114 |
| Iter 180 | Pass two complete; plan file and rule file landed |
| Iter 181 to 186 | R4, audit, R5, audit, F7, audit |
| Iter 187 | Pass three complete; drift 115 of 115 |
| Iter 188 | Silent-gap cleanup |

**Rollback.** Any iteration commit is revert-able. Plan file is a checkpoint, not a lockfile.

---

## Checkpoint log

- v1 through v4 — prior plan (signal-only smoke test), COMPLETE
- v5 — overwritten to cover ralph CRIT/HIGH burndown, 2026-04-25 01:55 Brisbane
- v5.1 — audit-gate amendment after iter 174 C1 finding, 2026-04-25 11:20 Brisbane
- v5.2 — hardening passes 179, 180, 187 added after user directive to harden and future-proof, 2026-04-25 11:45 Brisbane
- **Pre-plan-v5 iters landed this session:** 170 (break_ts), 171 (RHO_REJECT_THRESHOLD), 172 (B6 verify — CRIT), plus adjacent commits a6494e76 (test mock fix) and 8ca4e1c6 (Priority 0 rule)
- P1 (iter 173): COMPLETE, ACCEPT, commit `2f45a3e8`
- P2 (iter 174): COMPLETE fix, audit CONDITIONAL, commit `87dffa38`
- P3 (iter 175): COMPLETE fix, audit CONDITIONAL, commit `6dafda10`
- Iter 176 (R3): COMPLETE, audit pending, commit `64d0952d`
- Iter 177 (C1): IN FLIGHT
- Remaining: audit 178, hardening 179 and 180, HIGH tier 181 to 186, hardening 187, cleanup 188
