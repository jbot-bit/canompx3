---
slug: hwm-stage4-inactivity-monitor
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 4
of: 4
created: 2026-04-26
updated: 2026-04-26
parent_design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3 § 7)
prior_stage_audit: commit 1de0f17f Stage 3 audit-gate verdict CONDITIONAL → both findings closed at cc20f52c (C-1 Scenario 6 added; TM-2 path-in-warning assertions added; C-2 HANDOFF deferred to this stage as designed)
audit_gate: `.claude/rules/adversarial-audit-gate.md` — fires after this judgment commit on the new check function
task: HWM Stage 4 — pre-session `check_topstep_inactivity_window` delegating to Stage 2's `state_file_age_days` helper. Warn ≥25 days, block ≥30 days. Three deferred-findings rows (SILENT-6, UNSUPPORTED-5, UNSUPPORTED-6). HANDOFF closure across Stages 1-4 (closes Stage 3 audit-gate finding C-2). 6+ mutation-proof unit tests.
---

# Stage: hwm-stage4-inactivity-monitor

mode: IMPLEMENTATION
date: 2026-04-26

scope_lock:
  - trading_app/pre_session_check.py
  - tests/test_trading_app/test_pre_session_check.py
  - docs/ralph-loop/deferred-findings.md
  - HANDOFF.md
  - docs/runtime/stages/hwm-stage4-inactivity-monitor.md

## Why

Stage 2 shipped `state_file_age_days(path: Path) -> float | None` as the single source of truth for state-file age (with mtime-fallback for null/missing-timestamp cases per audit-gate SG1 fix). Stage 2's load-time `_load_state` gate fires only when the tracker is actually constructed — that's the live-trading path. But for signal-only sessions, manual operator pauses, and any session that doesn't construct a tracker, an aged state file goes undetected by Stage 2's gate.

Stage 4 adds a pre-session pre-flight check that surfaces the inactivity boundary regardless of whether the tracker is constructed. Warns at ≥25 days (operator action buffer); blocks at ≥30 days (TopStep XFA inactivity-closure boundary borrowed by analogy).

Three lower-severity findings from prior audit cycles (SILENT-6, UNSUPPORTED-5, UNSUPPORTED-6) get explicit deferred-findings rows so they are tracked, not lost. HANDOFF.md gets updated to mark all closed audit findings across Stages 1-4 — closes Stage 3 audit-gate finding C-2.

Grounding (operational hardening + cited TopStep parameter):

- `docs/research-input/topstep/topstep_xfa_parameters.txt:351` — "If there is no trading activity (no trades entered) on your Express Funded Account for more than 30 days, it may be subject to closure due to inactivity." (verbatim verified 2026-04-26)
- The 30-day figure is borrowed by ANALOGY for state-file age (the TopStep rule fires on no broker trades; this gate fires on no bot equity polls — semantically distinct, same numeric value). Same honesty disclaimer pattern as Stage 2.
- `institutional-rigor.md` § 4 (canonical sources, never re-encode) — delegates to `state_file_age_days` rather than computing age inline
- `institutional-rigor.md` § 6 (no silent failures) — log.warning on missing/unreadable files
- `integrity-guardian.md` § 2 (canonical sources) — single age-computation surface
- `integrity-guardian.md` § 3 (fail-closed) — ≥30 days blocks

NOT applicable: `docs/institutional/literature/`. Stage 4 is operational hardening, no statistical claim. Confirmed during 2026-04-26 grounding pass.

## Pre-execution audit findings — improvements over design v3 § 7

1. **Boundary direction explicit (CONSISTENCY IMPROVEMENT):** TopStep's text says "more than 30 days" (strict `>`). Stage 2's load gate uses `>= 30` for fail-closed margin (one-day-on-the-boundary still raises). Stage 4 uses `>= 30` to match Stage 2's convention. Both gates fire on the same day; the figure is borrowed by analogy anyway. Documented inline.

2. **Function returns `tuple[bool, str]` like sibling pre-session checks (CONSISTENCY).** All other `check_*` functions in `pre_session_check.py` return `(bool, str)`. Stage 4 follows the same contract. The "warning verdict (still OK to proceed)" from design v3 AC 2 maps to `(True, "WARNING: …")` — same pattern as `check_daily_equity` lines 184-185.

3. **Granular handling of corrupt-state files via `state_file_age_days` semantics (NO RE-ENCODING).** `state_file_age_days` already handles corrupt JSON via mtime fallback (Stage 2 audit-gate SG1 fix). When mtime fallback also fails (very stale unreadable file), it returns `None`. Stage 4 treats `None` as a state-file integrity failure → BLOCKED message — fail-closed. CORRUPT-named files are skipped (consistent with the rest of `pre_session_check.py`).

## Behavior changes

### B1. New `check_topstep_inactivity_window` function

```python
def check_topstep_inactivity_window() -> tuple[bool, str]:
    """Pre-session pre-flight: surface TopStep XFA inactivity-closure boundary.

    For each non-CORRUPT account_hwm_*.json in STATE_DIR, compute age in days
    via the canonical account_hwm_tracker.state_file_age_days helper (single
    source of truth — never reimplement the age computation; institutional-
    rigor.md § 4).

    Verdicts:
      - <25 days: OK
      - >=25 and <30 days: WARNING (still OK to proceed; operator action buffer)
      - >=30 days: BLOCKED (fail-closed per integrity-guardian.md § 3)
      - state_file_age_days returns None: BLOCKED (fail-closed; granular reason
        captured by state_file_age_days's own logging)

    @canonical-source: docs/research-input/topstep/topstep_xfa_parameters.txt:351
    @verbatim: "If there is no trading activity (no trades entered) on your
                Express Funded Account for more than 30 days, it may be subject
                to closure due to inactivity."
    @audit-finding: HWM persistence integrity audit 2026-04-25 (UNSUPPORTED-7)
                    — closed by Stage 4 of design v3.

    Honesty disclaimer: the 30-day figure is borrowed by ANALOGY. TopStep's
    rule fires on no broker trades. This gate fires on no bot equity polls.
    The semantics differ; the numeric value is shared by convention. Operator
    may legitimately trade manually with no bot — in that case the 30-day
    block is a false positive that the operator must resolve via state-file
    archive (.STALE_<YYYYMMDD>.json rename) or delete.
    """
```

Implementation pattern:

```python
from trading_app.account_hwm_tracker import state_file_age_days

hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
hwm_files = [f for f in hwm_files if "CORRUPT" not in f.name]
if not hwm_files:
    return True, "No account state files (first session — inactivity gate not applicable)"

results = []
any_block = False
for f in hwm_files:
    age = state_file_age_days(f)
    if age is None:
        any_block = True
        results.append(f"BLOCKED {f.name}: state file unreadable (cannot compute age)")
        continue
    age_int = int(age)
    if age >= 30.0:
        any_block = True
        results.append(
            f"BLOCKED {f.name}: {age_int}d old >= 30d inactivity boundary "
            f"(archive or delete state file to clear)"
        )
    elif age >= 25.0:
        # UNGROUNDED — operational buffer.
        # Rationale: 5-day buffer between WARNING and BLOCK gives the operator
        # advance notice to either resume bot trading, manually rotate the
        # state file, or accept the upcoming block. No literature citation.
        results.append(f"WARN {f.name}: {age_int}d old (block at 30d)")
    else:
        results.append(f"OK {f.name}: {age_int}d old")

msg = " | ".join(results)
return (not any_block), f"INACTIVITY: {msg}"
```

### B2. Wire into `run_checks`

Append the inactivity check to `run_checks` near the existing HWM checks (line ~573 area):

```python
ok, msg = check_topstep_inactivity_window()
results.append(("TopStep inactivity window", ok, msg))
```

### B3. Three deferred-findings rows in `docs/ralph-loop/deferred-findings.md`

Three new rows under "Open Findings" — preserves the principle that nothing gets lost:

| ID | Iter | Severity | Target | Description | Deferred Reason |
|---|---|---|---|---|---|
| HWM-SIL6 | hwm-stage4 | LOW | docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md | SILENT-6: v5.2 plan references S2/S3/S4 silent gaps that have no entry in the deferred-findings ledger. The references are documented in the Ralph plan but lack ledger rows for cross-cycle tracking. | Out of HWM-hardening scope. Escalate to Ralph at next-audit pass to add the missing rows OR clarify that S2/S3/S4 are pre-Ralph-burndown and outside ledger scope. |
| HWM-UNS5 | hwm-stage4 | LOW | docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md (R3 numeric values) | UNSUPPORTED-5: R3 numeric values in the Ralph plan are asserted without log-based measurement evidence. Acceptance criteria reference values that should be measured-and-cited rather than declared. | Escalate to Ralph for measurement during the next R3-touching iteration. |
| HWM-UNS6 | hwm-stage4 | LOW | docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md (R3 numeric values) | UNSUPPORTED-6: companion to UNSUPPORTED-5 — additional R3 numeric values without measurement evidence. | Same disposition as HWM-UNS5. |

### B4. HANDOFF.md update — close audit findings across Stages 1-4

Update the HWM-hardening section of HANDOFF.md:

- Mark CRITICAL-1, CRITICAL-2, CRITICAL-3 as **CLOSED** with commit hashes.
- Mark SILENT-1, 2, 3, 4, 5, 7 as **CLOSED** with commit hashes.
- Mark UNSUPPORTED-1, 2, 3, 4, 7 as **CLOSED** with commit hashes.
- List SILENT-6, UNSUPPORTED-5, UNSUPPORTED-6 as **DEFERRED** referencing `docs/ralph-loop/deferred-findings.md` rows HWM-SIL6, HWM-UNS5, HWM-UNS6.
- Mark Stage 1 / Stage 2 / Stage 3 / Stage 4 status: **LANDED** with commit hashes (5b172a44 / 68c63482 / 45720109 / df00589b for Stage 1; 1f29009b / e67f46f6 / c5be3453 for Stage 2; 1de0f17f / cc20f52c for Stage 3; this stage's commit hash for Stage 4).
- Reference `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` v3 as the parent design doc.

This closes Stage 3 audit-gate finding C-2 (HANDOFF staleness — "navigation trap for next reader").

## Acceptance criteria

1. `check_topstep_inactivity_window` returns `(True, ...)` for state files under 25 days old.
2. Returns `(True, "INACTIVITY: ... | WARN <name>: Nd old (block at 30d) | ...")` for state files at 25-29 days old.
3. Returns `(False, "INACTIVITY: ... | BLOCKED <name>: ...")` for state files at 30+ days old.
4. Returns `(False, "BLOCKED <name>: state file unreadable (cannot compute age)")` when `state_file_age_days` returns `None` (fail-closed — granular reason captured by state_file_age_days's own logging).
5. Returns `(True, "No account state files ...")` when STATE_DIR has no account_hwm_*.json files.
6. Skips files with "CORRUPT" in the name (consistent with the rest of pre_session_check.py).
7. Function carries the `@canonical-source`, `@verbatim`, and `@audit-finding` annotation block in its docstring.
8. The 25-day threshold carries an `UNGROUNDED` + `Rationale:` comment.
9. Function delegates age computation to `account_hwm_tracker.state_file_age_days` (greppable: no inline `mtime`, `last_equity_timestamp` parsing, or `datetime` arithmetic in `check_topstep_inactivity_window`).
10. `run_checks` calls `check_topstep_inactivity_window` and appends the result.
11. Three deferred-findings rows added to `docs/ralph-loop/deferred-findings.md` with the IDs HWM-SIL6, HWM-UNS5, HWM-UNS6.
12. HANDOFF.md updated with Stage 1-4 closure, audit-finding closure, and deferred-finding references.
13. Boundary tests: 25 days minus 1 second is OK; 25 days plus 1 second warns; 30 days minus 1 second warns; 30 days plus 1 second blocks. Use the established `patch("trading_app.account_hwm_tracker.datetime")` pattern.
14. Drift check passes at full count.
15. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

## Tests required (mutation-proof)

| Name | What it pins |
|---|---|
| `test_inactivity_check_under_25_days_returns_ok` | Healthy band. |
| `test_inactivity_check_25_days_plus_1s_warns_continues` | Lower-boundary direction (`>= 25`). Mutation: `> 25` flips. |
| `test_inactivity_check_25_days_minus_1s_silent_ok` | Boundary direction reverse. |
| `test_inactivity_check_29_days_warns` | Mid-band warn. |
| `test_inactivity_check_30_days_plus_1s_blocks` | Upper-boundary direction (`>= 30`). Mutation: `> 30` flips. |
| `test_inactivity_check_30_days_minus_1s_warns_does_not_block` | Boundary direction reverse. |
| `test_inactivity_check_state_unreadable_blocks` | None-return from helper → BLOCKED with "state file unreadable" message. Mutation: returning OK on None flips. |
| `test_inactivity_check_no_files_returns_ok` | Empty STATE_DIR pass. |
| `test_inactivity_check_skips_corrupt_named_files` | CORRUPT filter preserved. |
| `test_inactivity_check_carries_canonical_source_annotation` | Greppable docstring contains `@canonical-source`, `@verbatim`, `@audit-finding`, AND `topstep_xfa_parameters.txt:351`. Mutation: dropping any token flips. |
| `test_inactivity_check_25_day_buffer_has_ungrounded_rationale_comment` | Source-file scan: `UNGROUNDED` and `Rationale:` tokens within ±5 lines of the `25.0` literal in the function source. |
| `test_inactivity_check_delegates_to_state_file_age_days` | Mutation guard via `unittest.mock.patch` — `state_file_age_days` is the call site. Re-introducing inline mtime/timestamp parsing flips this test. |

## Blast Radius

- `trading_app/pre_session_check.py` — adds 1 new function (~50 lines including docstring and annotation block) plus 1 line in `run_checks` to wire it in.
- `tests/test_trading_app/test_pre_session_check.py` — 12 new tests in a `TestStage4InactivityWindow` class.
- `docs/ralph-loop/deferred-findings.md` — 3 new rows.
- `HANDOFF.md` — closure update for HWM Stages 1-4.

No tracker changes. No orchestrator changes. No DB schema. No external API change.

State files on disk: NOT modified. The new check is read-only on the same files Stages 1-3 already touch. Existing files continue to load correctly through the existing tracker path.

The 30-day BLOCK is a NEW fail-closed gate — pre-existing state file `account_hwm_20092334.json` (~20 days old at design date 2026-04-26) is in the WARN band but well clear of the BLOCK boundary. The file would currently emit a WARN line if the check ran today, not a block.

## Risk flags

- **Risk 1 (LOW):** existing pre-session test snapshots `run_checks` output; my new check appends a row. Mitigation: AC 12 explicitly permits these tests to be updated. Verify zero such snapshots exist during execution; document any updates in the commit body.
- **Risk 2 (LOW):** Stage 2's `state_file_age_days` mtime-fallback may report different age than the in-file timestamp on a state file whose `last_equity_timestamp` was rewritten by an out-of-band operator action (manual JSON edit). This is desired behavior (mtime tracks last-attempted-write — operationally fresher than in-file claim) but operators should know. Mitigation: documented in the function docstring.
- **Risk 3 (LOW):** new BLOCK on 30-day file could surprise an operator running a long manual-trading window. Mitigation: BLOCK message includes the resolution recipe ("archive or delete state file to clear"). Same recipe pattern as Stage 2's load-time raise.

## Rollback

`git revert` of the single judgment commit. `check_topstep_inactivity_window` and its `run_checks` registration disappear. Deferred-findings rows can be removed independently if Ralph closes them in his own iteration. HANDOFF.md changes are independent — keep them, restore them, or revert separately as needed.

## Audit-gate hand-off

After this commit lands, dispatch `evidence-auditor` per `.claude/rules/adversarial-audit-gate.md`. Auditor scope:

- This commit only.
- Verify all 15 acceptance criteria with execution evidence.
- Verify no regression in existing pre-session tests.
- Verify the function genuinely delegates to `state_file_age_days` (test exists; auditor should confirm test asserts call-count, not just return).
- Verify the 30-day BLOCK boundary direction (`>= 30` not `> 30`) matches Stage 2's convention.
- Verify the 25-day buffer's `UNGROUNDED` + `Rationale:` comment passes `pipeline/check_drift.py` Pass-Three rationale-discipline scan if/when that drift check enforces on `pre_session_check.py` (currently scopes `trading_app/live/` only — note in audit report).
- Independent trace: a state file with mtime fresher than the in-file `last_equity_timestamp` (operator manually edited the JSON to look older) — does the gate use mtime or the in-file timestamp? state_file_age_days's resolution order says: try in-file timestamp first; on missing/invalid, fall back to mtime. So a manually-edited "ancient" timestamp WOULD make the file appear stale. This is desired (mtime is the strictly-fresher lower bound; the in-file value is what the operator put there). Confirm.
- Verify no inline age-computation re-encoding in pre_session_check.py.
- Verify HANDOFF.md closure update is internally consistent (every "CLOSED" claim has a commit hash, every "DEFERRED" has a deferred-findings row).

This is the FINAL stage. After audit returns PASS, the HWM persistence integrity hardening design v3 is complete.
