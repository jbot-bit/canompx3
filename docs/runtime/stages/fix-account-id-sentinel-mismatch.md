---
task: Fix --account-id sentinel mismatch (live-start crash on Start Live button)
mode: IMPLEMENTATION
updated: 2026-05-16
scope_lock:
  - scripts/run_live_session.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/multi_runner.py
  - tests/test_scripts/test_run_live_session_preflight.py
blast_radius: "scripts/run_live_session.py CLI default change (0->None); session_orchestrator.py broadens auto-discover sentinel (==0 -> is None or ==0, backwards-compat); multi_runner.py type annotation widens (int -> int|None); one new test exercises copies>1 + account_id=None path. Reads: none. Writes: none. Capital-class live-start path."
acceptance:
  - python pipeline/check_drift.py exits 0
  - New test exercises copies>1 + account_id=None auto-discover path
  - Existing tests in test_session_orchestrator.py, test_run_live_session_preflight.py pass unchanged
  - grep "account_id == 0" trading_app/ scripts/ shows no orphan call sites (all paths handle both None and 0)
  - Manual smoke: dashboard "Start Live" on topstep_50k_mnq_auto reaches SessionOrchestrator.__init__ without RuntimeError from _select_primary_and_shadow_accounts
---

# Fix: --account-id Sentinel Mismatch

## The bug

`logs/session.log` 2026-05-15 09:04 confirms live-start crashes:

```
File "scripts/run_live_session.py", line 334, in _select_primary_and_shadow_accounts
RuntimeError: --account-id 0 is not in the broker's discovered accounts [21944866].
```

Triggered when user clicks "Start Live" in dashboard. Dashboard builds the command
WITHOUT passing `--account-id` (see `bot_dashboard.py:2374-2388`), so `args.account_id`
defaults to `0` per argparse declaration at `run_live_session.py:423-428`.

## Root cause (verified)

Two contradictory sentinel conventions for "auto-discover account_id":

- **`scripts/run_live_session.py:423-428`** — CLI `--account-id default=0`, help text
  reads "(default: auto-discover from API)".
- **`trading_app/live/session_orchestrator.py:542-543`** — `if account_id == 0:
  account_id = contracts.resolve_account_id()`. Treats `0` as auto-discover. ✅
- **`scripts/run_live_session.py:309-353` (`_select_primary_and_shadow_accounts`)** —
  treats any non-None integer as "user requested this specific ID" and rejects `0`
  because `0` is never in the broker's account list. ❌ This is the bug.

Bridging code at `run_live_session.py:688-691` (copy-trading branch) passes
`args.account_id` (which is `0`) through to `_select_primary_and_shadow_accounts`,
hitting the rejection path on `copies > 1` profiles. Active profile
`topstep_50k_mnq_auto` has `copies=2`, so this fires every time on that profile.

## Fix (design approved 2026-05-16, NOT YET IMPLEMENTED)

**Canonical convention going forward: `None` = auto-discover. `0` is legacy-tolerated.**

### Changes

1. **`scripts/run_live_session.py:423-428`** — CLI:
   - `default=0` → `default=None`
   - `type=int` → keep `type=int` (argparse coerces non-None to int)
   - Update help text to reflect None semantics (still "default: auto-discover from API")

2. **`trading_app/live/session_orchestrator.py:542`** — broaden the tolerance:
   - `if account_id == 0:` → `if account_id is None or account_id == 0:`
   - Keeps backwards compatibility for any caller still passing 0
   - All 22+ test fixtures construct with default `account_id=0` — they continue to work

3. **`trading_app/live/multi_runner.py:41`** — type only:
   - `account_id: int = 0` → `account_id: int | None = 0`
   - Default stays `0` for test backwards compat
   - Docstring: document that `None` and `0` both mean auto-discover (orchestrator
     handles both per § 2 above)

4. **Test** — `tests/test_scripts/test_run_live_session_preflight.py` (or new file
   `tests/test_scripts/test_run_live_session_account_id_sentinel.py`):
   - `test_select_primary_and_shadow_accounts_with_none_account_id_auto_discovers`
   - `test_argparse_default_account_id_is_none` — confirms CLI default
   - Optional: `test_orchestrator_account_id_zero_or_none_both_auto_discover`

### Why NOT change `_select_primary_and_shadow_accounts` to accept `0`

That would re-encode the "0 is auto" convention into a third file. Per
`.claude/rules/integrity-guardian.md` § 2, we want ONE canonical convention. `None` is
the Pythonic absent-value; making it the canonical sentinel and tolerating `0` in the
two places where test fixtures already use `0` is the cleaner refactor.

### Deliberately deferred (separate task)

- **A.6.5 Preflight Gap.** Preflight passes 6/6 even when live-start crashes because
  preflight runs in `signal_only` mode (no broker order-router resolution). Fix scope:
  add a preflight check that simulates the copy-trading account-resolution path. Open
  follow-up after this sentinel fix lands. Tracked in
  `docs/runtime/session-checkpoint-2026-05-14-go-live.md` § A.6.5.

### Self-review checklist (run BEFORE marking done, per
`.claude/rules/institutional-rigor.md` § 1)

- [ ] `python pipeline/check_drift.py` PASS
- [ ] `python -m pytest tests/test_trading_app/test_session_orchestrator.py
       tests/test_scripts/test_run_live_session_preflight.py -v` all green
- [ ] `grep -rn "account_id == 0" trading_app/ scripts/` — every match still handles
      both None and 0
- [ ] `grep -rn "default=0" scripts/run_live_session.py` — confirm only the changed
      flag is affected
- [ ] No `_ = unused_var` silencing introduced
- [ ] No re-encoded `resolve_account_id` logic
- [ ] Test exercises BOTH the bug (rejection of `0` in copy-trading) AND the fix
      (auto-discover with `None`)

### Adversarial audit (per `.claude/rules/adversarial-audit-gate.md`)

This is a `[judgment]` commit touching `trading_app/live/` with HIGH severity
(capital-class live-start blocker). After implementation, **dispatch `evidence-auditor`
in a separate context** to review the fix BEFORE next phase. Audit checklist:
- Sentinel-convention coherence across all 3 production files.
- Any code path where `0` or `None` is passed to a broker-API call that requires a real
  account ID.
- Test coverage of the rejection-of-bad-real-ID path (passing `--account-id 999999`
  must still raise).

## Context for the next session (pick up here)

When you resume:

1. Read this file first.
2. Run `git status --short` — should show:
   - `docs/runtime/stages/fix-account-id-sentinel-mismatch.md` (this file, uncommitted)
   - `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` (Stage 2 parked,
     uncommitted)
3. Switch this stage's `mode:` field from DESIGN → IMPLEMENTATION before editing
   any production file (per `.claude/rules/stage-gate-protocol.md`).
4. Implement the 3-file change + 1 test per § Fix above.
5. Run self-review checklist.
6. Commit (`[judgment]` tag, HIGH severity body).
7. Dispatch adversarial audit.
8. After audit PASS: revisit A.6.5 preflight-gap as a separate task.

## What was already done this session (2026-05-16)

- Stage 2 NQ-mini wiring PARKED. See
  `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` for plumbing-gap
  finding and three implementation options for future-me.
- Live-path audit completed. THIS bug found in logs/session.log 2026-05-15.
- Drift check: 133/133 PASS, 20 advisory. Clean.
- Active profile: only `topstep_50k_mnq_auto`. No MGC profile. MES profile inactive.
- `data/bot_state.json` contains MagicMock fixture leakage — separate concern, deferred.
- Plan/design approved by user. Implementation interrupted to /clear context first.
