---
task: Close `chordia_audit_unlock_pass_chordia_strategies` action-queue item with documented dispositions for the 2 truly remaining names.
mode: IMPLEMENTATION
slug: chordia-queue-closeout
created: 2026-05-03
worktree: C:/Users/joshd/canompx3-chordia-closeout
base: origin/main @ 702e175c
scope_lock:
  - docs/audit/hypotheses/2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.yaml
  - docs/audit/results/2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.md
  - docs/audit/results/2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.csv
  - docs/runtime/chordia_audit_log.yaml
  - docs/runtime/action-queue.yaml
  - docs/runtime/stages/chordia-queue-closeout.md
---

## Blast Radius

- `docs/audit/hypotheses/2026-05-03-*.yaml` — NEW pre-reg file, zero callers.
- `docs/audit/results/2026-05-03-*.{md,csv}` — NEW result artifacts written by `research/chordia_strict_unlock_v1.py`. Read by audit ledger only.
- `docs/runtime/chordia_audit_log.yaml` — APPEND one `audits:` row + one `notes:` block. Consumed by `trading_app/lane_allocator.py::apply_chordia_gate` and drift check #134. Idempotent if entry already exists at that strategy_id (allocator picks last entry).
- `docs/runtime/action-queue.yaml` — flip status of `chordia_audit_unlock_pass_chordia_strategies` from `open` to `closed`; update `next_action` and `last_verified_at`. Read by `/orient`, `/next`, action-queue tooling.
- Reads: `gold.db` (read-only via `chordia_strict_unlock_v1.py`); existing pre-reg templates from 2026-05-02.
- Writes: 4 new files + 2 file appends. No production code changes. No schema changes. No DB writes.

## Purpose

Action queue item `chordia_audit_unlock_pass_chordia_strategies` (P1, open since 2026-05-01) blocks "queue is clean" status. 2026-05-02 work landed 6 audit rows for the original 8 PASS_CHORDIA-without-audit names. Two names remain unresolved:

1. `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` — unaudited, runs through canonical strict-unlock runner.
2. `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075` — non-auditable through this runner by design (runner refuses non-default stop_multiplier per `chordia_strict_unlock_v1.py` lines 105-120). Requires `outcome_builder` rebuild at 0.75x stop, out of scope for this closeout. Document as a non-auditable note in the audit log so the queue exit-criteria (b) clause is satisfied.

Cleared queue → 100% attention available for Operational Track (Rithmic / AMP / EdgeClear) without nagging open items.

## Stages

### Stage 1 — Pre-reg YAML (this stage_lock file: `docs/audit/hypotheses/2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.yaml`)

Copy `docs/audit/hypotheses/2026-05-02-mnq-cmepreclose-xmesatr60-chordia-unlock-v1.yaml` template, change session from `CME_PRECLOSE` → `COMEX_SETTLE`, update slug, lock metadata.date to 2026-05-03.

Acceptance: file exists, parses as YAML, `scope.strategy_id == "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60"`, `filter_grounding_status.verdict == "UNSUPPORTED"` (per 2026-05-02 feasibility scan), `expected_trial_count == 1`, no theory grant claimed.

### Stage 2 — Run audit

Execute:
```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs/audit/hypotheses/2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.yaml
```

Acceptance: stdout shows `Verdict: <PASS_CHORDIA|PARK|FAIL_STRICT_CHORDIA|...>`, exit code 0. Result MD + CSV written.

### Stage 3 — Append audit log row + non-auditable note

`docs/runtime/chordia_audit_log.yaml`:
- Append one `audits:` entry with verdict from runner, `audit_date: 2026-05-03`, t_stat / threshold / sample_size from result MD.
- Add a top-level `notes:` block (or append to existing if any) documenting MES Tier-C non-auditability with rationale citing the runner refusal mechanism.

Acceptance: YAML still parses (`yaml.safe_load`); drift check #134 still passes.

### Stage 4 — Close action queue item

Flip `chordia_audit_unlock_pass_chordia_strategies.status` to `closed`. Update `next_action` to one-line summary + close-out date. Update `last_verified_at: 2026-05-03`. Cite the new audit row + the prior CME_PRECLOSE PARK + the non-auditable MES note.

Acceptance: YAML parses; status field is `closed`; `/orient` no longer surfaces this item as open.

### Stage 5 — Verify + commit

```
python pipeline/check_drift.py
```

Then re-run rebalance to confirm `lane_allocation.json` reflects the new audit row (COMEX_SETTLE X_MES_ATR60 RR1.0 either deploys, paused-by-correlation, or paused-by-PARK depending on Stage 2 verdict).

Commit with message capturing all changes; open PR to main.

Acceptance: drift check exits 0; commit lands; PR opened.

## Risks

- **Runner environment drift** — venv may have cryptography/authlib pin issues per `feedback_mcp_partial_install_state_2026_05_01.md`. Mitigation: dry-run the runner with `--help` first; if import errors, fix venv before Stage 2.
- **Verdict fidelity** — runner output is law. We transcribe, we do not interpret. If runner says PARK, audit row says PARK.
- **Allocator behavior change** — if verdict is PASS_CHORDIA, COMEX_SETTLE X_MES_ATR60 RR1.0 may enter the live book ranking. Likely demoted by correlation gate vs already-deployed COMEX_SETTLE OVNRNG_100 RR1.5 (same session, similar fire mask), but lane_allocation.json may show a 4th DEPLOY entry. Document either way in commit.

## Done criteria

1. `chordia_audit_log.yaml` has an audit row for `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` dated 2026-05-03 ✓
2. `chordia_audit_log.yaml` has a `notes:` entry documenting MES Tier-C non-auditability ✓
3. `action-queue.yaml` shows `chordia_audit_unlock_pass_chordia_strategies.status: closed` ✓
4. `python pipeline/check_drift.py` exits 0 ✓
5. `lane_allocation.json` rebalance run reflects new audit row (chordia_audit_age_days: 0 for the new strategy) ✓
6. Commit + PR opened ✓
