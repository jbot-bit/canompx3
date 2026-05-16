---
task: "Author ATR_P50 + ATR_P70 Chordia-strict (has_theory: false) pre-reg drafts for MNQ_COMEX_SETTLE_E2_RR1.0_CB1_*_O5"
mode: IMPLEMENTATION
slug: atr-p50-p70-chordia-unlock-prereg-authoring
scope_lock:
  - docs/audit/hypotheses/drafts/2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.yaml
  - docs/audit/hypotheses/drafts/2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.yaml
  - docs/runtime/stages/atr-p50-p70-chordia-unlock-prereg-authoring.md
---

## Blast Radius

- 2 new YAML drafts under `docs/audit/hypotheses/drafts/` — quarantine path, NOT loaded by validator or runner without explicit invocation.
- Zero edits to `pipeline/`, `trading_app/`, `scripts/`, `tests/`, or `docs/institutional/`.
- Reads: `trading_app/config.py::ALL_FILTERS` (verified ATR_P50, ATR_P70 present at lines 3211, 3216); `docs/audit/hypotheses/2026-05-02-mnq-usdata1000-vwapmid-o15-rr10-chordia-unlock-v1.yaml` (sibling template); `docs/audit/results/2026-05-12-chordia-audit-queue-top3-prereg-recommendation.md` (Mode A statistics source).
- Writes: 2 draft files + this stage file. No DB writes. No allocator writes. No `chordia_audit_log.yaml` writes.
- Downstream: each draft becomes the input to an authorized `bash scripts/infra/prereg-loop.sh <draft>` run in a separate next-thread; that run is NOT executed in this stage.
- Reversibility: deleting the 2 draft files restores zero downstream effect.

## Plan Reference

`C:\Users\joshd\.claude\plans\you-sort-it-out-stateless-clock.md` — approved 2026-05-16.

## Acceptance

- Both YAML files exist under `docs/audit/hypotheses/drafts/`.
- Both omit `theory_citation` entirely (field-presence trap).
- Both have `filter_grounding_status.verdict: "NO_THEORY_GRANT"`.
- Both have `chordia_threshold_basis` referencing `t >= 3.79`.
- Both have `scope.strategy_id` matching exact canonical id from recommendation MD line 44-45.
- `pipeline/check_drift.py` passes (Check 57 `check_hypothesis_minbtl_compliance` should pass: n_trials=1, instrument=MNQ, sentinel-date 2026-05-16 >= 2026-05-12 active).
- `scripts/tools/estimate_k_budget.py --hypothesis <draft>` returns passed=True (K=1 auto-pass).

## Completion (2026-05-17)

- Drafts written 2026-05-16 22:36 (P50 16320 bytes, P70 16836 bytes).
- Acceptance verified 2026-05-17:
  - Both files exist under `docs/audit/hypotheses/drafts/`.
  - Neither contains a `theory_citation:` field (grep confirmed) — loader will keep strict t >= 3.79.
  - Both carry `filter_grounding_status.verdict: "NO_THEORY_GRANT"` and `chordia_threshold_basis` referencing the Chordia 2018 strict bound (line 163-164 each).
  - `scope.strategy_id` matches recommendation MD lines 44-45 exactly: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50` and `..._ATR_P70`.
  - `scripts/tools/estimate_k_budget.py` returns `Verdict: PASS` (N=1, 6.65yr headroom) for both.
  - `python pipeline/check_drift.py` -> 130 PASSED / 0 skipped / 20 advisory. No new violations.
- Stage scope was draft authoring only. No `pipeline/`, `trading_app/`, `scripts/`, `tests/`, or `docs/institutional/` edits — dead-code sweep vacuously satisfied.
- Next-thread handoff: each draft becomes input to `bash scripts/infra/prereg-loop.sh <draft>` in a separate authorized session. That run is NOT executed here.
