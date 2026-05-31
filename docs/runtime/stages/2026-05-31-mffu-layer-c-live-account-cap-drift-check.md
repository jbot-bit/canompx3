---
task: "MFFU Layer C (Option A) — drift check that every live_funded payout policy declares max_live_accounts; populate Topstep; correct stale memo"
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - trading_app/prop_profiles.py
  - tests/test_tools/test_prop_live_account_cap_drift.py
  - docs/audit/2026-05-31-mffu-forced-progression-live-cap-memo.md
---

## Blast Radius
- `pipeline/check_drift.py` — NEW check `check_live_funded_firms_declare_max_live_accounts` + one CHECKS tuple registration. No existing check touched. Read-only over `PAYOUT_POLICIES` / `PROP_FIRM_SPECS`. Adds to the runtime check count (self-reported, not hardcoded).
- `trading_app/prop_profiles.py` — adds `firm_specific_rules={"max_live_accounts": 1, ...}` to the `topstep` PropFirmSpec (currently None). DATA add to a frozen config dataclass; no logic, no new field on the class (field already exists, line 60). MFFU specs already declare it.
- `tests/test_tools/test_prop_live_account_cap_drift.py` — NEW targeted test for the drift check (happy path, missing-data fail, firm-name-mismatch fail).
- `docs/audit/2026-05-31-mffu-forced-progression-live-cap-memo.md` — doc-only: mark data-already-built, anecdote RESOLVED, no avoidance path, max_live_accounts=1 is default/entry cap (discretionary exceptions per verbatim).
- Reads: none (no gold.db). Writes: none.
- Capital-path logic: NOT touched (Option A scope lock — no state machine, no ledger, no routing).

## Verbatim grounding
- Topstep 1-live cap: `docs/research-input/topstep/topstep_xfa_parameters.txt:228` "Reminder: Only 1 Live Funded Account is permitted." + `:235` "Traders can only have one (1) Live Funded Account."
- MFFU Builder 1-live cap: already encoded `prop_profiles.py:351` from `mffu_builder_50k.md` (article 14290805).
- No-avoidance-path: `docs/research-input/mffu/mffu_live_accounts_faq.md:237` "rejecting the move to a Live Funded Account is not possible once you are selected."
