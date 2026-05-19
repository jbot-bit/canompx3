# Stage — Triage validated_setups backlog (Improvement 3 of 3)

task: New `scripts/research/triage_validated_setups.py` script that
enumerates active validated_setups, filters to lanes with no existing
fast-lane result MD, ranks them by score components grounded in canonical
helpers (`research.oos_power.one_sample_power`,
`trading_app.eligibility.builder.parse_strategy_id`), and emits ranked
fast-lane v5.1 prereg drafts under `docs/audit/hypotheses/drafts/`. Pure
inventory expansion of the PROMOTE queue's upstream side; drafts live in
quarantine until the operator promotes intentionally.

mode: IMPLEMENTATION

## Scope Lock

- scripts/research/triage_validated_setups.py
- docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml
- pipeline/check_drift.py
- tests/test_research/test_triage_validated_setups.py

## Blast Radius

- `scripts/research/triage_validated_setups.py` — NEW. Reads
  `validated_setups`, `orb_outcomes`, `daily_features` (read-only); scans
  `docs/audit/results/` for existing fast-lane MDs (read-only); writes
  v5.1 prereg drafts to `docs/audit/hypotheses/drafts/<slug>.draft.yaml`
  (quarantine zone — LHP validator does not read drafts/).
- `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` — EDIT: adds
  optional `triage_provenance` block schema. Pure additive, existing
  hand-authored preregs unaffected.
- `pipeline/check_drift.py` — adds Check #165: every triage draft (file
  contains `triage_provenance:` block) must declare
  `triage_provenance.source_validated_setup_strategy_id`.
- Reads: gold.db (read-only), validated_setups, orb_outcomes,
  daily_features, docs/audit/results/*.md, docs/audit/hypotheses/.
- Writes: docs/audit/hypotheses/drafts/<slug>.draft.yaml.
- No capital-path mutation. No edit under `trading_app/`, `pipeline/`
  (other than check_drift.py), `docs/runtime/lane_allocation.json`,
  or `docs/runtime/chordia_audit_log.yaml`.

## Verification

1. `python pipeline/check_drift.py` passes.
2. `python -m pytest tests/test_research/test_triage_validated_setups.py -v`
   shows all-green; injection probes confirm Check #165 catches drafts
   without `triage_provenance.source_validated_setup_strategy_id`.
3. `python scripts/research/triage_validated_setups.py --top-k 3 --dry-run`
   round-trip: confirms drafts would land under `drafts/`, each with
   `triage_provenance.source_validated_setup_strategy_id` populated.
