# 2026-04-24 Codex Audit Remediation

**Status:** IMPLEMENTATION
**Date:** 2026-04-24
**Purpose:** close the Tier 3 guardrail-enforcement gaps found in the 3-day Codex audit (plan at `.claude/plans/we-have-an-emergency-compressed-cupcake.md`). Live-capital code (Tier 1/2) audited SAFE — this stage only hardens the drift layer, removes ceremonial tooling, and documents removed pre-commit advisories.

mode: IMPLEMENTATION

scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_context.py
  - tests/test_pipeline/test_check_drift_db.py
  - tests/test_pipeline/test_check_drift_slow_labels.py
  - docs/runtime/decision-ledger.md
  - docs/governance/system_authority_map.md
  - .claude/settings.json
  - scripts/tools/bias_grounding_guard.py
  - docs/audit/hypotheses/README.md
  - research/oos_power.py
  - research/oos_evidence.py
  - tests/test_research/test_oos_evidence.py

blast_radius: drift-check layer (pipeline/check_drift.py + companion tests) — adds check_handoff_compact_policy, tightens check_schema_query_consistency SQL detection to require verb+scope keyword pair (fixes work_queue.py docstring false positive), increments CHECKS count. Hook-side: .claude/settings.json loses bias_grounding_guard entry; scripts/tools/bias_grounding_guard.py deleted. Governance: system_authority_map.md re-rendered to clear existing Check 104 drift. Docs: decision-ledger.md gains M2.5 removal entry; oos_power/oos_evidence docstrings updated to cite docs/institutional/literature/ extract paths. No live-capital surfaces touched; no canonical source modified; drift check count increments by 1 (self-reporting).

## Inputs

- Audit plan: `.claude/plans/we-have-an-emergency-compressed-cupcake.md`
- Three Explore-agent reports (Tier 1/2/3) synthesized into the plan
- Phase A verification: drift exit 1 (3 real violations: work_queue.py docstring false positive, 16 stale validated_setups provenance rows [operational, out of scope], system_authority_map drift)
- Phase A-3: grep confirmed zero SQL writes in the 5 pre-reg-missing scripts → confirmatory-exempt
- Phase A-4: `git diff main@{3.days.ago}..main -- pipeline/cost_model.py` has zero numeric-line changes
- Phase A-5: `trading_app/eligibility/builder.py` delta is single `(list, tuple)` → `list | tuple` PEP 604 rewrite

## Exit criteria

1. `PYTHONPATH=. python pipeline/check_drift.py` exits 0 (work_queue false positive resolved + system_authority_map re-rendered + new HANDOFF compact check passes).
2. `python -m pytest tests/test_pipeline tests/test_trading_app tests/test_research -x -q` exits 0.
3. `scripts/tools/bias_grounding_guard.py` absent, hook entry absent from `.claude/settings.json`.
4. `docs/runtime/decision-ledger.md` contains a dated entry for M2.5 removal citing commit `b5b1492d`.
5. Out-of-scope flag raised to user: 16 `validated_setups` rows have stale provenance (operational, needs pipeline refresh, not a code bug).
