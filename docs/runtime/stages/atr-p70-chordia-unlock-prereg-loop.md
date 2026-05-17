---
task: Execute ATR_P70 Chordia-unlock prereg-loop on existing draft
mode: TRIVIAL
updated: 2026-05-17T11:00:00+10:00
scope_lock:
  - docs/audit/hypotheses/drafts/2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.yaml
  - docs/audit/results/
---

## Blast Radius
- No production code edits. Single shell invocation: `bash scripts/infra/prereg-loop.sh <draft-yaml>` → `prereg_front_door.py` → K-budget gate + static checks + bounded runner.
- Reads: gold.db (read-only canonical layer), draft yaml.
- Writes: NEW result MD + CSV under `docs/audit/results/` (mirrors P50 sibling output pattern from commit 091a03e9).
- No mutation of: validated_setups, allocator, experimental_strategies, chordia_audit_log.yaml.
- Confirmatory K=1 replay; MinBTL budget already cleared by P50 sibling.

## Acceptance
- prereg-loop exits 0
- New result MD authored under `docs/audit/results/` with verdict
- `python pipeline/check_drift.py` passes

## Completion (2026-05-17)

**Status:** COMPLETE.

**Files changed:**
- `docs/audit/hypotheses/drafts/2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.yaml` — added `metadata.theory_grant: false` per Amendment 3.3 (commit `8ab4fe13`, 2026-05-17) explicit-bool requirement. Draft authored 2026-05-16 predated the gate.
- `docs/audit/results/2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.md` — runner-authored, then appended Role decision section with OOS power tier verdict.
- `docs/audit/results/2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.csv` — runner-authored.

**MEASURED verdict:** `PASS_CHORDIA` (IS N=578, ExpR=0.1731, t=4.62 vs strict 3.79). OOS pooled sign matches at N=47.

**ROLE-DECISION verdict:** `UNVERIFIED_INSUFFICIENT_POWER` per RULE 3.3. OOS power 12.8% pooled / 11.1% long / 9.9% short — all STATISTICALLY_USELESS tier. Long-side OOS sign flip (ExpR=-0.041 vs IS +0.178) is noise-consistent at this power, not refutational. Mirrors P50 sibling discipline (commit `091a03e9`, 2026-05-17 AM).

**No mutation:** validated_setups, lane_allocation.json, experimental_strategies, chordia_audit_log.yaml, prop_profiles.py all untouched.

**Drift:** 11 pre-existing orphaned-SHA violations in `experimental_strategies` (Check 107) — unrelated to this stage. 0 new violations introduced.
