---
task: Extend ChordiaAuditEntry to load 6 fields currently silently dropped (t_stat, t_stat_source, t_stat_csv_recompute, oos_n, oos_power, audit_reaffirmed_date) — closes the institutional-rigor regression flagged on PR #221 evidence-auditor pass
mode: IMPLEMENTATION
scope_lock:
  - trading_app/chordia.py
  - tests/test_trading_app/test_chordia.py
---

## Blast Radius

- `trading_app/chordia.py` — extends `ChordiaAuditEntry` dataclass (line 169-184) with 6 new optional fields. Loader at lines 281-297 + 303-312 default-initializes when YAML omits them (backward compat — every existing audit row keeps working).
- `tests/test_trading_app/test_chordia.py` — adds new `TestLoadChordiaAuditLog` class with 4 parse-and-roundtrip cases. Existing 4 test classes (TestChordiaThreshold, TestComputeChordiaT, TestChordiaGate, TestSyntheticDistributions) untouched.
- Reads: `docs/runtime/chordia_audit_log.yaml` (canonical source, not modified by this stage).
- Writes: none in production; test fixtures synthesized in-memory.
- Affects (current consumers): zero. `lane_allocator.py:270,640` and `tests/test_trading_app/test_lane_allocator.py:750` only consume `verdict` + `audit_date`. Adding optional fields is strictly additive.
- Affects (future consumers): any audit-trail report that wants to surface t_stat reproducibility delta or OOS power context. Filed but not in scope of this stage.

## Why

Audit of merge commit `48452d80` (PR #221) found that `chordia_audit_log.yaml` carries 4 fields added by PR #213 (`t_stat_source`, `t_stat_csv_recompute`, `oos_n`, `oos_power`) plus 1 added by the merge resolution (`audit_reaffirmed_date`). The loader at `trading_app/chordia.py:281-297` builds a `ChordiaAuditEntry` from the YAML and silently drops any field not declared on the dataclass. Confirmed by grep on 2026-05-05: zero downstream consumers reference `entry.t_stat`, `entry.oos_power`, `entry.oos_n`, `entry.audit_reaffirmed_date`, or `entry.t_stat_source` anywhere in `trading_app/`, `pipeline/`, `scripts/`.

Schema in YAML and schema in code have diverged. This is **NOT** capital-risk (allocator only consumes `verdict` + `audit_date`), but it IS an institutional-rigor regression per `.claude/rules/integrity-guardian.md` § 7 ("Never trust metadata — always verify"): the addendum fields exist as YAML metadata not backed by code contract.

## Decisions resolved (2026-05-05 walkthrough)

All 4 stage-doc decisions resolved on the stage-doc default; rationale below.

| # | Decision | Resolution | Rationale |
|---|----------|-----------|-----------|
| 1 | `t_stat_csv_recompute` type | `float \| None` (default `None`) | Only existing value is float64 (4.323). Older rows omit the key → `None`. No row needs a string sentinel. |
| 2 | `oos_power` range check | `log.warning` if outside `[0.0, 1.0]`; accept value | Consistent with chordia.py:240-261 "loud-but-non-fatal for non-load-bearing fields" pattern. Allocator does not consume oos_power, so out-of-range must NOT fail-closed. |
| 3 | `audit_reaffirmed_date` shape | scalar `date \| None` (default `None`) | YAML evidence: 1 row, scalar ISO date. Multi-valued is a future need; YAML migration to list is non-breaking when justified. |
| 4 | Drift check #136 expansion | OUT OF SCOPE | Schema vs validation are different concerns. Only 1 row currently has both `t_stat` + `t_stat_csv_recompute`; per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`, do not build forcing functions on n=1. Re-file as separate stage when N≥3 rows have both. |

6 new fields (4 from PR #213 + 1 from PR #221 merge + `t_stat` peer):

| Field | Type | Default | Source row count |
|-------|------|---------|------------------|
| `t_stat` | `float \| None` | `None` | 11 of 13 audit rows (verified) |
| `t_stat_source` | `str \| None` | `None` | 1 row (`MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`) |
| `t_stat_csv_recompute` | `float \| None` | `None` | 1 row (same) |
| `oos_n` | `int \| None` | `None` | 1 row (same) |
| `oos_power` | `float \| None` | `None` | 1 row (same) |
| `audit_reaffirmed_date` | `date \| None` | `None` | 1 row (same) |

Note: `t_stat` is on 11 rows, not in the original 4 PR #213 fields list, but is the obvious peer to `t_stat_csv_recompute` and SHOULD be loaded for consistency. Sample size for justifying inclusion: 11 rows, well past the n=1 caution threshold.

## Files

- `trading_app/chordia.py`:
  - Extend `ChordiaAuditEntry` dataclass at lines 169-184 with 6 new fields, all optional with `None` defaults.
  - Modify loader at lines 281-297 + 303-312 to extract the new fields from the YAML `audits[]` list with safe coercion (`audit.get(...)` returning `None` when absent).
  - Add `oos_power` range-check log.warning between extraction and entry construction.
  - Date coercion for `audit_reaffirmed_date` mirrors existing `audit_date` handling (str → `date.fromisoformat`, else None).
- `tests/test_trading_app/test_chordia.py`:
  - New `TestLoadChordiaAuditLog` class with 4 cases:
    1. YAML with all 6 fields populated → dataclass has them with correct types.
    2. YAML missing all 6 fields → dataclass has all `None` defaults (backward compat).
    3. YAML with `oos_power=1.5` (out of range) → log.warning emitted, value accepted.
    4. YAML with `audit_reaffirmed_date` as ISO string → coerced to `date` object.

## Verification

- `python -m pytest tests/test_trading_app/test_chordia.py -q` → 4 new tests pass + 4 existing test classes still pass (regression).
- `python -m pytest tests/test_trading_app/test_lane_allocator.py -q` → existing allocator tests pass unchanged (proves additive-only contract).
- `python pipeline/check_drift.py` → 118+ checks PASS, including check #136 (chordia gate freshness) — no behavior change for the verdict/audit_date path.
- `grep -rn "ChordiaAuditEntry(" trading_app/ tests/` → every constructor call still type-checks (new fields all have defaults).
- Loaded the live `docs/runtime/chordia_audit_log.yaml` and verified `entry.t_stat == 4.361` and `entry.t_stat_csv_recompute == 4.323` for `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`.

## Out of scope (for follow-up stages, NOT this one)

1. Drift check enforcing `|t_stat - t_stat_csv_recompute| / t_stat < 0.05` (file when N≥3 rows have both fields).
2. Audit-trail surfacing tool that consumes the new fields (no current operator request).
3. Multi-valued `audit_reaffirmed_date` support (no current YAML row uses list form).
4. Validation that `oos_n >= 30` aligns with `oos_power` tier per RULE 3.3 (separate cross-field consistency check).

## Provenance

- Audit date: 2026-05-04 (PR #221 evidence-auditor pass)
- Walkthrough date: 2026-05-05 (this resolution doc)
- Audit reference: PR #221 merge commit `48452d80`, evidence-auditor agent ID `a430b416a83639462`
- PR #213 (`446fc798` / merge `e6024fe6`) introduced the 4 fields without consumer-side support
- Companion canon amendment: PR #224 (RULE 3.3 power floor) — required before this stage's tests reference RULE 3.3 directly. Implementation may proceed after PR #224 merges OR independently if tests do not cite RULE 3.3 (they do not need to — this stage is about schema, not validation).

## Implementation gate

Per `.claude/rules/workflow-preferences.md` § Implementation Gating, this stage file is DESIGN-resolved but does NOT authorize code edits yet. User must say "implement" / "build it" / "do it" / "go ship" before any edits to `trading_app/chordia.py` land.
