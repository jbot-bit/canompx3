---
task: Extend ChordiaAuditEntry to load 4 addendum fields (t_stat_source, t_stat_csv_recompute, oos_n, oos_power) + audit_reaffirmed_date — currently silently dropped at load
mode: DESIGN
scope_lock:
  - trading_app/chordia.py
  - tests/test_trading_app/test_chordia.py
---

## Blast Radius

- `trading_app/chordia.py` — extends `ChordiaAuditEntry` dataclass with 5 new optional fields (str / float / int / Optional[date]). Loader must default-initialize when YAML omits them (backward compat).
- `tests/test_trading_app/test_chordia.py` — adds parse-and-roundtrip cases for the 5 fields.
- Reads: `docs/runtime/chordia_audit_log.yaml` (canonical source).
- Writes: none.
- Affects: any consumer that wants to read the addendum fields. Currently `lane_allocator.py` only reads `verdict` + `audit_date`, so no immediate consumer change. New optional consumer: a future `audit_trail_report.py` that surfaces the t_stat reproducibility delta.

## Why

Audit of merge commit `48452d80` (PR #221) found that `chordia_audit_log.yaml` carries 4 fields added by PR #213 (`t_stat_source`, `t_stat_csv_recompute`, `oos_n`, `oos_power`) plus 1 added by my merge resolution (`audit_reaffirmed_date`). The loader at `trading_app/chordia.py` lines 169-184 builds a `ChordiaAuditEntry` from the YAML and silently drops any field not declared on the dataclass. This means:

1. Schema in YAML and schema in `chordia.py` have diverged.
2. Any downstream code that tries to access `entry.oos_power` raises `AttributeError`.
3. The reproducibility audit trail captured in PR #213 is invisible to programmatic consumers; only humans reading the YAML directly see it.

This is **NOT** a capital-risk issue (allocator only consumes `verdict` + `audit_date`), so it does not block PR #221. But it IS an institutional-rigor regression per `.claude/rules/integrity-guardian.md` § 7 ("Never trust metadata — always verify"): the addendum fields exist as YAML metadata not backed by code contract.

## Decision points to resolve before IMPLEMENTATION

1. **Field types**: Should `t_stat_csv_recompute` be `float | None` or `str | None` (to allow "N/A — pre-CSV-era audit row")? Default: `float | None` with `None` for older rows.
2. **`oos_power`**: Range-check at load (0.0 ≤ power ≤ 1.0)? Default: yes, log.warning + accept on out-of-range.
3. **`audit_reaffirmed_date`**: Multi-valued (list of reaffirmation dates)? Default: scalar `date | None`. Multi-valued is a future need.
4. **Drift check #136 expansion**: Should the chordia-gate check also enforce "if `t_stat_csv_recompute` exists, must be < 5% relative delta from `t_stat`"? Default: separate drift check, not in this stage.

## Files

- `trading_app/chordia.py` (extend dataclass)
- `tests/test_trading_app/test_chordia.py` (5 new test cases)

## Verification

- Unit tests: load a YAML with all 5 fields, assert dataclass has them.
- Unit tests: load a YAML missing all 5 fields, assert dataclass has `None` defaults.
- Drift check: existing #136 still passes (no behavior change for the verdict/audit_date path).

## Provenance

- Audit date: 2026-05-04
- Audit reference: PR #221 evidence-auditor pass on merge commit 48452d80 (auditor agent ID a430b416a83639462)
- PR #213 (`446fc798` / merge `e6024fe6`) introduced the 4 fields without consumer-side support.
