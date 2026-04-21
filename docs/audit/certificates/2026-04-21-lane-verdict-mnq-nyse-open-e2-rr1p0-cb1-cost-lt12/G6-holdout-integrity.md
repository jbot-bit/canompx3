# G6 — Holdout-Integrity Certificate

**Candidate:** `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
**Pre-reg:** `UNAVAILABLE — grandfathered deployed lane`
**Pre-reg commit SHA at lock:** `UNAVAILABLE`

---

## Purpose

This lane is being judged only against the binding Phase A snapshot. Phase A already established that all six active lanes have `discovery_date` after the sacred `2026-01-01` holdout boundary while also carrying populated OOS fields.

## Required evidence

### 1. Pre-reg lock timestamp

- Pre-reg file: `UNAVAILABLE`
- Pre-reg commit SHA: `UNAVAILABLE`
- Pre-reg commit date: `UNAVAILABLE`
- First evaluation run SHA: `UNAVAILABLE`
- First evaluation run date: `UNAVAILABLE`
- Evaluation date > Pre-reg date? [ ] YES [x] NO (FAIL — cannot certify)

### 2. Holdout boundary enforcement

- [ ] Script-level holdout enforcement could be certified from a pre-reg path
- [x] Phase A snapshot A6 shows this deployed lane was discovered after `2026-01-01`

### 3. No post-hoc iteration against 2026 OOS

- [ ] Zero between-commits certified from a pre-reg lineage
- [x] Cannot certify because no pre-reg lock is attached to this live row

### 4. Mode A compliance

- [ ] 2026 OOS was preserved as untouched holdout
- [x] This lane already shows `wf_tested=True` / `wf_passed=True` / `oos_exp_r=0.1003`
- Declared fresh-OOS start date: `UNAVAILABLE`

## Verdict

- [x] FAIL — `discovery_date=2026-04-11` with live OOS fields populated is not Mode-A holdout-clean.

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4
- `docs/institutional/pre_registered_criteria.md` Amendment 2.7

## Authored by / committed

- Author: `Codex`
- Commit SHA of candidate script: `5e768af8`
- Commit SHA of this certificate: `5e768af8`
