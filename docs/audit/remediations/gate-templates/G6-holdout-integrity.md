# G6 — Holdout-Integrity Certificate

**Candidate:** ________________________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`
**Pre-reg commit SHA at lock:** ________________

---

## Purpose

Per Chan 2013 Ch 1 p.4 (`docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`):

> Data-snooping bias arises when we iterate against the out-of-sample to tune parameters. Each iteration consumes the OOS's status as an independent test.

Enforced in-repo via `trading_app.holdout_policy.enforce_holdout_date()` and CLI `--holdout-date 2026-01-01` (Amendment 2.7 of `pre_registered_criteria.md`).

## Required evidence

### 1. Pre-reg lock timestamp

- Pre-reg file: `docs/audit/hypotheses/________.yaml`
- Pre-reg commit SHA: ______________________________
- Pre-reg commit date: ______________________________
- First evaluation run SHA: ______________________________
- First evaluation run date: ______________________________
- Evaluation date > Pre-reg date? [ ] YES [ ] NO (FAIL)

### 2. Holdout boundary enforcement

- [ ] Script uses `trading_app.holdout_policy.enforce_holdout_date()`
- [ ] OR: script uses CLI `--holdout-date 2026-01-01` with the arg actually threaded through
- [ ] SQL WHERE clause includes `trading_day < DATE '2026-01-01'` for IS queries

Evidence (grep of the script):
```
$ grep -n "holdout\|HOLDOUT_SACRED\|2026-01-01\|enforce_holdout" <script>
<paste output>
```

### 3. No post-hoc iteration against 2026 OOS

This is the Chan p.4 hazard. Check the git log for the pre-reg file and script:

```
$ git log --oneline --follow <pre-reg yaml>
$ git log --oneline --follow <script>
```

Any commit to the script AFTER the pre-reg lock AND before the OOS evaluation that changes a parameter, threshold, or feature selection is a data-snooping violation.

- [ ] Zero commits to candidate's script between pre-reg lock and first OOS evaluation
- [ ] OR: all between-commits were infrastructure-only (no parameter changes; reviewable via diff)

Evidence:
```
$ git diff <pre-reg-sha> <first-eval-sha> -- <script>
<paste output or statement of no diff>
```

### 4. Mode A compliance

If candidate cites 2026 OOS evidence:
- [ ] 2026 OOS was CONSUMED (not peeked + iterated) per Mode A discipline
- [ ] Fresh-OOS-from-today accrual window is declared

Declared fresh-OOS start date: ______________________________

## Verdict

- [ ] CLEAR — pre-reg locked before evaluation; no post-hoc iteration; Mode A enforced; fresh-OOS accrual declared.
- [ ] FAIL — pre-reg lock timestamp is AFTER first evaluation run, OR post-hoc script edits present, OR Mode A not enforced.

## Failure disposition

FAIL → candidate cannot advance. The OOS evidence is CONTAMINATED per Chan p.4. Path forward:
1. Rewrite pre-reg with fresh lock timestamp
2. Declare a NEW fresh-OOS accrual window (post-now)
3. Wait for OOS accrual
4. Re-run G6 with new evidence

No shortcut. No retroactive "oh we meant to lock first."

## Literature citation

- `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4
- `docs/institutional/pre_registered_criteria.md` Amendment 2.7

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of this certificate: ________________
