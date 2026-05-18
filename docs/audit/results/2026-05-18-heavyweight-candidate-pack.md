# Heavyweight Chordia candidate pack — 2026-05-18

**Status:** evidence pack only. No heavyweight prereg has been authored.
**Source:** `scripts/research/fast_lane_promote_queue.py` dry-run output 2026-05-18.
**Authoring policy:** operator authors the heavyweight prereg manually. This
pack supplies the evidence; no theory citations are invented.

## Candidate 1 — MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30

**Status label:** `UNVERIFIED_OOS_POWER`

OOS sample is `STATISTICALLY_USELESS` per RULE 3.3
(`.claude/rules/backtesting-methodology.md`). The heavyweight prereg, if
authored now, would have to declare the OOS dir_match check informational
only and route the verdict to `UNVERIFIED_INSUFFICIENT_POWER`. Operator may
prefer to park the lane until OOS N accumulates forward.

### FAST_LANE evidence
- result MD: `docs/audit/results/2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.md`
- prereg: `docs/audit/hypotheses/2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.yaml`
- FAST_LANE v5.1 verdict: PROMOTE (t=3.064)
- Filter direction: long (filter `PD_CLEAR_LONG` is in `E2_DIRECTION_SELECTOR_FILTER_PREFIXES`; pooled run == long-only run within rounding)

### IS stats (from result MD line 25)
| N_universe | N_fired | Fire% | ExpR | Sharpe | t |
|---:|---:|---:|---:|---:|---:|
| 1539 | 226 | 14.68% | 0.1708 | 0.2038 | 3.064 |

### Per-direction breakdown (filter-directional, short is leak)
| Side | N | ExpR | t |
|---|---:|---:|---:|
| Long | 221 | 0.1747 | 3.065 |
| Short | 5 | 0.0000 | nan |

### OOS stats (from result MD line 26)
| N_OOS | ExpR | t |
|---:|---:|---:|
| 14 | -0.0233 | -0.099 |

### OOS power computation (measured)
Computed via `research.oos_power.one_sample_power(d=0.2038, n=14, alpha=0.05)`:

| Metric | Value |
|---|---:|
| Cohen's d (proxy: IS Sharpe = mean/std) | 0.2038 |
| OOS sample size | 14 |
| OOS power to detect IS effect | **10.9%** |
| Power tier | **STATISTICALLY_USELESS** |
| N required for 80% power | 191 |

**Operational meaning:** at this OOS sample size, dir_match outcome is
noise-consistent. The sign-flip observed (IS positive, OOS slightly
negative) cannot confirm OR refute the IS finding. Verdict tier must be
`UNVERIFIED_INSUFFICIENT_POWER` (per RULE 3.3 +
`memory/feedback_chordia_oos_park_vs_unverified_power_floor.md`).

### Required heavyweight-prereg blocks (when operator authors)

These blocks MUST be present in any heavyweight prereg downstream of this
PROMOTE. The candidate pack does NOT pre-fill any of them — operator
authoring is required.

1. **theory_citation** — blank in this pack; operator must either supply a
   mechanism citation from `docs/institutional/literature/` or omit the
   field (per `memory/feedback_chordia_theory_citation_field_presence_trap.md`,
   any truthy value silently flips loader has_theory=True and downgrades
   strict t-hurdle from 3.79 to 3.00). Without theory grant, threshold is
   t_clustered ≥ 3.79.
2. **clustered SE at trading_day** — mandatory per
   `memory/feedback_clustered_se_trading_day_pooled_finding_guard.md`.
   Report `t_naive` and `t_clustered` side-by-side; naïve-flip kill if
   t_naive ≥ 3.79 but t_clustered < 3.79.
3. **OOS power floor block** — pre-committed verdict on binary OOS kill:
   `UNVERIFIED_INSUFFICIENT_POWER` until measured power ≥ 0.50 (per the
   numbers above; this is already required to be UNVERIFIED at lock).
4. **Era-stability block** — per-year IS ExpR table; no era N≥50 with
   ExpR < -0.05 (Criterion 9 of `pre_registered_criteria.md`).
5. **Harvey-Liu haircut** — IS-side Sharpe haircut at K_effective accounting
   per `memory/feedback_harvey_liu_haircut_not_oos_validation_substitute.md`.
   Does NOT substitute for OOS validation.
6. **N_unique_trading_days ≥ 30 cluster floor** — per
   `memory/feedback_n_unique_trading_days_floor_clustered_se.md`.

### Operator decision points

- **Option A — author heavyweight now, accept UNVERIFIED verdict.** Useful
  if you want the lane in `chordia_audit_log.yaml` with explicit
  UNVERIFIED status; clears it from the orphan-PROMOTE queue. Costs
  K=1 trial.
- **Option B — park until OOS N≥30 + power≥0.50.** Cheapest path. Lane stays
  QUEUED in `promote_queue.yaml`; revisit when N_OOS ≥ 30. No K spent.
- **Option C — sibling PD_CLEAR_SHORT investigation.** Out of scope for this
  candidate pack — different filter, different cell. Would require its own
  FAST_LANE screen first (if validated_setups has the row).

### What this pack does NOT do

- Does NOT author the heavyweight prereg.
- Does NOT supply a theory citation.
- Does NOT mutate `chordia_audit_log.yaml`, `validated_setups`,
  `lane_allocation.json`, or any allocator/live state.
- Does NOT recommend deployment.
- Does NOT spend new K against the trial budget.
