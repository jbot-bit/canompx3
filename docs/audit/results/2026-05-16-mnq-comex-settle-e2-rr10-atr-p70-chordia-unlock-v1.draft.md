# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70

**Prereq file:** `docs\audit\hypotheses\drafts\2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.yaml`
**Result CSV:** `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=578 and ExpR=0.1731; OOS sign matches at N_OOS=47.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Role decision (RULE 3.3 OOS power floor)

**ROLE-DECISION verdict:** `UNVERIFIED_INSUFFICIENT_POWER`

`PASS_CHORDIA` above is a MEASUREMENT — IS clears the strict Chordia (2018)
hurdle of `t >= 3.79` with no theory grant. It is NOT a deployment role
decision. Per `.claude/rules/backtesting-methodology.md` RULE 3.3, any binary
OOS gate (sign-flip, dir_match, p_oos) is only legitimately refutational
when the OOS sample carries >=50% power to detect the IS effect size;
below 50% the OOS slice is noise-consistent with both "signal alive" and
"signal dead" and cannot kill an IS finding.

OOS power computed via canonical helper `research/oos_power.py::oos_ttest_power`
(Welch two-sample non-central t; conservative two-arm proxy for the
single-arm Chordia framing). Not a direct `resources/` citation — this is
mechanical scipy.stats math, not a research claim.

| Slice | OOS N (each arm) | Cohen's d | Power | RULE 3.3 tier |
|---|---:|---:|---:|---|
| Pooled (proxy) | 47 | 0.192 | 15.2% | STATISTICALLY_USELESS |
| Long only      | 25 | 0.198 | 10.6% | STATISTICALLY_USELESS |
| Short only     | 22 | 0.185 |  9.2% | STATISTICALLY_USELESS |

All three tiers fall below the 50% floor, so the OOS slice cannot be used
to confirm OR refute the IS finding. Per-direction OOS sign-flip on long
(IS +0.1781 -> OOS -0.0410) at 10.6% power is noise-consistent and CANNOT
support a "long signal dead" claim. Required per-group N for 80% power is
~400-460; current OOS holds ~25-47.

**Do NOT call this lane deploy-ready.** No mutation of
`validated_setups`, `experimental_strategies`, `chordia_audit_log.yaml`,
`docs/runtime/lane_allocation.json`, or `prop_profiles.py`. This is the
same discipline as the ATR_P50 sibling (commit `091a03e9`) and the
VWAP_MID_ALIGNED_O30 verdict two days prior (commit `88a03d19`).

Literature grounding for the gate logic (not the power math):

- Chordia, Goyal & Saretto 2018 — `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`. Source of the `t >= 3.79` strict hurdle for the no-theory case.
- Harvey & Liu 2015 — `docs/institutional/literature/harvey_liu_2015_backtesting.md`. OOS as a Sharpe haircut / discount multiplier, not a binary veto. Underwrites the "do not refute on underpowered OOS" rule.
- López de Prado 2018 AFML Ch 12 — `docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md`. CPCV as preferred alternative when OOS holdout is too short for a legitimate binary refutation; cited as the structurally correct path once OOS sample is at ~400/group.

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 578 | 38.69% | 3 | 0 | 0.1731 | 0.0670 | 0.1923 | 4.623 | 0.00000 |
| OOS | 77 | 47 | 61.04% | 1 | 0 | 0.1610 | 0.0983 | 0.1725 | 1.183 | 0.23687 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 318 | 0.1781 | 3.537 | 260 | 0.1670 | 2.976 |
| OOS | 25 | -0.0410 | -0.214 | 22 | 0.3905 | 2.101 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ATR_P70', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\drafts\2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.md`
- `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p70-chordia-unlock-v1.draft.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## Role decision (post-measurement, 2026-05-17)

**ROLE-DECISION verdict:** `UNVERIFIED_INSUFFICIENT_POWER`

The MEASURED verdict `PASS_CHORDIA` reports that IS clears strict t≥3.79 (N=578, ExpR=0.1731, t=4.62). The role-decision verdict is distinct: it asks whether OOS evidence is statistically powered to confirm or refute the IS finding, per RULE 3.3 (OOS power floor) in `.claude/rules/backtesting-methodology.md`.

### OOS power computation (canonical helper `research/oos_power.py`)

| Direction | IS delta | N_OOS_fired | N_OOS_off | Cohen's d | Power @ α=0.05 | Tier |
|---|---:|---:|---:|---:|---:|---|
| Pooled | 0.1731 | 47 | 30 | 0.192 | 12.8% | STATISTICALLY_USELESS |
| Long | 0.1781 | 25 | 30 | 0.198 | 11.1% | STATISTICALLY_USELESS |
| Short | 0.1670 | 22 | 30 | 0.186 | 9.9% | STATISTICALLY_USELESS |

N per group for 80% power: ~426 (pooled) / ~402 (long) / ~457 (short). OOS has 47 fired trades total.

### Why UNVERIFIED, not PARK or DEAD

- All three power tiers are `STATISTICALLY_USELESS` (<50%); per RULE 3.3 tier table, verdict MUST be `UNVERIFIED`, never `DEAD`.
- OOS Long shows ExpR=-0.041 (sign flip vs IS +0.178), but at 11.1% power this is noise-consistent with the IS effect remaining alive — cannot distinguish "signal dead", "signal alive", or "signal reversed".
- OOS Short shows ExpR=+0.391 (lift vs IS +0.167), also noise-consistent at 9.9% power.
- This matches the prior session's discipline on the paired ATR_P50 sibling (verdict `UNVERIFIED_INSUFFICIENT_POWER`, commit `091a03e9`) and the 88a03d19 VWAP_MID_ALIGNED O30 override two days prior.

### Governance

- No mutation of `validated_setups`, `lane_allocation.json`, `experimental_strategies`, `chordia_audit_log.yaml`, or `prop_profiles.py`. Pure measurement + role-decision append.
- Same-author governance disclosure (per recommendation MD Caveat 5): paired with ATR_P50 sibling authored 2026-05-16 by the same author; SR-review-registry independent-review gate not satisfied. Disclosure recorded in `feedback_bootstrap_disclosure_not_separation_of_duties.md`.
- Reopen criteria: would require N_OOS to grow to ≥400 per group (~5–10 more years of MNQ COMEX_SETTLE data) OR an independent confirmatory pre-reg authored by a different operator OR a Harvey-Liu Sharpe-haircut framing replacing the binary OOS gate.

### Literature anchors

- Harvey & Liu 2015 (`docs/institutional/literature/harvey_liu_2015_backtesting.md`) — OOS as Sharpe discount, not binary veto.
- López de Prado 2018 AFML Ch 12 (`docs/institutional/literature/lopez_de_prado_2018_afml_ch_3_7_8.md`) — CPCV addresses underpowered OOS slices for ML asset managers.
- Chordia et al 2018 (`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`) — t≥3.79 strict empirical bound (this is the IS hurdle, satisfied).
