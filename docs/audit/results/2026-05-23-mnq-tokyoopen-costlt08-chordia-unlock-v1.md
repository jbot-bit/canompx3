# Chordia strict unlock audit — MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08

**Prereq file:** `docs\audit\hypotheses\2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.00, has_theory=True) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_PROTOCOL_A`

IS clears theory threshold 3.00 with N=427 and ExpR=0.2037; OOS sign matches at N_OOS=69.

**MEASURED theory mode:** `THEORY_GRANT_ON_SESSION_MECHANISM_ONLY`
**MEASURED threshold applied:** `3.00`
**MEASURED loader has_theory:** `True`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1551 | 427 | 27.53% | 0 | 0 | 0.2037 | 0.0561 | 0.1726 | 3.566 | 0.00036 |
| OOS | 81 | 69 | 85.19% | 0 | 0 | 0.2767 | 0.2357 | 0.2313 | 1.921 | 0.05469 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 203 | 0.1618 | 1.950 | 224 | 0.2416 | 3.065 |
| OOS | 35 | 0.2959 | 1.450 | 34 | 0.2570 | 1.246 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'COST_LT08', 'TOKYO_OPEN')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.md`
- `docs\audit\results\2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## Cohort match (canonical vs validated_setups)

- `validated_setups.sample_size` (active row, 2026-05-23 query) = **427** (wins+losses).
- Canonical Mode A IS `N_fired` = **427** (this run).
- `|N_canonical − N_validated_setups|` = **0**. No Mode A vs Mode B divergence; the validated_setups row is consistent with strict Mode A replay on (IS = `trading_day < 2026-01-01`).

## BH-FDR at K_family=1

- p_two = **0.00036** (IS one-sample, df = 426).
- BH-adjusted q at K_family=1, K_lane=1, K_session=1, K_global=1 → q ≡ p = **0.00036 < 0.05**.
- Multiple-testing gate cleared trivially.

## N_unique_trading_days

| Split | N_fired | N_unique_trading_days |
|---|---:|---:|
| IS | 427 | 427 |
| OOS | 69 | 69 |

Clustered-SE floor (N_unique_trading_days ≥ 30 doctrine, `feedback_n_unique_trading_days_floor_clustered_se.md`) satisfied trivially: every fire is on a distinct trading day, so `t_naive` = `t_clustered`.

## Per-year breakdown (Criterion 9 era stability)

| Year | Split | N_fired | ExpR | Sharpe | wins | losses | scratch | WR | era_kill? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2020 | IS | 90 | 0.2812 | 0.2386 | 49 | 41 | 0 | 0.544 | no |
| 2021 | IS | 40 | 0.2277 | 0.1925 | 21 | 19 | 0 | 0.525 | no |
| 2022 | IS | 99 | 0.2620 | 0.2217 | 53 | 46 | 0 | 0.535 | no |
| 2023 | IS | 16 | -0.2728 | -0.2449 | 5 | 11 | 0 | 0.312 | no |
| 2024 | IS | 52 | 0.2692 | 0.2268 | 28 | 24 | 0 | 0.538 | no |
| 2025 | IS | 130 | 0.1306 | 0.1098 | 62 | 68 | 0 | 0.477 | no |
| 2026 | OOS | 69 | 0.2767 | 0.2313 | 37 | 32 | 0 | 0.536 | no |

- **Era kill triggered:** **False** (Criterion 9 rule: `ExpR < -0.05 AND N >= 50`; 2023 ExpR=-0.27 but N=16 < 50 → does NOT kill).
- `all_years_positive = False` is confirmed (2023 negative). The 2023 cell is below the era-stability significance floor; carry as an era-stability caveat, not a kill.
- 2025 shows decay (+0.131, N=130, still positive) — flag for routine fitness monitoring post-deploy.

## OOS power tier (RULE 3.3 — `research.oos_power.one_sample_power`)

- IS effect size (one-sample framing): Cohen's d = `|t_IS| / sqrt(N_IS)` = 3.566 / sqrt(427) = **0.173**.
- N_OOS_fired = **69**.
- OOS power at alpha=0.05 two-sided: **29.3%**.
- N needed for 80% power at d=0.173: **266** OOS trades.
- **RULE 3.3 tier: `STATISTICALLY_USELESS`** (power < 0.50).

Implication: the OOS sign match (+ / +) is **not refutational evidence** and cannot rescue an IS verdict. Likewise, an OOS sign-flip at this power would not be refutational either. OOS contributes descriptive corroboration only. The Protocol A verdict is driven by IS t ≥ 3.00 with theory grant, NOT by OOS confirmation.

## OOS sign + dir_match

- IS ExpR sign: **+** (0.2037).
- OOS ExpR sign: **+** (0.2767).
- `OOS_sign_match`: **True**.
- N_OOS_fired = 69 (≥ 30 threshold), but OOS power tier = STATISTICALLY_USELESS, so the sign match is informational only.

## ARITHMETIC_ONLY check (RULE 8 — backtesting-methodology.md § 8.2)

`|wr_spread|` (filtered vs non-filtered cohort WR) and `|Δ_IS|` (filtered vs non-filtered cohort ExpR) require the non-COST_LT08 cohort, which is OUT OF SCOPE for this bounded exact-lane replay. The upstream comprehensive scan that selected this cell already evaluated the arithmetic_only gate (`research/comprehensive_deployed_lane_scan.py::test_cell`).

In-cohort accounting (what the replay CAN compute):

| Quantity | Value |
|---|---:|
| N_IS_fired (decided wins+losses) | 427 |
| Scratch count IS | 0 |
| WR_IS (decided) | 0.511 |
| ExpR_IS (scratch=0 effective) | 0.2037 |
| Fire-rate IS | 27.53% |

Fire-rate is within RULE 8.1 acceptable band (5% < 27.53% < 95%); no `extreme_fire` flag.

## Verdict (recap)

- **Protocol A theory threshold (Harvey-Liu with-theory):** t_IS = **3.566 ≥ 3.00** → **CLEARED**.
- **Strict Chordia sizing-up threshold (no-theory):** t_IS = 3.566 < 3.79 → **NOT CLEARED**. **No sizing-up authorization.**
- **Theory grant:** `theory_grant: true`, basis = Chan 2013 Ch 7 p.155-157 (cash-session-open stop-cascade). Extract present at `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`.
- **Attachment class:** parent SESSION-ENTRY mechanism only. **NOT a claim that COST_LT08 is itself a literature-grounded alpha mechanism.** COST_LT08 is a deployable cost-screen filter riding on a session-grounded edge. See **Amendment 3.4 (PROVISIONAL)** at `docs/institutional/pre_registered_criteria.md` and doctrine question at `docs/audit/doctrine-questions/2026-05-23-protocol-a-theory-grant-attachment-class.md`.
- **OOS:** sign-matches IS (+ / +), but at 29.3% power the sign match is descriptive only.
- **Era stability (Criterion 9):** clean (no era kill); 2023 underperformance flagged but below N≥50 significance floor.
- **Cohort match:** exact (N=427 in both validated_setups and Mode A canonical replay).
- **ARITHMETIC_ONLY:** not flagged within in-cohort accounting; out-of-cohort wr_spread test deferred to upstream scan.

**Final verdict: `PASS_PROTOCOL_A` with caveat.** Cell is 1-contract deployment-eligible under Amendment 3.0 with-theory rules, subject to:
- Amendment 3.4 (PROVISIONAL) attachment-class caveat — grant covers session-entry mechanism, NOT cost-screen alpha.
- No sizing-up authorization (strict 3.79 not cleared).
- Allocator correlation gate against incumbents and TOKYO_OPEN regime check.
- The cell does NOT serve as precedent for new mechanism-class-transfer grants until Amendment 3.4 re-audit closes.

## Audit-log entry (already written)

The PASS_PROTOCOL_A entry was written to `docs/runtime/chordia_audit_log.yaml` at lines 709-722 during the prereg + runner cycle. That entry's note carries the explicit session-entry-mechanism caveat required under Amendment 3.4 (PROVISIONAL) § 5. This stage does NOT modify the audit-log entry; it documents the verdict's grounding and the doctrine constraint under which it stands.

## Adversarial pigeonhole check (literature-grounded)

Before locking PASS_PROTOCOL_A under the mechanism-class-transfer construction, the following alternate framings were considered:

**1. "Margin to 3.00 is only 0.566 — is the verdict robust?"**
Within doctrine. Harvey-Liu 2015 establishes 3.00 as the with-theory threshold for cross-sectional anomalies (`harvey_liu_zhu_2015_cross_section.md`). The Protocol A threshold is binary FDP-derived; 3.566 ≥ 3.00 is a clearance, not "almost passing." There is no CI-fuzzing rule that demotes a passing t to "marginal."

**2. "Strict Chordia 3.79 is missed by 5.91% — does this matter for deployment?"**
Yes, doctrinally. Strict Chordia clearance authorizes sizing-up (Amendment 3.0); Protocol A clearance authorizes 1-contract deployment only. The 5.91% miss is the difference between "single-contract eligible" and "size-up eligible." This is reflected in the verdict caveat (no sizing-up).

**3. "Does Chan Ch 7 ground COST_LT08 as filter-class alpha?"**
**No.** Verified by direct extract search (`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md` lines 1-80): Chan grounds the parent session-entry stop-cascade mechanism (p.155, p.156-157 FSTX gap-momentum, p.167 support/resistance level breach). Chan contains zero mentions of `cost`, `cost ratio`, `friction`, `cost screen`, `narrow ORB`, `small range`, or `COST_LT`. The theory grant is therefore an attachment-class call: Amendment 3.4 (PROVISIONAL) Option B allows the parent-mechanism grant to cover this cell, with the explicit caveat that COST_LT08 is a deployable cost-screen filter, NOT a literature-grounded alpha mechanism.

**4. "Could Harris 2002 (adverse selection) elevate COST_LT08 to filter-class alpha?"**
**No.** Harris Ch 14 § 14.2 (`harris_2002_trading_exchanges_microstructure.md:64-90`) establishes adverse-selection cost as a *deduction* from realized edge, NOT a *predictor* of edge. A filter that selects low-cost-ratio days does not predict breakout success; it removes high-cost-deduction days from the realized P&L stream. Harris's framework explicitly treats this as `ARITHMETIC_ONLY` territory (operational cost-management). Granting filter-class alpha on Harris grounds would be a categorical pigeonhole error. This rules out an Option A escape route (no Harris-based path to filter-class theory_grant).

**5. "Could Fitschen Ch 3 alone ground this cell?"**
Partial. Fitschen Ch 3 (`fitschen_2013_path_of_least_resistance.md:42-57`) grounds intraday-trend-following on equity indices. That is consistent with the parent-mechanism grant but no stronger than Chan Ch 7 for the SAME attachment-class question. Citing Fitschen instead of Chan would not change the doctrine call.

**6. "Is the one-sample framing correct for OOS power?"**
Verified. IS t-stat was computed as `mean / (std / sqrt(N))` one-sample on pooled per-trade pnl_r. Cohen's d derivation `|t_IS|/sqrt(N_IS) = 0.1726` matches direct `mean/std = 0.2037/1.1802 = 0.1726` to 4 decimals. Power 29.3% via `research.oos_power.one_sample_power` is canonical.

**7. "Did I miss a per-direction sign-flip?"**
Verified clean. From the directional breakdown: IS long ExpR=+0.162, IS short ExpR=+0.242, OOS long ExpR=+0.296, OOS short ExpR=+0.257 — all four signs are positive and aligned. Direction sign-coherence is consistent with `pooled-finding-rule.md` (no heterogeneity).

**8. "Is 2023 (N=16, ExpR=-0.27) a true era-stability concern?"**
Flag carry-forward, not a kill. Criterion 9 requires N≥50 within an era cell before a negative ExpR triggers kill. 2023 N=16 below the floor; kill rule formally does not fire. But this cell is the proximate cause of `all_years_positive=False` in the validated_setups row. Routine post-deploy fitness monitoring is required.

**9. "Is the attachment-class call itself a pigeonhole?"**
This is the live doctrinal question — see Amendment 3.4 (PROVISIONAL) and the doctrine question note. The operator's decision is provisional Option B with re-audit gate: existing precedents stand for continuity, NO new mechanism-class-transfer grants are permitted until Amendment 3.4 re-audit closes. This verdict stands under that provisional doctrine and does NOT serve as precedent for new grants.

**Verdict stands as PASS_PROTOCOL_A under Amendment 3.4 (PROVISIONAL) with the explicit caveats:**
- Parent SESSION-ENTRY mechanism grant only — no COST_LT08 alpha claim.
- 1-contract deployment-eligible — no sizing-up.
- Not a precedent for new mechanism-class-transfer grants.
