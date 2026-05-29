---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/2026-05-29-mgc-o30-long-pooled-base-and-overlay.md#per-session-base-breakdown
flip_rate_pct: 40.0
heterogeneity_ack: true
---

# MGC O30 Long — Pooled Base + Conditional Overlay (Q1)

**Mandate:** Q1 framing correction to the 2026-05-29 MGC CPCV audit. Test whether the 6 underpowered `(session x DOW/vol)` CPCV slices are ONE MGC O30 long drift effect (poolable) or six noise draws. Pool the base FIRST, then test each condition as a RULE 2 Pass-1/Pass-2 overlay. Read-only canonical layers; NO DB writes; NO threshold relaxation. The fair fight can KILL MGC O30 long harder than the slice framing did.

**Scope:** MGC O30 long E2 CB1, RR in {1, 1.5, 2}. Costs: `COST_SPECS['MGC'].total_friction=5.74`; `pnl_r` already net -> scored directly. Sacred holdout: 2026-01-01 (Mode A).

**Thresholds (read-only):** Chordia strict t>=3.79 (no theory), power floor 0.5 (RULE 3.3). BH-FDR q=0.05 at K=family (the overlay conditions), NOT K=1992 selection budget.


## VERDICT: DEAD_FOR_ORB

pooled base flat (best base t=1.46 < 3.79) and no overlay adds powered residual edge -> MGC O30 long DEAD for ORB; stop slicing; SR-monitor signal-only shadow (Criterion 12) is the only honest path


## Q1a — Pooled base (all DOW)

### Pooled across ALL overlay sessions

| RR | N | ExpR | t | p | power | tier | dir_match (IS->OOS) |
|---|---|---|---|---|---|---|---|
| 1 | 4281 | -0.0161 | -1.28 | 0.2011 | 0.25 | STATISTICALLY_USELESS | False (IS -0.022 N=3944 -> OOS +0.054 N=337, OOS power 0.08/STATISTICALLY_USELESS) |
| 1.5 | 4281 | +0.0025 | 0.17 | 0.8673 | 0.05 | STATISTICALLY_USELESS | False (IS -0.007 N=3944 -> OOS +0.119 N=337, OOS power 0.05/STATISTICALLY_USELESS) |
| 2 | 4281 | +0.0245 | 1.41 | 0.1590 | 0.29 | STATISTICALLY_USELESS | True (IS +0.017 N=3944 -> OOS +0.115 N=337, OOS power 0.06/STATISTICALLY_USELESS) |

### Per-session base breakdown

| session | RR | N | ExpR | t | p | power | tier | dir_match |
|---|---|---|---|---|---|---|---|---|
| EUROPE_FLOW | 1 | 510 | -0.0252 | -0.66 | 0.5125 | 0.10 | STATISTICALLY_USELESS | False |
| EUROPE_FLOW | 1.5 | 510 | -0.0013 | -0.03 | 0.9782 | 0.05 | STATISTICALLY_USELESS | True |
| EUROPE_FLOW | 2 | 510 | +0.0528 | 0.93 | 0.3511 | 0.15 | STATISTICALLY_USELESS | False |
| LONDON_METALS | 1 | 506 | +0.0070 | 0.18 | 0.8572 | 0.05 | STATISTICALLY_USELESS | False |
| LONDON_METALS | 1.5 | 506 | -0.0169 | -0.35 | 0.7294 | 0.06 | STATISTICALLY_USELESS | True |
| LONDON_METALS | 2 | 506 | -0.0153 | -0.27 | 0.7860 | 0.06 | STATISTICALLY_USELESS | False |
| NYSE_OPEN | 1 | 499 | +0.0372 | 1.03 | 0.3029 | 0.18 | STATISTICALLY_USELESS | True |
| NYSE_OPEN | 1.5 | 499 | +0.0480 | 1.16 | 0.2475 | 0.21 | STATISTICALLY_USELESS | True |
| NYSE_OPEN | 2 | 499 | +0.0663 | 1.46 | 0.1444 | 0.31 | STATISTICALLY_USELESS | True |
| SINGAPORE_OPEN | 1 | 501 | -0.0258 | -0.65 | 0.5128 | 0.10 | STATISTICALLY_USELESS | False |
| SINGAPORE_OPEN | 1.5 | 501 | +0.0075 | 0.15 | 0.8780 | 0.05 | STATISTICALLY_USELESS | False |
| SINGAPORE_OPEN | 2 | 501 | +0.0309 | 0.54 | 0.5868 | 0.08 | STATISTICALLY_USELESS | True |
| US_DATA_830 | 1 | 507 | -0.0025 | -0.07 | 0.9467 | 0.05 | STATISTICALLY_USELESS | False |
| US_DATA_830 | 1.5 | 507 | +0.0210 | 0.47 | 0.6402 | 0.08 | STATISTICALLY_USELESS | False |
| US_DATA_830 | 2 | 507 | +0.0622 | 1.22 | 0.2227 | 0.23 | STATISTICALLY_USELESS | True |

### Per-year stability — strongest pooled cell (RR2)

| year | N | ExpR | t |
|---|---|---|---|
| 2022 | 551 | -0.0772 | -1.69 |
| 2023 | 1089 | -0.0741 | -2.32 |
| 2024 | 1121 | +0.0317 | 0.95 |
| 2025 | 1183 | +0.1299 | 3.68 |
| 2026 | 337 | +0.1154 | 1.68 |

## Q1b — Conditional overlays (RULE 2 Pass-1/Pass-2)

Pass-1 = condition lift (on vs off) on the full base universe for that session+RR. Pass-2 = the on-condition cell's own t/power (residual edge given the base). BH-FDR at K=family on the Pass-1 lift p-values.

| # | session | RR | condition | N_on | ExpR_on | N_off | ExpR_off | P1 lift | P1 t | P1 p | BH-pass | P2 t | P2 power | P2 tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | US_DATA_830 | 2 | day_of_week==1 | 100 | +0.3283 | 407 | -0.0032 | +0.3315 | 2.44 | 0.0160 | True | 2.64 | 0.74 | DIRECTIONAL_ONLY |
| 2 | NYSE_OPEN | 2 | day_of_week==3 | 103 | +0.2374 | 396 | +0.0219 | +0.2156 | 2.03 | 0.0441 | True | 2.56 | 0.72 | DIRECTIONAL_ONLY |
| 3 | NYSE_OPEN | 1 | day_of_week==3 | 103 | +0.1814 | 396 | -0.0003 | +0.1817 | 2.18 | 0.0309 | True | 2.50 | 0.70 | DIRECTIONAL_ONLY |
| 4 | SINGAPORE_OPEN | 2 | day_of_week==4 | 98 | +0.3118 | 403 | -0.0374 | +0.3492 | 2.42 | 0.0168 | True | 2.40 | 0.66 | DIRECTIONAL_ONLY |
| 5 | EUROPE_FLOW | 2 | atr_20_pct>=60 | 296 | +0.1738 | 214 | -0.1146 | +0.2884 | 2.57 | 0.0105 | True | 2.27 | 0.62 | DIRECTIONAL_ONLY |
| 6 | LONDON_METALS | 1.5 | overnight_range_pct>=80 | 124 | +0.2157 | 382 | -0.0924 | +0.3081 | 2.66 | 0.0084 | True | 2.12 | 0.56 | DIRECTIONAL_ONLY |

## Summary

- Pooled-finding flip rate (per-session vs pooled sign): 40.0% (>=25% — heterogeneity acknowledged)
- Overlays adding powered+BH-confirmed residual edge: 0/6
- **Disposition: DEAD_FOR_ORB** — pooled base flat (best base t=1.46 < 3.79) and no overlay adds powered residual edge -> MGC O30 long DEAD for ORB; stop slicing; SR-monitor signal-only shadow (Criterion 12) is the only honest path
- No threshold relaxed. No DB write. K=family (not K=1992). The CPCV audit's K=1992 was the discovery SELECTION budget; this confirmatory re-frame tests already-surfaced cells, so the honest family is the 6 overlay conditions.
- MGC O30 long path forward: SR-monitor signal-only shadow per `pre_registered_criteria.md` Criterion 12 (grounded `pepelyshev_polunchenko_2015_cusum_sr`). NOT a calendar wait, NOT a threshold relaxation. STOP slicing into thinner DOW windows.

## Reproduction / outputs

```bash
.venv/Scripts/python.exe research/mgc_o30_long_pooled_base.py
```

- Canonical layers only: `orb_outcomes JOIN daily_features` on `(trading_day, symbol, orb_minutes)` (RULE 9 triple-join safe). No DB writes.
- Costs from `pipeline.cost_model.COST_SPECS['MGC'].total_friction = 5.74`; `pnl_r` is already net, scored directly.
- Power tiers via `research.oos_power` (one-sample framing); BH-FDR via the family of 6 overlay conditions (K=family, NOT K=1992 selection budget).
- Sacred holdout `2026-01-01` (Mode A) per `trading_app/holdout_policy.py`.
- Tables above (pooled base, per-session breakdown, per-year RR2, conditional overlays) are the script's stdout, transcribed verbatim.

## Caveats / limitations / disconfirming evidence

- **Heterogeneity acknowledged:** per-session-vs-pooled sign flip rate is 40.0% (≥25% threshold) — the pooled base headline hides opposite-sign per-session behaviour; read the per-session table, not the pooled row alone.
- **Strongest pooled cell (RR2) is unstable across years:** 2022 t=−1.69, 2023 t=−2.32, then 2024 +0.95, 2025 +3.68, 2026 +1.68. The positive pooled drift is carried by 2025; pre-2024 is negative. This is regime-contingent, not a stable edge.
- **All 6 overlays land DIRECTIONAL_ONLY (power 0.56–0.74), none CAN_REFUTE:** their Pass-1 BH-pass + Pass-2 t≈2.1–2.6 is informational, not confirmatory. None reaches Chordia strict t≥3.79 or the 0.80 power floor — they cannot promote and cannot be called dead on this sample.
- **OOS is statistically useless** (N=337, power 0.05–0.08) — the IS→OOS dir_match outcomes are noise-consistent and carry no refutational weight (RULE 3.3).
- **Disconfirming the DEAD label:** the verdict is "DEAD_FOR_ORB" on the *pooled base* failing strict-t, not a claim that every cell is noise. The 6 overlays remain UNVERIFIED, not killed; the SR-monitor signal-only shadow exists precisely to catch a real residual edge these underpowered slices cannot confirm.
