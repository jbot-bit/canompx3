# PR #48 sizer-rule skeptical re-audit

Re-audit of PR #59 (`2026-04-21-pr48-sizer-rule-oos-backtest-v1.md`) per user directive "Stop. Did we evaluate this properly?" (2026-04-21).

Script: `research/pr48_sizer_rule_skeptical_reaudit_v1.py`. No capital action. PR #59 artifacts unchanged; this doc supersedes the interpretation ("deploy-eligible on MES + MGC") with a correction.

## Verdict - classification per instrument

| Instrument | PR #59 verdict | Re-audit verdict | Why changed |
|---|---|---|---|
| MNQ | SIZER_WEAK | **DEAD (as sizer)** | Pooled delta +0.006R, bootstrap 95% CI crosses zero, Spearman p=0.12 (insig), 55% lanes flip sign vs pooled. |
| MES | SIZER_ALIVE | **MISCLASSIFIED - real rank effect, wrong deployment form** | Sizer Sharpe is still NEGATIVE (-0.082 -> -0.050). Sizer "works" by losing slower. Filter-on-Q5 alternative yields +0.20R uplift vs uniform (vs sizer's +0.030R) on same OOS. |
| MGC | SIZER_ALIVE | **ALIVE but sub-optimal form - Q5-filter dominates sizer** | Pattern is real (Spearman p=0.002, +0.022 Sharpe uplift, monotonic quintiles), but Q5-only filter uplift = +0.19R (~6x bigger than +0.032R sizer delta). Deployment form should be filter, not linear sizer. Bootstrap CI lower bound +0.0004 (touching zero). |

## Right question? - NO

PR #59 asked "does the IS quintile-linear multiplier curve beat uniform OOS?" and answered "yes on MES/MGC." The **right** question was: "what is the highest-EV deployment form of the rel_vol rank signal?" Sizer was presumed because Carver Ch 10 continuous-sizer lens. Data says filter. Monotonicity diagnostic from PR #59 itself:

- MES quintiles: -0.199 / -0.176 / -0.083 / -0.147 / +0.112. Q1-Q4 all negative; only Q5 positive. Binary signal ("trade iff Q5"), not smooth gradient.
- MNQ quintiles: +0.016 / +0.035 / +0.064 / +0.142 / **+0.010**. Q5 crashes vs Q4. Inverted-U.
- MGC quintiles: -0.026 / -0.023 / +0.033 / +0.143 / +0.263. Only MGC is genuinely monotonic.

## Alternative framings - ROI table (same OOS -> directional, not confirmatory)

| Framing | MNQ | MES | MGC | Relative EV vs PR #59 sizer |
|---|---|---|---|---|
| (a) Linear SIZER (PR #59) | +0.006R delta | +0.030R delta | +0.032R delta | baseline |
| (b) Q5-only FILTER | +0.010R (N=137) | **+0.112R (N=155)** | **+0.263R (N=95)** | MES ~4x, MGC ~8x |
| (c) Q4+Q5 FILTER | +0.085R (N=320) | -0.011R (N=296) | **+0.194R (N=222)** | MGC ~6x; MES negative |
| (d) Conditioner (confluence gate) | untested | untested | untested | - |
| (e) Allocator (per-lane capital weight) | untested | untested | untested | - |

Best per-instrument framing: **MNQ Q4+Q5 filter**, **MES Q5-only filter**, **MGC Q5-only filter**. Sizer form (a) is dominated on every instrument with a positive finding.

## The 4 outputs

**Best opportunity:** MGC Q5-only filter. +0.263R per trade on N=95 OOS. Spearman p=0.002. Monotonic IS->OOS. Filter-form EV ~4x the PR #59 sizer delta.

**Biggest blocker:** Same-OOS contamination risk. Having now looked at Q4+Q5 / Q5 filter ExpR on this OOS, any filter-form pre-reg shipping tomorrow on this exact data is p-hacking. Honest path: filter-form pre-reg NOW but held as RESEARCH_SURVIVOR until ~50 fresh OOS trades accrue per instrument.

**Biggest miss:** Not testing alternative deployment forms BEFORE locking the sizer pre-reg. Anchored on Carver Ch 10 continuous-sizer lens when `docs/institutional/mechanism_priors.md` lists R1 FILTER ahead of R2 SIZER. Should pre-reg 2-3 deployment forms per hypothesis file.

**Next best test:** Filter-form pre-reg (Q5-only + Q4+Q5) IS-trained thresholds on **MGC + MES**, plus a Sharpe-positive gate (not just paired-t on delta). MNQ excluded (Spearman insig).

## PR #59 deliverables - corrected status

| PR #59 deliverable | Original status | Re-audit status |
|---|---|---|
| Pre-reg YAML | LOCKED | LOCKED (pre-reg honest within its declared scope) |
| Sizer backtest script | shipped | shipped (unchanged) |
| Result MD | SIZER_ALIVE on MES + MGC | stands as recorded; interpretation superseded |
| HANDOFF queue item #6 (shadow-deploy) | top priority | **CANCELLED** - sizer not deploy-eligible |

No writes to `validated_setups` / `edge_families` / `lane_allocation` / `live_config`. No capital action.

---

## Raw audit tables

## Headline with bootstrap CI + Sharpe

| Inst | N | Delta (R/trade) | Paired t | p | 95% CI (bootstrap) | SR uniform | SR sizer |
|---|---:|---:|---:|---:|---|---:|---:|
| MNQ | 771 | +0.00627 | +0.440 | 0.3302 | [-0.0217, +0.0345] | +0.050 | +0.052 |
| MES | 702 | +0.03025 | +2.084 | 0.0188 | [+0.0015, +0.0589] | -0.082 | -0.050 |
| MGC | 601 | +0.03175 | +2.000 | 0.0230 | [+0.0008, +0.0619] | +0.059 | +0.081 |

## Rank->pnl Spearman (does quintile rank predict OOS pnl_r?)

| Inst | Spearman rho | p |
|---|---:|---:|
| MNQ | +0.0566 | 0.1161 |
| MES | +0.1188 | 0.0016 |
| MGC | +0.1278 | 0.0017 |

## Alternative framings: filter Q4+Q5 only / Q5 only

| Inst | Q4+Q5 N | Q4+Q5 ExpR | Q5 N | Q5 ExpR |
|---|---:|---:|---:|---:|
| MNQ | 320 | +0.0851 | 137 | +0.0096 |
| MES | 296 | -0.0112 | 155 | +0.1123 |
| MGC | 222 | +0.1941 | 95 | +0.2626 |

## Per-direction breakdown (sizer forces long+short symmetry)

| Inst | Dir | N | Uniform ExpR | Sizer ExpR | Delta | t |
|---|---|---:|---:|---:|---:|---:|
| MNQ | long | 386 | +0.0987 | +0.1067 | +0.00799 | +0.403 |
| MNQ | short | 385 | +0.0191 | +0.0237 | +0.00455 | +0.222 |
| MES | long | 353 | -0.0457 | -0.0090 | +0.03678 | +1.846 |
| MES | short | 349 | -0.1352 | -0.1116 | +0.02365 | +1.118 |
| MGC | long | 342 | +0.0676 | +0.0937 | +0.02612 | +1.253 |
| MGC | short | 259 | +0.0721 | +0.1113 | +0.03918 | +1.600 |

## Per-lane heterogeneity (sessions x direction; >=20 OOS trades)

| Inst | Lane | N | Uniform | Sizer | Delta |
|---|---|---:|---:|---:|---:|
| MNQ | CME_PRECLOSE_long | 28 | -0.0795 | +0.1150 | +0.19450 |
| MNQ | BRISBANE_1025_short | 32 | +0.1469 | +0.2945 | +0.14764 |
| MNQ | LONDON_METALS_long | 41 | +0.0890 | +0.2316 | +0.14260 |
| MNQ | SINGAPORE_OPEN_short | 30 | +0.3267 | +0.4528 | +0.12616 |
| MNQ | TOKYO_OPEN_long | 36 | +0.2426 | +0.3360 | +0.09343 |
| MNQ | NYSE_OPEN_short | 40 | -0.0178 | +0.0431 | +0.06081 |
| MNQ | LONDON_METALS_short | 31 | -0.0042 | +0.0391 | +0.04324 |
| MNQ | COMEX_SETTLE_long | 32 | -0.1127 | -0.0699 | +0.04279 |
| MNQ | CME_PRECLOSE_short | 34 | -0.3204 | -0.2889 | +0.03150 |
| MNQ | US_DATA_1000_long | 35 | -0.0272 | +0.0003 | +0.02746 |
| MNQ | US_DATA_830_short | 38 | -0.1900 | -0.1950 | -0.00496 |
| MNQ | SINGAPORE_OPEN_long | 42 | +0.0953 | +0.0749 | -0.02042 |
| MNQ | US_DATA_1000_short | 33 | -0.0420 | -0.0659 | -0.02389 |
| MNQ | BRISBANE_1025_long | 40 | +0.0803 | +0.0544 | -0.02591 |
| MNQ | TOKYO_OPEN_short | 36 | +0.0553 | +0.0237 | -0.03163 |
| MNQ | COMEX_SETTLE_short | 34 | +0.1173 | +0.0801 | -0.03712 |
| MNQ | US_DATA_830_long | 32 | -0.1766 | -0.2303 | -0.05371 |
| MNQ | EUROPE_FLOW_long | 37 | +0.4562 | +0.3890 | -0.06722 |
| MNQ | EUROPE_FLOW_short | 35 | +0.1200 | +0.0388 | -0.08122 |
| MNQ | NYSE_OPEN_long | 28 | +0.2308 | +0.1337 | -0.09714 |
| MNQ | CME_REOPEN_long | 21 | +0.1345 | +0.0132 | -0.12134 |
| MNQ | CME_REOPEN_short | 27 | +0.0535 | -0.0939 | -0.14739 |
| MES | SINGAPORE_OPEN_short | 32 | -0.1337 | -0.0089 | +0.12480 |
| MES | LONDON_METALS_long | 39 | -0.1057 | +0.0007 | +0.10634 |
| MES | NYSE_OPEN_short | 37 | +0.0165 | +0.0859 | +0.06935 |
| MES | COMEX_SETTLE_short | 29 | +0.0008 | +0.0629 | +0.06211 |
| MES | LONDON_METALS_short | 33 | -0.0018 | +0.0554 | +0.05720 |
| MES | TOKYO_OPEN_long | 38 | +0.1447 | +0.1967 | +0.05192 |
| MES | CME_PRECLOSE_long | 28 | -0.1484 | -0.0988 | +0.04958 |
| MES | EUROPE_FLOW_long | 35 | -0.1224 | -0.0769 | +0.04555 |
| MES | US_DATA_1000_long | 28 | +0.1637 | +0.2076 | +0.04389 |
| MES | NYSE_OPEN_long | 33 | +0.0095 | +0.0517 | +0.04222 |
| MES | COMEX_SETTLE_long | 39 | -0.2029 | -0.1639 | +0.03906 |
| MES | CME_PRECLOSE_short | 30 | -0.2932 | -0.2544 | +0.03879 |
| MES | US_DATA_1000_short | 39 | -0.0995 | -0.0618 | +0.03769 |
| MES | TOKYO_OPEN_short | 34 | -0.1614 | -0.1362 | +0.02521 |
| MES | US_DATA_830_long | 34 | -0.1999 | -0.1831 | +0.01682 |
| MES | SINGAPORE_OPEN_long | 40 | +0.0452 | +0.0540 | +0.00880 |
| MES | US_DATA_830_short | 34 | -0.2119 | -0.2120 | -0.00017 |
| MES | CME_REOPEN_long | 25 | -0.0189 | -0.0388 | -0.01992 |
| MES | EUROPE_FLOW_short | 37 | -0.2543 | -0.3284 | -0.07407 |
| MES | CME_REOPEN_short | 28 | -0.2086 | -0.2961 | -0.08750 |
| MGC | US_DATA_1000_short | 31 | +0.0856 | +0.2309 | +0.14523 |
| MGC | NYSE_OPEN_short | 32 | +0.3530 | +0.4672 | +0.11427 |
| MGC | US_DATA_830_long | 36 | -0.1558 | -0.0501 | +0.10567 |
| MGC | EUROPE_FLOW_short | 25 | -0.0608 | +0.0041 | +0.06490 |
| MGC | TOKYO_OPEN_short | 31 | -0.0839 | -0.0190 | +0.06489 |
| MGC | EUROPE_FLOW_long | 46 | -0.0954 | -0.0360 | +0.05940 |
| MGC | CME_REOPEN_long | 26 | +0.3975 | +0.4541 | +0.05660 |
| MGC | COMEX_SETTLE_long | 34 | -0.1785 | -0.1296 | +0.04890 |
| MGC | LONDON_METALS_long | 39 | -0.0487 | -0.0061 | +0.04262 |
| MGC | NYSE_OPEN_long | 37 | +0.1693 | +0.1977 | +0.02836 |
| MGC | LONDON_METALS_short | 32 | +0.1640 | +0.1903 | +0.02630 |
| MGC | COMEX_SETTLE_short | 32 | -0.2749 | -0.2627 | +0.01220 |
| MGC | TOKYO_OPEN_long | 40 | +0.2409 | +0.2483 | +0.00743 |
| MGC | US_DATA_830_short | 33 | +0.1431 | +0.1310 | -0.01213 |
| MGC | SINGAPORE_OPEN_short | 25 | +0.2587 | +0.2265 | -0.03223 |
| MGC | SINGAPORE_OPEN_long | 45 | +0.4484 | +0.4112 | -0.03721 |
| MGC | US_DATA_1000_long | 39 | -0.1366 | -0.1897 | -0.05314 |

**MNQ** pooled delta=+0.00627; lanes pos=10, neg=12, negative-share=55% (>=25% = heterogeneity artefact per memory rule)
**MES** pooled delta=+0.03025; lanes pos=16, neg=4, negative-share=20% (>=25% = heterogeneity artefact per memory rule)
**MGC** pooled delta=+0.03175; lanes pos=13, neg=4, negative-share=24% (>=25% = heterogeneity artefact per memory rule)

## Jackknife leave-one-lane-out (is one lane carrying signal?)

| Inst | Min delta | Max delta | Pooled | Range |
|---|---:|---:|---:|---:|
| MNQ | -0.00138 | +0.01185 | +0.00627 | +0.01323 |
| MES | +0.02574 | +0.03606 | +0.03025 | +0.01032 |
| MGC | +0.02558 | +0.03764 | +0.03175 | +0.01206 |

## Leakage check (thresholds rebuilt on OOS itself vs IS-trained)

| Inst | IS-trained delta (PR #59) | OOS-trained delta (cheat) | Cheat t |
|---|---:|---:|---:|
| MNQ | +0.00627 | +0.00000 | +nan |
| MES | +0.03025 | +0.00000 | +nan |
| MGC | +0.03175 | +0.00000 | +nan |
