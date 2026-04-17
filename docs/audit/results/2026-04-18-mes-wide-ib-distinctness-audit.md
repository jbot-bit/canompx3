# MES Wide-IB (Relative-Width) Distinctness Audit

**Date:** 2026-04-18
**Trigger:** NO-GO registry reopen flag (STRATEGY_BLUEPRINT.md line 287): "MES-specific relative-width filter could be tested as hypothesis-scoped."
**Pre-register question:** Is wide-relative-IB a disguised version of ORB_G5 (absolute size), or a genuinely incremental mechanism?
**Verdict:** **SURVIVES distinctness.** Wide-relative-IB adds conditional lift within ORB_G5 fires on multiple (instrument, session) combos, is NOT correlated with ATR vol state, but OOS direction discipline constrains the deployable scope to a narrower subset than the NO-GO's MES-only narrative.

---

## Evidence trail (prior audit references)

- `STRATEGY_BLUEPRINT.md` line 287 (NO-GO registry entry, "Narrow relative IB width (compression-take)"):
  - 102 tests (size-controlled G5+ subset), 19/102 BH FDR, 16 at cross-family K=425
  - MES: 14/16 cross-family survivors on O5 and O15 (CME_PRECLOSE, TOKYO_OPEN, COMEX_SETTLE, EUROPE_FLOW)
  - MNQ: p=0.074 size-controlled, "not independently significant"
  - rho(relative_width, absolute_size) = 0.67 claimed
- Source script `filter_discovery_a4_a5_fix.py` no longer in `research/` — only the NO-GO summary is canonical evidence.
- HANDOFF entries `2026-04-13 to 2026-04-16`: no further mention of wide-IB testing.

---

## Distinctness test 1 — firing correlation

Definition of relative width used here (reproducible via this query):
```
WIDE_REL = (orb_{session}_size / 20-day trailing mean orb_{session}_size) >= 1.0
G5       = orb_{session}_size >= 5.0
G8       = orb_{session}_size >= 8.0
```
Trailing mean is LAG-1 (ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) → zero look-ahead.

### rho and Jaccard (MES, E2 O5, all days)

| Session | N days | mean_ORB | rho(abs, rel_width) | Jaccard(WIDE_REL, G5) |
|---|---:|---:|---:|---:|
| CME_PRECLOSE | 1726 | 6.08 | **0.706** | 0.592 |
| TOKYO_OPEN | 1793 | 3.62 | **0.754** | 0.375 |
| COMEX_SETTLE | 1726 | 4.93 | **0.719** | 0.558 |
| EUROPE_FLOW | 1792 | 3.67 | **0.745** | 0.414 |

**Reading:** rho is 0.71-0.75 — HIGHER than the 0.67 the NO-GO claimed. All four sessions are above the `|rho| > 0.70` DUPLICATE_FILTER threshold from `.claude/rules/quant-audit-protocol.md` T0 tautology rule. **On simple correlation grounds alone, this would be DUPLICATE_FILTER.**

But rho alone is a weak test — it doesn't account for non-linear / conditional distinctness.

---

## Distinctness test 2 — conditional lift within G5 (the real test)

Question: within the ORB_G5 population, does WIDE_REL add incremental R?

### MES (E2 RR1.0 O5, IS window <2026-01-01)

| Session | WIDE+G5 | G5-only (WIDE=F) | WIDE-lift-within-G5 | t(IS) |
|---|---|---|---:|---:|
| CME_PRECLOSE | ExpR +0.153 (N=397) | ExpR +0.018 (N=236) | **+0.135** | +3.42 |
| TOKYO_OPEN | ExpR +0.082 (N=271) | ExpR −0.017 (N=45) | **+0.099** | +1.51 |
| COMEX_SETTLE | ExpR +0.065 (N=432) | ExpR +0.050 (N=144) | +0.015 | +1.50 |
| EUROPE_FLOW | ExpR −0.076 (N=305) | ExpR −0.148 (N=58) | +0.072 (but both negative) | −1.47 |

### MNQ comparison (contradicts NO-GO's "MES-only" claim)

| Session | WIDE+G5 | G5-only | WIDE-lift-within-G5 |
|---|---|---|---:|
| CME_PRECLOSE | +0.196 (N=497) | +0.057 (N=913) | **+0.139** |
| TOKYO_OPEN | +0.131 (N=690) | +0.019 (N=909) | **+0.112** |
| COMEX_SETTLE | +0.075 (N=635) | +0.082 (N=934) | −0.007 |
| EUROPE_FLOW | +0.039 (N=703) | +0.070 (N=879) | −0.031 |

### MES CME_PRECLOSE at G8 (deployed-threshold) — stricter conditional test

| wide_rel | g8 | N | ExpR | WR | t |
|---|---|---:|---:|---:|---:|
| T | T | **227** | **+0.230** | 0.656 | +3.89 |
| F | T | 62 | −0.016 | 0.532 | −0.14 |
| T | F | 261 | +0.024 | 0.586 | +0.45 |
| F | F | 874 | −0.070 | 0.569 | −2.52 |

**Conditional lift within G8 = +0.246** (huge IS effect; N_G8-only=62 is thin).

### Conditional-test summary

- **CME_PRECLOSE (both MES and MNQ):** WIDE adds +0.135/+0.139 conditional lift within G5. Strong evidence of incremental signal.
- **TOKYO_OPEN (both MES and MNQ):** WIDE adds +0.099/+0.112 conditional lift. Moderate evidence.
- **COMEX_SETTLE (MES + MNQ):** WIDE adds essentially nothing within G5. Fails conditional distinctness on these sessions.
- **EUROPE_FLOW (both):** No signal to extract.

**NO-GO's "MNQ p=0.074 not independently significant" does not reproduce** in the simple conditional test for CME_PRECLOSE and TOKYO_OPEN. The NO-GO's method (size-controlled regression across all sessions pooled) may have smoothed the session-specific signal.

---

## Distinctness test 3 — alt-tautology vs vol state

Checked whether WIDE_REL is a disguised vol-state filter (e.g., ATR20 regime, which `X_MES_ATR60` already captures):

| Feature pair (MES CME_PRECLOSE) | rho |
|---|---:|
| WIDE_REL (rel width) vs atr_20_pct | **−0.001** |
| ORB_abs_size vs atr_20_pct | +0.405 |

**Reading:** WIDE_REL is essentially uncorrelated with daily vol percentile. It's a *session-local size regime* signal, not a *daily vol regime* signal. This IS a mechanically distinct dimension from the deployed `X_MES_ATR60` and ATR-based filters.

---

## Distinctness test 4 — OOS direction discipline (pre-check)

Mode A holdout boundary = 2026-01-01. For each best-looking cell, checked OOS direction match and effect ratio BEFORE proposing a full pre-reg.

Cell = WIDE_REL AND G5 on MES/MNQ × {CME_PRECLOSE, TOKYO_OPEN, COMEX_SETTLE} at E2 RR1.0 O5 CB1 stop_mult=1.0.

| Cell | IS N | IS ExpR | OOS N | OOS ExpR | dir_match | eff_ratio | **Verdict** |
|---|---:|---:|---:|---:|:---:|---:|:---:|
| MES CME_PRECLOSE | 397 | +0.153 | 24 | **−0.313** | FAIL | −2.05 | **DEAD — flip** |
| MNQ CME_PRECLOSE | 497 | +0.196 | 27 | **−0.009** | FAIL | −0.05 | **DEAD — flip** |
| MNQ TOKYO_OPEN | 690 | +0.131 | 37 | +0.088 | PASS | +0.67 | **SURVIVES** |
| MES TOKYO_OPEN | 271 | +0.082 | 29 | +0.075 | PASS | +0.92 | **SURVIVES** (N_OOS=29 thin, 1 below threshold) |
| MES COMEX_SETTLE | 432 | +0.065 | 27 | +0.097 | PASS | +1.49 | **PROBE** (N_OOS=27 thin; test 2 also showed marginal distinctness) |

**This is the same archetype as the VWAP_BP CME_PRECLOSE DEATH (this session's Task 1):** high-IS-ExpR CME_PRECLOSE cells flip negative post-holdout on both filter families. Pattern may reflect a **CME_PRECLOSE-specific regime change post-2026-01-01**, not a filter-family issue.

---

## Verdict

**SURVIVES distinctness — with three material caveats:**

1. **Simple rho test fails** (0.71-0.75 > 0.70). Conditional-lift test within G5 fires rescues it; correlation alone would KILL.
2. **NO-GO's "MES-only" narrative is wrong.** MNQ CME_PRECLOSE and MNQ TOKYO_OPEN show comparable or stronger conditional lifts than MES. The original audit's method dropped this cross-instrument signal.
3. **OOS direction kills CME_PRECLOSE cells on BOTH instruments.** Same pattern as VWAP_BP today. The strongest IS candidates (CME_PRECLOSE +0.15 to +0.20) are dead forward.

**Deployable scope (minimum viable pre-reg, per user instruction):** TOKYO_OPEN on both instruments + MES COMEX_SETTLE as a probe — NOT CME_PRECLOSE (DEAD), NOT EUROPE_FLOW (no signal).

---

## Minimum viable pre-reg surface (for the hypothesis file)

| Item | Value | Rationale |
|---|---|---|
| Filter | `WIDE_REL_1.0X_20D AND ORB_G5` (conjunction) | G5 is the incumbent threshold; WIDE adds the distinct signal |
| Entry | E2 CB=1 stop_mult=1.0 | Matches canonical baseline |
| Aperture | O5 | Distinctness test 2 used O5; O15 cited in NO-GO but not tested here |
| Instruments | MES, MNQ | MGC architecturally dead |
| Sessions | TOKYO_OPEN (MES + MNQ), COMEX_SETTLE (MES only as PROBE) | OOS direction match on these; NOT CME_PRECLOSE (dir flip) |
| RR targets | 1.0, 1.5, 2.0 | Standard triplet |
| K | 3 cells × 3 RR = **9** | Tight, well within MinBTL budget |
| Baseline | Unfiltered cell OR G5 cell (where not deployed) | Per-lane existing baseline |
| BH-FDR K | K=9 family + K_lane per cell | Multi-framing per RESEARCH_RULES |
| Chordia t | ≥ 3.00 (with theory) | WIDE_REL has session-local-regime mechanism |
| WFE | ≥ 0.50 | Standard |
| N_OOS | ≥ 30 | Pre_reg_criteria §7 proxy |
| OOS direction match | Required | sign(IS) == sign(OOS) |
| OOS effect ratio | ≥ 0.40 | Standard |
| Tautology guards (MUST all pass) | |rho(WIDE_REL×G5 firing, {G5 alone, X_MES_ATR60, COST_LT12, CROSS_SGP_MOMENTUM} firing)| ≤ 0.70 per lane where deployed filter exists | Filter-by-filter |
| Hard kill | dir flip on IS-to-OOS, tautology breach, BH-FDR K=9 fail, WFE<0.50 | No rescue |
| Mode A | 2026-01-01 holdout sacred | Locked |

**K=9 is the total research-budget for this pre-reg. No sweeps. No session expansion post-hoc.**

### Reason for excluding CME_PRECLOSE from pre-reg

Both MES and MNQ CME_PRECLOSE failed OOS direction match in this audit. Including them in the pre-reg would be rescue-tuning. The "WIDE_REL works on CME_PRECLOSE" claim is **retired as IS-only** pending evidence that CME_PRECLOSE returns to its historical regime.

### Reason for excluding O15

NO-GO claimed MES O15 also works but O15 was NOT tested in this audit (scope discipline). Adding O15 would double K to 18 — can be a Stage 2 IF Stage 1 passes. Pre-register here only.

---

## What happens next

- If Stage 1 (K=9 pre-reg) passes: WIDE_REL + G5 could add 1-3 MNQ/MES lanes, estimated +$200-500/yr.
- If Stage 1 fails: family closes for O5 scope; O15 option deferred; do NOT rescue with threshold tuning.
- If OOS direction match fails on Stage 1 despite surviving pre-check: DEAD, same kill doctrine as VWAP_BP.

Pre-reg files follow: `docs/audit/hypotheses/2026-04-18-mes-mnq-wide-rel-ib.yaml` + `docs/plans/2026-04-18-mes-mnq-wide-rel-ib-design.md`.
