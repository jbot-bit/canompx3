# Carry Encoding Exploration — Design

**Date:** 2026-04-16
**Status:** LOCKED DESIGN (pre-registered before execution)
**Predecessor:** W2e prior-session carry audit (binary gate DEAD)
**Hypothesis file:** `docs/audit/hypotheses/2026-04-16-carry-encoding-exploration.yaml`

---

## 1. Why this study exists

W2e tested the strongest form of the carry hypothesis — binary carry gating
conditioned by garch_high — and found it dead. But the collinearity check
revealed that carry and garch are almost perfectly orthogonal (corr = +0.016).
Independent information likely exists; binary encoding could not access it.

The failure was **encoding degeneracy**, not absence of information:
- Late-day sessions (COMEX_SETTLE 99.8%, EUROPE_FLOW 96.0%): "has a prior win"
  is nearly constant → no variance → no signal.
- Early-day sessions (TOKYO_OPEN 8.4%): almost no priors exist → feature absent.

This study asks: **can continuous carry encodings access the independent
information that binary gating could not?**

---

## 2. Why orthogonality alone is not enough

Two features can be perfectly independent and both still useless. Example:
a random number and garch_high have corr ≈ 0, but the random number adds
nothing. The carry-garch orthogonality finding (corr = +0.016) means carry
is NOT a garch echo — but it does NOT mean carry contains usable signal.

The orthogonality finding opens the DOOR (carry may add independent
information). The encoding study TESTS whether anything useful comes through
that door.

The specific test: does target-session pnl_r vary monotonically with the
continuous carry encoding? If yes, the information is real and accessible.
If no, the information is either not there or not capturable from price data.

---

## 3. What this study does NOT do

- Does NOT reopen binary carry gates (DEAD per W2e).
- Does NOT test carry as a standalone deployment signal.
- Does NOT make deployment or allocator claims.
- Does NOT test more than 3 encodings (K budget = 18, encoding-level K = 3).
- Does NOT tune thresholds after seeing results.
- Does NOT use 2026 OOS for promotion — only for directional check.

---

## 4. Ranked candidate encodings

### Rank 1: E1 — most_recent_prior_pnl_r

The simplest possible continuous encoding. The pnl_r of the single most
recently resolved prior session trade. Continuous [-1, +1]. Changes with
each new session resolution, so never constant on late-day sessions.
NULL on early-day sessions where no prior exists — honestly missing.

**Why rank 1:** simplest, fewest degrees of freedom, most interpretable.
If E1 doesn't work, more complex encodings are unlikely to help (Occam).

**Best role:** R7 confluence input or R3 sizing modifier.

### Rank 2: E2 — recency_weighted_carry_score

Weighted sum of all resolved prior session pnl_r values, with exponential
time-decay (half-life = 2 hours, pre-committed). Uses more information than
E1 by incorporating ALL priors, not just the most recent. The time-weighting
prevents late-day accumulation degeneracy.

**Why rank 2:** uses more data, but adds one free parameter (lambda).
Sensitivity check at ±50% lambda is mandatory. If E2 ≈ E1 in the results,
the extra complexity is not justified.

**Best role:** R7 confluence input or R8 portfolio context.

### Rank 3: E3 — direction_aware_carry_intensity

E1 multiplied by a direction-alignment sign: +1 if the prior session's
break_dir matches the target's, -1 if opposed. This tests whether the
DIRECTION of prior momentum matters, not just its magnitude.

**Why rank 3:** adds a conditioning dimension (direction) that W2e already
tested in binary form and found dead for veto. E3 is the continuous version
of the same direction question. If E3's negative tail shows strong signal
despite W2e's veto-pair death, that would be a genuine new finding. If it
doesn't, direction conditioning adds nothing beyond E1.

**Best role:** R7 confluence input (directional context) or R3 sizing modifier.

---

## 5. Test protocol (per encoding)

1. **Coverage:** what fraction of target rows have a non-NULL value?
   Kill if < 30%.
2. **Quintile split:** equal-count quintiles within IS (pre-2026). Report
   ExpR + WR per quintile.
3. **Monotonicity:** Spearman rank-corr between quintile rank and ExpR.
   Pass if |rho| >= 0.80 (4/5 or 5/5 quintiles monotonic).
4. **WR check:** WR spread across extreme quintiles >=5%? If WR flat but
   ExpR moves → ARITHMETIC_ONLY.
5. **Garch interaction:** repeat quintile split within garch_high and
   garch_low subsets. Does monotonicity strengthen in garch_high?
6. **OOS direction:** does the sign of monotonicity hold in 2026?
   Descriptive only, not a promotion gate.
7. **Sensitivity (E2 only):** repeat at half-life = 1h and 4h.

---

## 6. Session-group design

| Group | Sessions | Why separate |
|---|---|---|
| late_day | COMEX_SETTLE | Many priors (10+), binary carry was 99.8% constant. Continuous encoding has the most variance here. |
| mid_day | EUROPE_FLOW, SINGAPORE_OPEN | Moderate priors (3-8), binary carry was 76-96%. Mixed regime. |
| early_day | TOKYO_OPEN | Few priors (0-1), binary carry was 8.4%. E1/E2/E3 will be mostly NULL — report honestly. |

---

## 7. Promotion and kill gates

**Promote to dedicated R7 confluence study if:**
- At least 1 of 3 encodings passes quintile monotonicity at BH-FDR K=3
- WR moves (not ARITHMETIC_ONLY)
- Coverage >= 50% on at least one session group
- Garch_high subset direction matches or strengthens

**Park carry family entirely if:**
- All 3 encodings fail monotonicity
- Or all pass but are ARITHMETIC_ONLY
- Or coverage < 30% everywhere

**Grey zone (signal exists but weak):**
- 1 encoding passes but at borderline significance
- → file as watchlist, do not promote to implementation

---

## 8. Implementation path after this study

If any encoding passes:

1. Pre-register a dedicated R7 confluence input study.
2. Use the passing encoding(s) as one component of a Carver-style forecast
   combiner, alongside garch_forecast_vol_pct and deployed filter state.
3. Scope: single highest-N validated family, K ≤ 10.
4. R7 confluence does not change trade selection — every trade is still
   taken. It only attaches a score. This is the safest implementation
   class because it cannot reduce N or create selection bias.

If all encodings fail:

1. Park the carry family entirely.
2. Update `docs/STRATEGY_BLUEPRINT.md` NO-GO registry.
3. Do not revisit without a structurally different carry definition.
