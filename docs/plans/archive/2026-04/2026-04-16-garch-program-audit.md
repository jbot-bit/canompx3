---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch Program Audit

**Date:** 2026-04-16  
**Status:** ACTIVE AUDIT  
**Purpose:** re-audit the full `garch` program at institutional level after the
allocator work exposed both a real deployment-use possibility and a real risk
of asking the wrong question in the wrong order.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/regime-and-rr-handling-framework.md`
- `docs/audit/results/2026-04-16-garch-regime-audit-synthesis.md`
- `docs/audit/results/2026-04-16-garch-structural-decomposition.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`

---

## 1. Executive read

The `garch` program is still worth pursuing, but only if it is treated as a
**state-family program**, not a single-filter or single-map program.

Current honest read:

1. `garch_forecast_vol_pct` is look-ahead clean and holdout-disciplined in the
   existing research scripts.
2. The broad surface is not random, but it is also not a clean universal
   deployment filter.
3. `garch` overlaps materially with other volatility-state proxies,
   especially `atr_20_pct`, but it is not identical to them.
4. The main unresolved question is not "does `garch` exist?" but:
   - what role should it play?
   - what additional state information makes it economically useful?
5. The program becomes wrong when it jumps too early into profile translation
   or treats deployment geometry as proof of market truth.

---

## 2. What `garch` currently looks like

Based on the completed audits:

### 2.1 What it is

- a pre-entry market-state variable
- a volatility-state proxy with real sign structure in natural families
- a plausible `R3` / `R7` / `R8` object:
  - size modifier
  - confluence / state score
  - allocator / routing input

### 2.2 What it is not yet

- a validated standalone signal edge
- a universal binary `take / skip` filter
- a proven deployment doctrine

### 2.3 What the raw evidence supports

From prior reruns:

- validated-scope broad overlay question:
  - `45` strategies
  - `429` tests
  - `0` BH survivors
- broad exact role run:
  - `430` rows
  - `2630` tests
  - `1` BH mean survivor, `0` Sharpe survivors
- family-framed reruns:
  - several natural regime families survive

So the right statement is:

> `garch` has regime-family structure, but broad production-overlay proof is
> not there.

---

## 3. Structural constraint

The current bottleneck is **not** simply multiple testing.

The real constraints are:

### C1 - weak directional specificity on its own

`garch` looks like a volatility-state measure, not a direct directional trigger.
That means it may improve:
- when to trust a setup
- which setup deserves allocation
- how aggressively to participate

More than it improves:
- whether to predict up or down from scratch

### C2 - overlap with adjacent vol proxies

The structural decomposition showed:

- `corr(garch_pct, atr_20_pct)` about `+0.71` to `+0.77`
- `corr(garch_pct, overnight_range_pct)` about `+0.29` to `+0.31`
- `corr(garch_pct, atr_vel_ratio)` about `+0.33` to `+0.38`

So a big constraint is:

> if `garch` is economically useful, it may be because it helps form a better
> **vol-state family** rather than because it is a unique alpha source.

### C3 - wrong economic question can kill valid uses

Examples already seen:

- A2 asked whether continuous state survives integer-contract translation in
  single-lane sizing; it did not
- A4a asked whether active-profile slot routing adds value; profile slot caps
  never bound, so the test was null by construction

This means:

> a negative result on the wrong surface is not the same as a negative result on
> the idea itself.

---

## 4. Could `garch` work with another signal?

Yes, but only in the correct role.

### 4.1 The correct pairing logic

`garch` is most plausible as a **state conditioner**, not the trade trigger
itself.

That means the clean pairing shape is:

- **direction / setup signal** supplies the reason to take a trade
- **garch / state family** supplies when that setup is likely better or worse

In repo terms:

- trade trigger candidates:
  - existing ORB breakout eligibility
  - validated structural filters
  - level / context filters from `mechanism_priors.md`
- state-conditioning candidates:
  - `garch_forecast_vol_pct`
  - `atr_20_pct`
  - `overnight_range_pct`
  - `atr_vel_ratio`

### 4.2 The wrong pairing logic

Wrong:
- "pair `garch` with random nearby filter and hope specificity appears"

Right:
- "pair `garch` with one mechanism-shaped partner that explains why the setup
  should be better in that state"

### 4.3 Plausible pair types

1. **Latent expansion**
   - high `garch`
   - lower / moderate overnight realized range
   - existing breakout trigger

2. **Active continuation**
   - high `garch`
   - high ATR
   - continuation-friendly session / setup

3. **Avoid hostile low-state**
   - low `garch`
   - setup still valid structurally
   - participation reduced or skipped

These are not random filter combinations. They are state + setup pairings.

---

## 5. What we should and should not broaden into

### 5.1 What should broaden now

The program should broaden into:

1. **state-family distinctness**
   - does `garch` still add value after controlling for ATR / overnight / ATR velocity?
2. **mechanism-specific pairing**
   - does one realistic state+setup family outperform the setup alone?
3. **binding-budget allocation**
   - once the value question is answered on a binding shelf

### 5.2 What should not broaden now

Do not broaden into:

- random filter stacking
- arbitrary new signals with no mechanism
- profile translation before utility is shown
- app defaults before shadow proof

---

## 6. Correct program ordering

The corrected institutional order is:

### Stage S1 - State distinctness audit

Question:
- what does `garch` add beyond nearby vol-state proxies?

Required checks:
- overlap / correlation
- sign persistence under proxy strata
- incremental-value tests, not just pairwise correlation

### Stage S2 - Mechanism pairing audit

Question:
- does `garch` improve a specific setup family when paired for a specific
  mechanism reason?

Required form:
- one mechanism family at a time
- one partner or composite at a time

### Stage S3 - Binding-shelf utility audit

Question:
- does the state family improve utility on a genuinely binding validated shelf?

This is where allocator value is tested honestly.

### Stage S4 - Profile translation

Question:
- if utility exists, how does it survive Topstep vs self-funded geometry?

Only here does profile become first-class.

### Stage S5 - Forward shadow

Question:
- does the surviving doctrine hold out of sample in live shadow?

---

## 7. Practical answer to the user's question

### "Should we be exploring other filter effects and correlations?"

Yes.

But the right object is:
- **state-family audit**

Not:
- "try every filter near `garch`"

### "Would `garch` potentially work as a signal with another signal?"

Yes, but mostly as:
- a **conditioner**
- a **confluence state**
- or an **allocator score**

Not yet as:
- an independent directional signal

### "What is our current constraint, and how do we overcome it?"

Current constraint:
- `garch` alone is too close to a general vol-state proxy to justify a broad
  production claim, and some tests so far have asked the wrong economic
  question

Correct way to overcome:
- prove distinctness or useful complementarity with adjacent state proxies
- prove mechanism-specific utility on the validated shelf
- then translate the survivor into deployment

---

## 8. Recommendation

The program is directionally correct **only if** it is reframed as:

> a vol-state family program in which `garch` is one serious input, not the
> whole story

So the next proper work is:

1. run a dedicated **state distinctness / incremental-value audit**
2. then run one **mechanism-shaped pairing audit**
3. only after that continue allocator translation

This is the best way to avoid:
- tunnel vision
- random filter mining
- deployment-first confusion
- false negative conclusions from the wrong test surface
