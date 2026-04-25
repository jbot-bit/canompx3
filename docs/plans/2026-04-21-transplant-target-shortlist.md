# Transplant Target Decision Sheet

**Date:** 2026-04-21  
**Requested by:** user shortlist request (repo/module transplant candidates)  
**Status:** decision-ready shortlist (top 3 identified)

---

## Scope and scoring method

This sheet shortlists external repo/module transplant targets for this codebase, with explicit scoring across:

1. **Data compatibility** (fit with `pandas`/`duckdb`/`pyarrow` canonical stack)
2. **Licensing** (compatibility with current project usage model)
3. **Testability** (ability to write deterministic local tests)
4. **Maintenance burden** (operational drag after adoption)
5. **Integration blast radius** (surface area touched in this repo)

### Scoring rubric

- Score each dimension on **1-5** (5 = best).
- Composite score is unweighted sum (max = 25).
- Maintenance burden and integration blast radius are scored as **"higher is safer/lower burden"**.

---

## Candidate universe (initial pass)

1. `unionai-oss/pandera`
2. `great-expectations/great_expectations`
3. `dagster-io/dagster`
4. `polakowo/vectorbt`
5. `kernc/backtesting.py`
6. `mementum/backtrader`

These were selected as realistic transplant classes for this repo's current architecture:
- data-quality contracts (Pandera / Great Expectations)
- orchestration envelope (Dagster)
- strategy/backtest engines (vectorbt / backtesting.py / backtrader)

---

## Evidence snapshot (2026-04-21)

- **Pandera**: MIT license, active releases (`v0.31.1`, Apr 15, 2026), test directory present, pandas support explicit in README.  
- **Great Expectations**: Apache-2.0 license, active releases (`1.16.1`, Apr 15, 2026), mature test surface.  
- **Dagster**: Apache-2.0 license, active releases (`1.13.1 core`, Apr 17, 2026), large/active ecosystem.  
- **VectorBT OSS**: Apache 2.0 **with Commons Clause** (commercial-use restriction risk for some deployment models).  
- **backtesting.py**: AGPL-3.0 license (strong copyleft risk).  
- **backtrader**: GPL-3.0 license (strong copyleft risk).

---

## Scoring table

| Candidate | Data compat | Licensing | Testability | Maint. burden | Blast radius | Total (/25) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **Pandera** | 5 | 5 | 5 | 4 | 5 | **24** | Direct fit for `daily_features` / `orb_outcomes` contracts; minimal code disturbance. |
| **Great Expectations** | 4 | 5 | 4 | 3 | 3 | **19** | Strong data-quality framework; heavier scaffolding and runtime overhead than Pandera. |
| **Dagster** | 3 | 5 | 4 | 2 | 2 | **16** | Useful orchestrator, but high integration surface vs current script-first pipeline. |
| VectorBT OSS | 3 | 2 | 3 | 3 | 2 | 13 | License terms (Commons Clause) make policy/legal fit less clean than permissive OSS options. |
| backtesting.py | 3 | 1 | 4 | 3 | 2 | 13 | AGPL obligations create high adoption risk for mixed/private workflows. |
| backtrader | 2 | 1 | 3 | 2 | 2 | 10 | GPL-3.0 + older architecture; high migration cost from current canonical layers. |

---

## Top 3 transplant targets (decision)

## 1) Pandera (Primary target)

**Why it made top 1:**
- Highest compatibility with current canonical data layer (`pandas`/`duckdb` outputs).
- Lowest blast radius: can be introduced as contract checks around existing pipeline boundaries.
- Strong fit with repo fail-closed doctrine: schema validation can become hard gates in build/validation flows.

**Recommended transplant shape:**
- Add schema contracts for high-value tables (`daily_features`, `orb_outcomes`, selected read models).
- Wire checks into existing verification/audit commands before broad rollout.

## 2) Great Expectations (Secondary target)

**Why it made top 2:**
- Strong validation ecosystem and Apache-2.0 license.
- Good option for richer expectation suites, docs, and batch validations.

**Why not top 1:**
- Heavier framework footprint than Pandera for this repo's current architecture.
- Higher maintenance burden and integration complexity if introduced as a broad platform.

**Recommended transplant shape:**
- Pilot on one canonical dataset slice only.
- Keep as optional validation layer until operational payoff is proven.

## 3) Dagster (Conditional target)

**Why it made top 3:**
- Very active, well-maintained orchestration platform.
- Potential long-term fit if pipeline orchestration and observability requirements expand materially.

**Why it is conditional:**
- Largest blast radius and migration burden among viable candidates.
- Current repo already has script-centric workflow + guardrails; replacing orchestration now would be a strategic change, not a tactical transplant.

**Recommended transplant shape:**
- Treat as Phase-2/3 orchestration option, not immediate adoption.
- Evaluate only after incremental contract hardening (Pandera path) is stable.

---

## Explicit non-targets from this pass

- **vectorbt OSS**: license terms are less clean than MIT/Apache alternatives for broad internal/external usage patterns.
- **backtesting.py**: AGPL license risk.
- **backtrader**: GPL license risk plus larger migration friction.

---

## Project alignment notes (cross-repo consistency)

To align with existing project doctrine (`fail-closed`, canonical data truth, audit-first):

1. Start with **contract validation at canonical boundaries**, not strategy-engine replacement.
2. Prefer **low-blast incremental transplants** before platform-level rewrites.
3. Keep integration bound to existing command paths and verification gates.
4. Avoid introducing a second truth layer or bypassing canonical tables.

This keeps transplant decisions consistent with current architecture, governance, and verification posture.
