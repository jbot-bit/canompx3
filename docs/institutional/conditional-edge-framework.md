# Conditional Edge Framework

**Date:** 2026-04-22
**Authority:** complements `RESEARCH_RULES.md`, `docs/STRATEGY_BLUEPRINT.md`, `docs/institutional/mechanism_priors.md`, and `docs/institutional/pre_registered_criteria.md`.
**Purpose:** stop the project from forcing every finding into the wrong question. Most real intraday edges in this repo are conditional. This doc defines how to classify, test, and promote them without confusing `selected-trade mean`, `policy EV`, `portfolio value`, and `standalone viability`.

---

## 1. Core claim

For this project, most useful findings are **conditional state variables**, not standalone strategies.

That is not a weakness. It is the normal shape of market microstructure research:

- volatility and participation variables often work as `filters` or `allocators`
- level geometry often works as `conditioners` or `confluence`
- some variables reduce losses without creating a positive standalone expectancy
- some variables improve the book only after sizing or portfolio integration

The mistake is not "finding conditional signals". The mistake is **testing the wrong role**.

---

## 2. The role taxonomy

Every new finding must be classified before statistical judgement.

| Role | Question | Typical metric | Promotion target |
|---|---|---|---|
| `standalone` | Is this itself a tradeable lane? | `selected_trade_mean`, full validation stack | new strategy candidate |
| `filter` | Does skipping bad states improve the parent? | `policy_ev_per_opportunity`, drawdown reduction | binary take/skip overlay |
| `conditioner` | Does this state tell us when the parent works? | subset vs complement delta, regime stability | context overlay / shadow rule |
| `allocator` | Should capital be scaled up/down by this state? | capital-normalized EV, weighted policy EV | sizer / weight map |
| `confluence` | Does the overlap of signals hold more edge than either alone? | overlap vs parent and component deltas | narrow sleeve / high-conviction arm |
| `execution` | Does this state change entry, stop, or target behavior? | implementation lift net of friction | execution modifier |

**Default rule:** if the mechanism acts on participation, friction, congestion, or location, assume `filter`, `conditioner`, or `allocator` first. Do **not** default to `standalone`.

---

## 3. Evaluation order

When a finding is conditional, the tests must follow this order:

1. **Mechanism check**
   Confirm the variable is pre-trade knowable and has a structural story grounded in repo rules or local literature.

2. **Role declaration**
   State the intended role before running the test:
   `standalone`, `filter`, `conditioner`, `allocator`, `confluence`, or `execution`.

3. **Parent declaration**
   Define the exact parent population the condition will act on.
   If the parent is vague, the test is vague.

4. **Comparator declaration**
   Define the exact comparison:
   `parent vs filtered parent`, `subset vs complement`, `binary filter vs continuous sizer`, or `overlap vs component`.

5. **Primary metric declaration**
   Match metric to role:
   - `standalone`: `selected_trade_mean_r`, `Sharpe`, full 12 criteria
   - `filter`: `policy_ev_per_opportunity_r`, loss removal, drawdown
   - `conditioner`: subset/complement delta, sign stability, year stability
   - `allocator`: weighted policy EV, capital efficiency, portfolio contribution
   - `confluence`: overlap delta vs parent and vs each leg
   - `execution`: implementation lift net of realistic friction

6. **IS freeze**
   Thresholds, breakpoints, and weighting rules are frozen on IS only.

7. **Sacred OOS application**
   Apply the frozen rule to `2026-01-01` onward without tuning. Thin OOS can monitor direction and implementation sanity, but it cannot be used to re-specify the rule.

---

## 4. What counts as success

Conditional findings are promoted only when they clear the role-appropriate bar.

### `standalone`
- must beat zero after costs
- must clear the normal validation stack
- failure here does **not** kill the finding as a filter or allocator

### `filter`
- must improve `policy_ev_per_opportunity_r` or materially reduce drawdown versus the parent
- selected-trade mean alone is insufficient

### `conditioner`
- must show a stable subset vs complement spread
- if the parent remains negative, this may still be useful as a context map, but not as a promoted lane

### `allocator`
- must improve weighted policy EV or portfolio EV under frozen sizing
- must be compared against the binary filter alternative

### `confluence`
- must beat the parent and the component signals, not just look good in isolation
- tiny overlap cells with no nearby support are fragile until proven otherwise

### `execution`
- must use realistic trigger timing, costs, and while-open constraints
- post-resolution simulations do not count

---

## 5. Failure modes this framework is meant to stop

This repo has repeatedly risked the same five mistakes:

1. **Standalone-or-dead framing**
   A real context variable gets discarded because it is not a trade by itself.

2. **Selected-trade mean confusion**
   A filter looks good because the remaining trades are better, but policy EV per opportunity falls.

3. **OOS-thin overreaction**
   A 3 to 4 month holdout is treated as closure when it is only a monitor.

4. **One-cell hero trade bias**
   A tiny overlap cell is over-promoted before testing the broader role.

5. **Role drift**
   A finding is discovered as a conditioner, described as a filter, and promoted as a standalone edge.

---

## 6. Project-level rules

### Rule 1 — every prereg declares the role

Every new hypothesis file must say whether it is testing a `standalone`, `filter`, `conditioner`, `allocator`, `confluence`, or `execution` question.

### Rule 2 — every conditional prereg declares parent + comparator

A conditional study is invalid without:

- the exact parent population
- the exact comparator
- the primary decision metric

### Rule 3 — role decides the metric

Do not judge a filter by standalone mean expectancy.  
Do not judge an allocator by binary trade count.  
Do not judge a confluence cell without comparing it to its legs.

### Rule 4 — OOS stays sacred

The holdout from `2026-01-01` onward is still sacred. It is used to apply frozen rules, not to pick them.

### Rule 5 — promotion language must match the role

Allowed:
- `deployable filter candidate`
- `shadow sizer candidate`
- `portfolio-only conditioner`
- `context map, not standalone`

Not allowed:
- `confirmed edge` when only a conditional role has been tested
- `dead` when only the standalone framing failed

---

## 7. Research and promotion workflow

1. Write prereg with `research_question_type`.
2. Declare the role on every hypothesis.
3. Freeze thresholds or weight maps on IS.
4. Run the canonical script on `bars_1m`, `daily_features`, `orb_outcomes` only.
5. Report both:
   - `selected trade quality`
   - `policy / portfolio value`
6. Apply the frozen rule to sacred OOS unchanged.
7. Promote only to the role actually tested.

---

## 8. Local grounding

This framework is consistent with local repo doctrine and local literature:

- `RESEARCH_RULES.md`
  theory-first, no data snooping, no standalone claims from weak samples
- `docs/institutional/mechanism_priors.md`
  already defines multiple deployment roles `R1` through `R8`
- `docs/institutional/edge-finding-playbook.md`
  already warns that single-factor and portfolio-modifier roles can matter more than new standalone lanes
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`
  theories first, backtests second
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`
  bounded trial budgets
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
  sizing is a first-class implementation layer, not an afterthought
- `docs/institutional/literature/carver_2015_ch11_portfolios.md`
  portfolio contribution matters, not just isolated trade stats

---

## 9. Default interpretation map for this repo

Use this unless a prereg says otherwise:

- `rel_vol`, `orb_volume`, participation shape`
  start as `filter` or `allocator`
- `prior-day levels`, `pivot proximity`, `displacement`
  start as `conditioner` or `confluence`
- `cost / ORB size / friction variables`
  start as `filter`
- `timing / break-sequence / stop-target geometry`
  start as `execution`

Only promote to `standalone` when the variable itself defines a complete trade population and survives that stricter question.

---

## 10. Immediate application

The first bounded implementation study under this framework is:

- `PR48 participation-shape`
- question type: `conditional_role`
- roles compared:
  - `parent`
  - `Q4+Q5 filter`
  - `Q5 filter`
  - `continuous quintile sizer`
- instruments:
  - `MNQ` as shadow / comparator
  - `MES`, `MGC` as promotion-eligible arms

That study is pre-registered separately. This doc governs how to interpret it.
