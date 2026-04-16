# Garch A4b Binding-Budget Design

**Date:** 2026-04-16  
**Status:** ACTIVE NEXT DESIGN  
**Purpose:** replace the null-by-construction `A4a` active-profile routing test
with a corrected allocator design that answers utility first on a genuinely
binding scarce-resource surface.

**Authority chain:**
- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`
- `docs/plans/2026-04-16-garch-a4-portfolio-ranking-design.md`
- `docs/audit/results/2026-04-16-garch-a4-portfolio-ranking-replay-topstep-50k-mnq-auto.md`
- `docs/audit/results/2026-04-16-garch-a4-portfolio-ranking-replay-self-funded-tradovate.md`

---

## 1. Why A4a is demoted

`A4a` used the **active profile book** as the allocator universe and the
profile slot cap as the scarce-resource budget.

Raw result:
- `topstep_50k_mnq_auto`: `6` lanes, slot budget `7`, max eligible/day `6`,
  collision days `0 / 1789`
- `self_funded_tradovate`: `10` lanes, slot budget `10`, max eligible/day `10`,
  collision days `0 / 882`

So the first routing test was a no-op by construction. The budget never bound.

Correct interpretation:
- this does **not** falsify allocator value
- it falsifies the choice of scarcity surface

---

## 2. Corrected question

The next allocator question should be:

> Does a locked pre-entry state score improve selection under a **binding**
> budget on the validated replayable shelf, before profile-specific deployment
> translation is applied?

That is the right order:

1. prove or reject utility on a binding shelf
2. then translate the survivor into profile/account geometry

---

## 3. Corrected scarcity surface

The next allocator surface must satisfy both:

1. **binding**
   - there are more eligible validated opportunities than the budget can take
2. **realistic**
   - the budget corresponds to an actual operational or capital constraint

This means the next universe cannot be the already-pruned active profile book.

It should instead be:
- replayable validated shelf
- filtered for instrument/session compatibility where necessary
- then constrained by a pre-declared budget

---

## 4. Comparator problem that must be solved first

Before `A4b` is executable, the neutral comparator must be locked honestly.

Plausible canonical comparator surfaces:

1. `trading_app/prop_portfolio.py`
   - static book-construction comparator under DD, contract, and slot budgets
2. `trading_app/lane_allocator.py`
   - dynamic lane-selection comparator from the validated shelf

What is not allowed:
- inventing a new neutral baseline ad hoc
- using the current active book order as if it were the validated-shelf neutral route

So the immediate blocker for `A4b` is:
- audit which repo surface is the correct neutral comparator for a binding
  validated-shelf allocator test

---

## 5. Working design for A4b

### Universe

- validated replayable shelf
- canonical trade paths only
- no `live_config` truth

### Candidate

- locked composite pre-entry score
- one mechanism family only

### Budget

- genuinely binding
- pre-declared
- tied to operational reality, not tuned after first look

### Evaluation

- utility first
- profile translation second

---

## 6. Next order

1. audit and lock the neutral comparator surface
2. write the executable `A4b` hypothesis file from that comparator
3. only then implement and run the revised allocator replay

---

## 7. Current doctrine

- `A4a` active-profile slot routing is **demoted**
- allocator research is still alive
- the next proper step is **binding-budget validated-shelf allocation**, not
  more active-profile routing
