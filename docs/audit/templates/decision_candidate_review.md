# Decision Candidate Review — TEMPLATE

> Companion to `docs/governance/research_decision_governor.md`. Copy this file to
> a dated, named instance per candidate (e.g.
> `docs/audit/results/2026-06-05-<candidate-slug>-decision-review.md`) and fill it
> in. **Every answer must paste live command output — an asserted answer is not an
> answer.** Answer Q12 first; it tells you which subset applies (governor § 3).

---

**Candidate:** `<one-line description of the proposed change>`
**Date:** `<YYYY-MM-DD>`
**Reviewer:** `<who>`

## Q12 — Decision class(es) (route first)
> Governor § 3. Which of: research-validation / portfolio / account-sizing /
> deployment-gate / classification. May be more than one.

```
<class(es) — then run only those questions + Q12 + Q13>
```

---

## Required questions (run only those your class needs; Q12 & Q13 always)

### Q1 — Claimed edge (ExpR/Sharpe, N) — *research-validation*
> Grounding: `validated_setups` row / scan output.
```
<paste output>
```

### Q2 — Role R1–R8 — *classification*
> Grounding: `mechanism_priors.md` § 4 R1–R8 taxonomy.
```
<R-role + one-line justification>
```

### Q3 — Already killed/parked? — *research-validation*
> Grounding: `/nogo <topic>` + `STRATEGY_BLUEPRINT.md` § NO-GO.
```
<paste /nogo verdict>
```

### Q4 — Mechanism real or story? — *research-validation*
> Grounding: `mechanism_priors.md` § 2 + theory_citation (a `docs/institutional/literature/` extract).
```
<mechanism + citation, or "STORY — no extract">
```

### Q5 — Knowable strictly before entry? — *feature-validity*
> Grounding: banned-lookahead list (`.claude/rules/daily-features-joins.md` § Look-Ahead).
```
<yes/no + which features>
```

### Q6 — Uses any banned lookahead field? — *feature-validity*
> Grounding: `/crg-lineage <column>`.
```
<paste lineage; flag any of break_ts / break_delay_min / double_break / mae_r / …>
```

### Q7 — Duplicates/correlates an existing filter? — *portfolio*
> Grounding: `backtesting-methodology.md` tautology RULE (correlation ceiling owned there).
```
<correlation vs existing book>
```

### Q8 — Honest K / trial count? — *research-validation*
> Grounding: `research-catalog estimate_k_budget` (MinBTL; Criterion 2 bound).
```
<paste estimate_k_budget output>
```

### Q9 — Survives the locked criteria? — *research-validation*
> Grounding: Criteria 2,3,4,6,7,8,9 (`pre_registered_criteria.md`, enforced by
> `strategy_validator.py`) + `backtesting-methodology.md`. List PASS/FAIL per
> named criterion; do NOT restate the numeric floors here — cite the gate.
```
<per-criterion PASS/FAIL>
```

### Q10 — Improves PORTFOLIO EV after correlation + DD constraints? — *portfolio*
> Grounding: `strategy-lab get_lane_allocation_summary` (read `staleness`).
```
<paste lane summary delta>
```

### Q11 — Passes ACCOUNT SURVIVAL at target profile? — *account-sizing / deployment*
> Grounding: `evaluate_profile_survival(write_state=False)` /
> `live_readiness_report.py --profile-id …`. Budget = `effective_strict_dd_budget()`
> — re-run it, never quote from memory.
```
<paste survival verdict: op_pass %, budget $, max90dDD $, breach_days>
```

### Q13 — What higher-EV open item are we IGNORING by doing this? — *ALL, no exemption*
> Grounding: `research-catalog list_open_hypotheses` (open count) + the live
> active-blocker list. Name at least one alternative and compare blocker count.
```
<the alternative we're not doing, and why this candidate still wins (or doesn't)>
```

---

## Verdict
> Union of layer blockers + the Q13 answer. If Q13 names a higher-EV item with
> fewer blockers, STOP and reconsider.

```
GO / NO-GO / RECONSIDER  —  <one-line reason, blocker list>
```

---
---

# WORKED EXAMPLE — C11 / topstep_50k_mnq_auto (filled 2026-06-05)

> Proves the loop end-to-end with **no new code** — every answer is a live tool/
> code-grounded read, exactly as the template demands. This is the candidate the
> recent tunnel formed around; running it through the governor surfaces what the
> tunnel skipped (Q10, Q13).

**Candidate:** Arm `topstep_50k_mnq_auto` live (C11 currently blocks it).
**Date:** 2026-06-05. **Reviewer:** decision-governor PASS-2 worked example.

### Q12 — Decision class(es)
```
deployment-gate (primary) + account-sizing. NOT research-validation — the lanes
are already validated; the question is survival + arming, not edge discovery.
→ Run Q11 (+14 preflight), Q13. Skip Q1/Q4/Q5/Q6/Q8/Q9 (no new edge claim).
```

### Q11 — Account survival at target profile
```
Grounding: evaluate_profile_survival(write_state=False), account_survival.py:75/636.
- budget = effective_strict_dd_budget() = 0.90 × MLL (express) = $1,800 at 50k
  (NOT the stale $1,600 in retired batons — re-run confirms $1,800).
- max90dDD = $2,038.84 (UNCAPPED baseline) > budget $1,800 → FAILS by ~$239 on
  DD magnitude only; breach_days = 0.
- contracts_per_trade_micro = 1 hardcoded → account size is moot for earnings;
  bigger MLL buys only headroom (governor § 5).
VERDICT: FAIL on DD magnitude. Known fixes: tighten ORB cap (cap_x0.80 → ~$1,594,
clears) OR bigger-MLL profile (buys headroom, zero edge gain).
```

### Q13 — Higher-EV item being ignored
```
Grounding: research-catalog list_open_hypotheses → 137 open hypotheses (2026-06-05);
lane summary → 3 active / 845 paused, staleness OK 6d.
The C11 fix is a ~$239 DD-headroom problem on ONE profile, gated behind a tighter
ORB cap that's already understood. Against 137 open hypotheses and 845 paused
lanes, the marginal EV of forcing this one profile live is bounded by its single
book's edge. HONEST CALL: the cap fix is cheap and nearly done, so finishing it is
fine — but Q13 flags that "buy a bigger account to clear C11" is NOT higher-EV
(it's moot at 1 micro), and that broad re-scanning of the 845 paused lanes likely
dominates any single-profile arming effort. Do not let C11 absorb attention the
paused-lane backlog deserves.
```

### Verdict
```
RECONSIDER → finish the cheap cap fix (cap_x0.80 clears at $1,800); do NOT buy a
bigger account (moot at 1 micro); Q13 says the paused-lane backlog is the real
higher-EV pool. C11 stays NO-GO until cap is wired + bracket-parity audit 9b3fc530
closes. (All live state — re-verify before acting; blockers move.)
```
