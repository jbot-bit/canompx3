# Research / Decision Governor

**Class:** `contract` (per `document_authority.md` § Document classes).
**Owning source:** the canonical surfaces named in each row below — this doc owns
*routing*, never *thresholds*.
**Verification gate:** the worked-example walk in
`docs/audit/templates/decision_candidate_review.md` must reproduce against live
tools; `python pipeline/check_drift.py` stays green. If any pointer below stops
resolving, this doc is stale and the linked canonical surface wins.

Date: 2026-06-05.

---

## 1. Why this exists (the failure it fixes)

Recent C11 / fourth-lane / account-size work **tunnel-visioned**: one candidate
was checked deeply while the layers it interacts with (portfolio EV, account
survival, validation state, deployment gate) went unexamined, and no step asked
*"what higher-EV item are we ignoring by doing this?"*

Every gate this governor routes to **already exists** (C1–C13, MinBTL, the
14-gate live preflight, the lane allocator, account survival). The gap was never
a missing gate — it was the **absence of a single front-door that runs all of
them against ONE candidate, at decision time, and prints blockers +
classification + higher-EV alternatives.**

This doc is that front-door. It is a **router and checklist**, not a gate. It
**composes** existing canonical surfaces and **re-encodes none of them**
(`institutional-rigor.md` § "no re-encoded canonical logic"). If this doc ever
states a numeric threshold, that is a bug — it must *point* to the surface that
owns the number.

### What this is NOT
- NOT a new validation gate. C1–C13 in `pre_registered_criteria.md` are the gate.
- NOT a source of any threshold, budget, t-stat, or K-bound.
- NOT mandatory ceremony for trivial changes (see `workflow-preferences.md`
  trivial tier). It governs **decisions about what to research, deploy, or size**
  — not config tweaks or doc edits.

---

## 2. The decision stack — 8 layers, canonical source, stale-risk, verify

A candidate almost always touches more than one layer. The tunnel happens when
you reason about one layer and assume the rest. For each layer: **what is true**,
**what derived copy of it goes stale**, **how it fails silently**, and **the one
command that re-grounds it.**

| # | Layer | Canonical source of truth | Derived output that goes stale | Known silent-failure mode | Verify-before-trust |
|---|---|---|---|---|---|
| L1 | **Data truth** | `bars_1m`, `daily_features`, `orb_outcomes` (built by `pipeline/build_daily_features.py`, `pipeline/ingest_dbn_*`; schema `pipeline/init_db.py`) | `docs/runtime/baselines/*.json` | stale `daily_features` row for `(day,symbol,orb_minutes)`; deprecated scratch-copy DB | `gold-db get_db_freshness` / `get_db_health` |
| L2 | **Feature validity** | `pipeline/build_daily_features.py`; banned-lookahead list in `.claude/rules/daily-features-joins.md` § Look-Ahead | feature columns derived post-ORB | banned field leaks (`break_ts`, `break_delay_min`, `double_break`, `mae_r`, …) | `/crg-lineage <column>`; class-bug drift checks |
| L3 | **Strategy validation** | `pre_registered_criteria.md` (Criteria 1–13, LOCKED 2026-04-07) enforced by `trading_app/strategy_validator.py`; `validated_setups` table | `validated_setups` rows; `fast_lane_status.yaml`, `promote_queue.yaml` | post-hoc threshold relaxation; grandfathered RESEARCH-PROVISIONAL rows treated as OOS-clean (Criterion 8, as revised by Amendment 2.7) | `research-catalog estimate_k_budget`; `strategy-lab get_strategy_readiness` |
| L4 | **Portfolio construction** | `trading_app/lane_allocator.py` → `docs/runtime/lane_allocation.json` | `lane_allocation.json` (`staleness` field) | adding a lane without checking correlation / EV vs the existing book — the literal tunnel | `strategy-lab get_lane_allocation_summary` (read `staleness`) |
| L5 | **Prop-account survival** | `trading_app/account_survival.py` `effective_strict_dd_budget()` (budget = fraction × MLL); `trading_app/prop_profiles.py` `ACCOUNT_PROFILES` / `ACCOUNT_TIERS` | `STATE_DIR/account_survival_{profile}.json` | budget-fingerprint mismatch → stale FAIL reads as a real NO-GO; prop cap leaking into self-funded sizing | re-run `evaluate_profile_survival(write_state=False)`; `live_readiness_report.py --profile-id …` |
| L6 | **Live deployment** | `scripts/run_live_session.py` `PREFLIGHT_CHECKS` (14 gates — count = `len(PREFLIGHT_CHECKS)`); `session_orchestrator.py` bracket/fill verifiers | survival-report fingerprint, journal lock | telemetry-maturity gate (WAIVED for express-funded per doctrine); fingerprint staleness reads as NO-GO | `run_live_session.py --instrument MNQ --preflight` (read-only dry-run) |
| L7 | **Monitoring / telemetry** | `trading_app/strategy_fitness.py` `classify_fitness()`; `trading_app/live/telemetry_maturity.py` (C12 SR monitor) | fitness classification cache | decay unnoticed; 30-day maturity floor blocking a funded launch | `gold-db get_strategy_fitness summary_only=True` |
| L8 | **Repo / governance** | `pipeline/check_drift.py` (count self-reported at runtime); `docs/governance/document_authority.md`; `.githooks/pre-commit` | `REPO_MAP.md`, generated context docs, **`chatgpt_bundle/`** | stale prose surviving a code change; binding-vs-advisory confusion | `python pipeline/check_drift.py` |

**Counts are deliberately not hard-coded here.** "14 gates" = `len(PREFLIGHT_CHECKS)`,
"Criteria 1–13" is the locked range, drift count is self-reported. If a count in
this table disagrees with the command output, the command wins and this row is stale.

---

## 3. The decision-class router (route BEFORE you check)

The honest fix is **not** to force all 13 questions on every candidate — a pure
account-sizing decision should not have to answer "is the mechanism real?". First
classify, then apply the relevant subset. A candidate may belong to more than one
class; run every class it touches.

| Decision class | What it is | Mandatory questions (§4) | Primary layers |
|---|---|---|---|
| **research-validation** | a new edge / filter / feature claim | Q1, Q3, Q4, Q5, Q6, Q8, Q9 | L1–L3 |
| **portfolio** | add / drop / reweight a lane | Q1, Q7, Q10, Q13 | L3, L4 |
| **account-sizing** | account tier, contract count, DD budget | Q11, Q13 | L5, L7 |
| **deployment-gate** | arm / pause a live session | Q11 + the 14 preflight gates (L6) | L5, L6, L7 |
| **classification** | "which role is this?" (R1–R8) | Q2 | `mechanism_priors.md` § 4 |

Q12 ("which class is this overall?") and Q13 ("what higher-EV item are we
ignoring?") apply to **every** candidate — Q13 is the anti-tunnel guard the recent
work lacked, and it has no exemption.

---

## 4. The question set — each grounded by its canonical source

Answer only the questions your class(es) require (§3), plus Q12 and Q13 always.
**Every answer must paste the output of its grounding command** — an asserted
answer is not an answer (this is the same discipline as the Q-loop's own thesis).

| # | Question | Class it tests | Grounded by |
|---|---|---|---|
| Q1 | What is the claimed edge (ExpR / Sharpe, N)? | research-validation | `validated_setups` row / scan output |
| Q2 | **Which role is it?** FILTER / direction / size / stop / target / entry / confluence / allocator | classification | `mechanism_priors.md` § 4 **R1–R8** taxonomy (canonical axis) |
| Q3 | Already killed or parked? | research-validation | `/nogo` (research-catalog) + `STRATEGY_BLUEPRINT.md` § NO-GO |
| Q4 | Mechanism real or story? | research-validation | `mechanism_priors.md` § 2 + `theory_citation` |
| Q5 | Knowable strictly before entry? | feature-validity | banned-lookahead list (L2) |
| Q6 | Uses any banned lookahead field? | feature-validity | `/crg-lineage <column>` |
| Q7 | Duplicates / correlates an existing filter? | portfolio | `backtesting-methodology.md` tautology RULE (correlation ceiling owned there — cite, don't restate) |
| Q8 | Honest K / trial count? | research-validation | `estimate_k_budget` (MinBTL; Criterion 2 bound) |
| Q9 | Survives the locked criteria? (MinBTL / BH-FDR / Chordia t-floor / WFE / N / OOS / era split / cost-stress) | research-validation | Criteria 2,3,4,6,7,8,9 + `backtesting-methodology.md` |
| Q10 | Improves **portfolio** EV after correlation + drawdown constraints? | portfolio | `lane_allocator.py` / lane summary |
| Q11 | Passes **account survival** at the target profile? | account-sizing / deployment | `account_survival.py` `evaluate_profile_survival()` |
| Q12 | **Which decision class(es)** is this? | classification | this doc § 3 |
| Q13 | **What higher-EV open item are we ignoring by doing this?** | anti-tunnel (ALL) | open-hypothesis count (`research-catalog list_open_hypotheses`) + the active blocker list |

**Honest limit of Q9:** the *exact* numeric floors (Chordia t, WFE, N, q-value)
live in `pre_registered_criteria.md` and are enforced by `strategy_validator.py`.
This doc lists the criterion **names** so the answer is routed to the right gate;
it does not state the numbers, because a second copy of a number is a second thing
that drifts.

---

## 5. Stale / contradictory assumptions this governor must reconcile

These were live contradictions across batons and docs as of 2026-06-05. The
governor's job is to force re-grounding so the wrong half doesn't get acted on.

- **Account size is a survival-headroom decision, NOT an earnings decision** —
  *until multi-micro sizing lands.* `account_survival.py` hardcodes
  `contracts_per_trade_micro = 1` (verified `account_survival.py:636`), so every
  Topstep tier runs the **same** gross edge; a bigger MLL buys only drawdown
  headroom. Conflicting batons ($100k-now vs $50k+cap) must be read through this
  lens. (The vol-scaled sizer `compute_position_size_vol_scaled` exists but is
  clamped to `max_contracts = 1`, so the moot-ness holds today.)
- **C11 budget = $1,800**, not $1,600. Canonical = `effective_strict_dd_budget()`
  (`account_survival.py:75`) = 0.90 × MLL (express fraction) → $1,800 at the 50k
  tier. Multiple retired batons cite $1,600 — **STALE**; re-run the function, never
  quote the number from memory.
- **124 `validated_setups` are grandfathered RESEARCH-PROVISIONAL, NOT OOS-clean**
  (Criterion 8 as revised by Amendment 2.7; `pre_registered_criteria.md:18`). A
  deployed lane is not an OOS-clean lane. The governor must force this distinction
  on any portfolio decision touching grandfathered rows.
- **Paper / shadow / live separation** is real: `--signal-only` / `--demo` /
  `--live` in `run_live_session.py`. Telemetry-maturity is WAIVED for
  express-funded profiles **only** (`telemetry-maturity-waiver.md`); real-capital
  self-funded profiles keep the floor.
- **Active blockers are LIVE state, read them live:** C11 DD-vs-budget gap; audit
  `9b3fc530` (bracket-parity) status; `prop_profiles.py` peer-ownership. Never
  carry these from memory into a decision — they move.
- **Binding vs advisory** is authoritative in `document_authority.md`.
  `chatgpt_bundle/` (a hand-pasted ChatGPT mirror), `HANDOFF.md`, `docs/plans/`,
  and `ROADMAP.md` are advisory / stale-prone. **`chatgpt_bundle/` carries no
  regenerator and is flagged stale-prone** — see its `_BUNDLE_README.md` marker;
  do not cite it as canonical.
- **Literature freshness:** verify any citation against
  `docs/institutional/literature/` extracts live (137 open hypotheses as of
  2026-06-05 — that pool *is* the Q13 opportunity set).

---

## 6. How to use it

1. Open `docs/audit/templates/decision_candidate_review.md`, copy it to a dated
   file per candidate.
2. Answer Q12 first → it tells you which question subset applies (§3).
3. For each required question, **run the grounding command and paste the output.**
4. The candidate's verdict is the union of the layer blockers, plus the explicit
   Q13 answer. If Q13 names a higher-EV item with fewer blockers, that is the
   signal to stop and reconsider — which is the entire point.

This doc **composes**; it does not gate. The gates are C1–C13, the 14 preflight
checks, the K-budget estimator, the lane allocator, and account survival — each
owns its own thresholds, and each is cited above so the answer is grounded, not
asserted.

---

## Related

- `docs/institutional/pre_registered_criteria.md` — the locked validation gate (L3).
- `docs/institutional/mechanism_priors.md` § 4 — the R1–R8 role taxonomy (Q2).
- `.claude/rules/backtesting-methodology.md` — the 13 backtest RULES (Q7, Q9).
- `scripts/tools/live_readiness_report.py` — composed deployment readiness (L5/L6).
- `scripts/run_live_session.py` § `PREFLIGHT_CHECKS` — the 14 live gates (L6).
- `docs/governance/document_authority.md` — binding-vs-advisory map (L8).
- `.claude/rules/self-funded-sizing-doctrine.md` — risk-first sizing (L5, § 5).
- `.claude/rules/telemetry-maturity-waiver.md` — express-funded waiver (L6/L7).
- `docs/audit/templates/decision_candidate_review.md` — the fill-in companion.
