# Qwen "What's Next" List — Ground-Truth Fact-Check Verdict

**Date:** 2026-06-12
**Type:** read-only fact-check (no capital, no schema, no candidate validation, no DB writes)
**Scope:** the 4 "immediate execution order" items in the submitted Qwen list ONLY — verify each
against live ground truth; out of scope = any new candidate validation, scan, or capital change.
**Trigger:** an externally-generated (Qwen) "remaining high-EV vectors" list with an
"immediate execution order" was submitted for action. Operator instruction: *"check this from
qwen and action what is true and valid… it might have made some [file names] up; see if you
can logically make it proper, institutional grade, grounded in lit."*

**Purpose:** record the TRUE current status of every "next move" Qwen named, so the next
session (or external paste) cannot re-chase a resolved / by-design / stale-snapshot item.
**This is n=2 of the same class** — companion to the 2026-06-11 precedent below.

**Companion precedent (same class, prior day):**
`docs/audit/results/2026-06-11-qwen-overlay-claims-reaudit.md` — a Qwen overlay that fabricated
~40% of its specific claims and propagated a stale "awaiting OOS" framing. Today's list repeats
the pattern: ~3 of 4 "immediate" items rest on April-2026 snapshots.

**Method:** each claim cross-checked against (a) **live `gold.db`** row counts, (b) live
`pipeline.check_drift` holdout guards (executed, not read), (c) canonical source files
(`trading_app/holdout_policy.py`, `TRADING_RULES.md`), (d) `docs/STRATEGY_BLUEPRINT.md`
current-state lines. **No Qwen number is trusted; every figure is re-derived live or cited.**

---

## Verdict table

| # | Qwen claim | Verdict | Ground truth (live / canonical) |
|---|---|---|---|
| 1 | "Mode A rebuild `--holdout-date 2026-01-01` to cleanse 124 research-provisional setups → restore ~117 clean baseline." | **STALE — NO ACTION** | `validated_setups` = **871 rows** (848 active / 23 retired), promoted **2026-04-11 → 2026-05-24** — after the Apr-3-8 Mode B window and its Apr-8 rescission. No "124 provisional cohort" exists. `check_holdout_contamination()` and `check_holdout_policy_declaration_consistency()` both **PASS (0 violations)** when executed live. Flag/date are real (`trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM = date(2026,1,1)`), but the cleanse *rationale* is months out of date. |
| 2 | "Rebuild `edge_families` — table at 0 rows per BLUEPRINT §9; unblocks `lane_allocator.py`." | **REFUTED — DEAD ACTION** | `edge_families` = **528 rows** (MNQ 475 / MES 47 / MGC 6), built/verified **2026-06-11** (`STRATEGY_BLUEPRINT.md:386`). The "§9 says 0 rows" citation is **fabricated** — §9 is "Active Research Threads" (`:400`). "`lane_allocator.py` depends on it" is **REFUTED** — lane_allocator does not read the table. Builder `scripts/tools/build_edge_families.py` exists and already ran. |
| 3 | "Sort active lanes by ExpR/Sharpe before the match loop — frictionless alpha capture for `max_per_orb=1` collisions." | **REAL behavior, BY DESIGN — capital change, not a bug** | `TRADING_RULES.md:644-645` documents lane-order (not EV) as a **deliberate, audited** Layer-7 choice: *"There is no ExpR-based or Sharpe-based ranking… If priority matters operationally, reorder lanes."* Backtester **and** execution engine share the ordering → **no backtest-vs-live divergence** ("frictionless" is false). Changing trade selection is a Tier-B capital / trading-logic change needing its own pre-registered research, not a one-line sort. |
| 4 | "Resolve scratch PnL to market in the backtester (`outcome_builder.py`) to remove allocator bias." | **REAL file, ALREADY DESIGNED as conservative bias — not a defect** | `outcome_builder.py` labels scratch `pnl_r=NULL` intentionally; `execution_engine.py:593-666 on_trading_day_end()` **already** MTM-resolves scratch live. Spec: `docs/specs/outcome_builder_scratch_eod_mtm.md`. `TRADING_RULES.md:647-648`: live **exceeds** backtest on scratch-heavy strategies — a **conservative** bias (Layer-2 audit: scratches avg +0.40R MES). Qwen's "fix" would make backtests *more optimistic* (removes the safety margin) → methodology change requiring research sign-off, not a bug repair. |

**Hallucinations caught:** "edge_families at 0 rows," "BLUEPRINT §9 says so," "lane_allocator
depends on edge_families," "124 provisional setups still carried."
**File names Qwen got right (state claims about them still wrong):** `holdout_policy.py`,
`build_edge_families.py`, `outcome_builder.py`, `session_orchestrator.py`.

---

## Lit grounding (why items 3/4 are gated, not killed)
Items 3 and 4 are *legitimate questions*, not worthless — but both alter the discovery/scoring
or live-selection surface, so the repo's institutional gates apply **before** they can be
actioned:
- **Multiplicity / overfitting (Bailey & López de Prado, MinBTL; Harvey & Liu haircut):** any
  change to lane priority or scratch scoring re-touches the strategy-scoring universe and must
  be pre-registered with a K-budget estimate (`docs/institutional/pre_registered_criteria.md`;
  `research-catalog` `estimate_k_budget` gate). Not a "frictionless" tweak.
- **Conservatism direction (RESEARCH_RULES.md):** item 4 would *reduce* backtest conservatism.
  A documented safety margin is not "bias to fix" — flipping it needs explicit methodology
  sign-off; a too-optimistic backtest is the more dangerous failure direction.
- **Route:** `/research` with a pre-reg hypothesis YAML — NOT a one-line code change.

## Caveats / honest gaps (second-pass)
- **The 871-row `validated_setups` count is itself large.** This fact-check does NOT bless it as
  overfitting-clean — multiplicity governance (family-head dedup, FDR, MinBTL) lives in the
  validation pipeline and is out of scope here. This doc only refutes the *"124 provisional /
  needs cleanse"* framing, not the broader question of how many independent edges 871 represents.
- **Verdicts are point-in-time (2026-06-12).** A future DB rebuild or scan can move any row
  count; re-verify against live surfaces before acting, per the Volatile Data Rule.
- **Item 3's exact orchestrator loop line was not personally re-read.** The verdict rests on the
  canonical *doctrine* (`TRADING_RULES.md:644-645`) that lane-order is intentional and shared by
  backtest+live — sufficient to refute "silent bug" regardless of the precise line number.

## Lesson logged
External-advisor "next-step" lists are **claims against a SNAPSHOT** — they read a months-old
state and present resolved work as pending. Falsify each against **live gold.db row counts +
executed drift state** before actioning. Here ~3 of 4 "immediate" items were stale or by-design.
Same trap, second instance — see
`memory/feedback_audit_findings_are_claims_falsify_each_before_acting_2026_06_07.md` and the
2026-06-11 overlay reaudit.

## Related
- `docs/audit/results/2026-06-11-qwen-overlay-claims-reaudit.md` — n=1, same class (prior day).
- `docs/governance/ai_session_canonical_prompt.md` — corrected anti-hallucination prompt.
- `trading_app/holdout_policy.py` — `HOLDOUT_SACRED_FROM` authority (Amendment 2.7).
- `TRADING_RULES.md:644-648` — Signal Collision Priority + Backtest-to-Live Scratch Gap.
- `docs/specs/outcome_builder_scratch_eod_mtm.md` — scratch MTM spec.

## Reproduction

Each verdict is re-derivable from these commands against live state (point-in-time 2026-06-12):

```bash
# Item 1 — validated_setups row count + holdout cleanliness (no "124 provisional cohort")
python -c "import duckdb; from pipeline.paths import GOLD_DB_PATH; \
  print(duckdb.connect(str(GOLD_DB_PATH), read_only=True).execute( \
  \"SELECT status, COUNT(*) FROM validated_setups GROUP BY status\").fetchall())"
python pipeline/check_drift.py 2>&1 | grep -iE "holdout (contamination|policy declaration)"

# Item 2 — edge_families row count (Qwen claimed 0; citation '§9 says so' is fabricated)
python -c "import duckdb; from pipeline.paths import GOLD_DB_PATH; \
  print(duckdb.connect(str(GOLD_DB_PATH), read_only=True).execute( \
  \"SELECT instrument, COUNT(*) FROM edge_families GROUP BY instrument\").fetchall())"

# Items 3 & 4 — by-design canonical doctrine (not bugs), read the cited lines:
sed -n '644,648p' TRADING_RULES.md                      # lane-order + scratch-MTM are audited choices
```

**Outputs (this run):** `validated_setups` = 871 (848 active / 23 retired); both holdout drift
guards PASS (0 violations); `edge_families` = 528 (MNQ 475 / MES 47 / MGC 6); `TRADING_RULES.md`
lines 644-648 confirm lane-order (not EV) and conservative scratch bias are deliberate. See the
verdict table above for the full per-item mapping.
