# Pre-Reg Stub — HTF Level Break Hypothesis (Path A, DEFERRED)

**Status:** DEFERRED — not started. Stub held open pending completion of Path C (finish H2/rel_vol book) and then a design session to firm the pre-reg into a runnable hypothesis.

**Created:** 2026-04-15
**Parent handover:** `docs/handoffs/2026-04-15-session-handover.md` § Tier 1
**Trigger:** User flagged need for BIG-signal hypotheses (prev-week, prev-month, major levels), not small quality-modifier filters (rel_vol, garch). Path C chosen first; Path A pinned here.

---

## Core hypothesis (provisional wording — refine before commit)

> When the ORB break direction aligns with (a) a break of the prev-week or prev-month high/low AND (b) is the first touch of that HTF level in the current week/month, the post-break trade shows higher ExpR and WR than breaks that occur inside the prior-week/month range.

Mechanism prior: Fitschen Ch 3 (intraday trend-follow on commodities + stock indices), Murphy (technical analysis on structural levels), Chan Ch 4 (mean-reversion vs trend at levels). HTF-level-break mechanism is classic breakout structure.

---

## Required pipeline features (not yet built)

Add to `pipeline/build_daily_features.py` (or a new module called from it):

| Feature | Definition | Storage |
|---------|-----------|---------|
| `prev_week_high` | max(session_high) for Mon-Fri of PRIOR week (ISO week) | daily_features |
| `prev_week_low` | min(session_low) for prior week | daily_features |
| `prev_week_close` | close of last bar of prior week | daily_features |
| `prev_month_high` | max(session_high) for calendar PRIOR month | daily_features |
| `prev_month_low` | min(session_low) for prior month | daily_features |
| `prev_month_close` | close of last bar of prior month | daily_features |
| `week_high_touched_to_date` | boolean: has current week already traded through prev_week_high? | daily_features (needs day-by-day accumulation) |
| `month_high_touched_to_date` | boolean equivalent for month | daily_features |

All features must be trade-time-knowable — populate from fully closed prior-week/month data only. ORB-session execution agnostic.

**Implementation audit before coding:** verify bars_1m has enough horizon for a 4-5 week lookback (for first-week-of-year edge cases), verify week-boundary handling matches Brisbane trading-day model (09:00 local boundary), verify ISO week vs calendar week convention matches whatever MNQ data providers use.

---

## Proposed pre-reg structure (flesh out later)

Per `docs/institutional/pre_registered_criteria.md`:

1. **Numbered hypotheses** — minimum 3, maximum 8. Each with exact SQL and mechanism citation.
2. **Enumeration budget** — N_trials ≤ 300 on clean MNQ/MES (per Bailey MinBTL). Likely scope: 2 HTF levels × 12 sessions × 3 instruments × 2 apertures (O15, O30 only — O5 likely too noisy) × 2 RRs × 2 directions × first-touch-vs-not = ~576 raw combos — **needs reduction to ≤300**.
3. **Kill criteria** — if < 5 cells pass T0+T1+T3 after BH-FDR at K=(budget), declare Path A dead for HTF levels and pursue round-number levels or Market Profile instead (Path A.2).
4. **Expected fire rate** — prev-week-high is touched ~30-40% of Mondays, tapering through the week. Too-rare cells ( < 5% fire) auto-excluded per RULE 8.1.

---

## Literature grounding needed

| Source | Have? | If not, acquire before pre-reg |
|--------|-------|-------------------------------|
| Fitschen 2013 Ch 3 intraday trend-follow | ✅ in `docs/institutional/literature/` | — |
| Murphy (technical analysis on levels) | ❌ | priority before pre-reg |
| Chan Ch 4 level mean-reversion | ✅ `resources/Algorithmic_Trading_Chan.pdf` | extract needed |
| Dalton *Markets in Profile* | ❌ | priority for Path A.2 (Market Profile) |
| O'Neill *How to Make Money in Stocks* (new highs) | ❌ | optional; not commodities-specific |

---

## Entry / exit / sizing proposal (design this in Path A kickoff)

- **Entry:** same E2 stop-market at ORB break (no new entry model needed — just a FILTER on existing entry model).
- **Filter scope:** overlay filter on existing ORB lanes; not a standalone strategy class initially. Stage 1 binary pre-reg per mechanism_priors.md R1.
- **If validated:** promote to R3 CONFIRMATION (combined with deployed lane filter) or R5 SIZER (Carver forecast) per `docs/institutional/mechanism_priors.md`.

---

## When to kick off

- After Path C completes (DSR calibration + composite + H2 closure). Estimated Path C completion: current session.
- After any non-ORB terminal work (Phase E) reports so the level-design doesn't duplicate effort.
- Before starting, revisit `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md` to ensure non-duplication with parallel non-ORB design.

**Next action when resumed:** design session (mode: plan) to convert this stub into a proper pre-reg file `docs/audit/hypotheses/YYYY-MM-DD-htf-level-break.md` per the pre_registered_criteria template. No code until pre-reg is committed.
