# Pre-Reg Stub — HTF Level Break Hypothesis (Path A, DEFERRED / PARTIALLY CLOSED)

**Status:** DEFERRED at creation; updated 2026-04-20 after verification. The simple v1 "take the break through prev-week / prev-month level" family was implemented, pre-registered, and FAMILY-KILLED on 2026-04-18. This stub now remains open only for structurally new HTF mechanisms, not for the already-killed v1 family.

**Created:** 2026-04-15
**Parent handover:** `docs/handoffs/2026-04-15-session-handover.md` § Tier 1
**Trigger:** User flagged need for BIG-signal hypotheses (prev-week, prev-month, major levels), not small quality-modifier filters (rel_vol, garch). Path C chosen first; Path A pinned here.

---

## Core hypothesis (provisional wording — refine before commit)

> When the ORB break direction aligns with (a) a break of the prev-week or prev-month high/low AND (b) is the first touch of that HTF level in the current week/month, the post-break trade shows higher ExpR and WR than breaks that occur inside the prior-week/month range.

Mechanism prior: Fitschen Ch 3 (intraday trend-follow on commodities + stock indices), Murphy (technical analysis on structural levels), Chan Ch 4 (mean-reversion vs trend at levels). HTF-level-break mechanism is classic breakout structure.

---

## Feature state (2026-04-20 correction)

The original version of this stub said the HTF fields below were "not yet built."
That is now stale. As verified on 2026-04-20:

- `prev_week_*` and `prev_month_*` are already built in `pipeline/build_daily_features.py`
- canonical population path is `_apply_htf_level_fields(...)`
- narrow-run seed loading was fixed in commit `234c7d0d`
- one lingering stale row in `gold.db` (`MGC 2026-04-17 O5`) was repaired via `scripts/backfill_htf_levels.py`
- v1 scan results are already on disk:
  - `docs/audit/results/2026-04-18-htf-path-a-prev-week-v1-scan.md`
  - `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md`
  - both verdicts = `FAMILY KILL`

What remains unbuilt are the *touched-to-date / first-touch* features and any
new rolling-high / rolling-low variants:

| Feature | Definition | Storage |
|---------|-----------|---------|
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
| Chan Ch 4 level mean-reversion | ❌ category error for HTF break grounding | removed 2026-04-19 after TOC verification |
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

**Next action when resumed:** only reopen if the new pre-reg is structurally different from the killed v1 family. Acceptable reopen shapes:

1. first-touch / touched-to-date gating once those features exist
2. distance-to-HTF-level / inside-vs-outside range conditioning rather than simple break-through predicates
3. literature-grounded HTF level theory that changes the prior and the evaluation pathway

Do not reopen the simple "prev-week / prev-month break-aligned take filter" family.
