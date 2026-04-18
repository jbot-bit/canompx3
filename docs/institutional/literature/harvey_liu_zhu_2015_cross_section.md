# …and the Cross-Section of Expected Returns — Harvey, Liu, Zhu (2015)

**Reference:** Campbell R. Harvey (Duke), Yan Liu (Texas A&M), Heqing Zhu (Duke).
**Publication:** *The Review of Financial Studies*, Vol. 29, Issue 1, January 2016, pp. 5–68.
**First version:** October 2013. Working paper widely circulated 2014–2015 (often cited as "HLZ 2015" or "HLZ 2016").
**Status in /resources:** **NOT PRESENT.** Tier 2 PDF acquisition required to close IMP-1 of 2026-04-18 grounding audit.

**Criticality:** 🟡 **MEDIUM** today, **HIGH** if a 3.00 ≤ t < 3.79 with-theory candidate appears. This extract is the canonical Tier 1 source for the `t ≥ 3.00` with-theory threshold in `pre_registered_criteria.md` Criterion 4. Currently the threshold is traceable only via Chordia et al 2018's indirect reference.

---

## Status: STUB PENDING PDF ACQUISITION

This file is a **STUB** created during the 2026-04-18 grounding audit (IMP-1 remediation) to:

1. Reserve the canonical Tier 1 path so future readers have a stable citation target.
2. Document the indirection explicitly — the with-theory t≥3 threshold is currently traceable only via Chordia's one-step-removed reference.
3. Provide a remediation checklist so the next person who touches this can close IMP-1.

**No verbatim page-anchored spans from the HLZ paper appear below** — training-memory transcription is banned by project canon (`CLAUDE.md` § Local Academic Grounding Rule). Any quoted content must come from the actual PDF once acquired.

---

## What is grounded RIGHT NOW (via Tier 1 indirect reference)

The HLZ "t ≥ 3" recommendation is traceable — but only via Chordia et al 2018, which explicitly references HLZ's proposal:

**Source:** `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md:20`, quoting Chordia et al 2018 p5 verbatim:

> "While these thresholds are quite a bit higher than the conventional thresholds of 1.96, they are not far from the suggestion of Harvey, Liu, and Zhu (2015) to use a threshold of three."

**What this establishes:**
- HLZ-2015 proposes a t-stat threshold of 3 for factor significance.
- The threshold is a recommendation, not a hard bar (per Chordia's framing).
- HLZ-2015's threshold is LOWER than Chordia's t=3.79 (which was derived from a 2M-strategy CRSP/COMPUSTAT sample under FDP-StepM with cross-correlation adjustment).

**What this does NOT establish:**
- HLZ-2015's own derivation of t=3 (which methodological framework, which sample, which correction).
- Whether HLZ-2015 distinguishes "with-theory" vs "without-theory" candidates — this distinction is embedded in `pre_registered_criteria.md` Criterion 4 but not verified against HLZ-2015's actual text.
- The actual MHT correction method HLZ-2015 recommends (Bonferroni, Holm, BHY, or a bespoke approach).
- Whether t=3 is an IS threshold or an OOS threshold.

## From training memory — NOT verified against local PDF

**Labelled per CLAUDE.md § Local Academic Grounding Rule. Do NOT cite as authoritative until PDF arrives.**

- HLZ-2015 RFS paper "…and the Cross-Section of Expected Returns" surveys 316 claimed equity factors over ~50 years of finance literature.
- Famous Figure 1 in the paper plots the cumulative number of discovered factors over time, showing the "factor zoo" explosion.
- The paper proposes THREE MHT methods — Bonferroni, Holm, and BHY — with BHY being the authors' preferred approach for financial applications (consistent with Harvey-Liu 2015 JPM "Backtesting" which we DO have locally at `harvey_liu_2015_backtesting.md:66`: "we advocate the BHY method").
- The t ≥ 3.0 recommendation is positioned as a rough heuristic after MHT adjustment for a factor survey of the size of HLZ's.
- The "with-theory" carve-out in our Criterion 4 rule is a project-specific refinement of HLZ's general recommendation; HLZ-2015 itself does not explicitly partition candidates into with-theory / without-theory buckets (to the best of unverified training memory).

None of the above is citable in a deployment decision until verified against the actual PDF.

---

## Why the indirection is acceptable NOW but not INDEFINITELY

### Why acceptable now

As of the 2026-04-18 grounding audit:
- All in-scope zero-pass verdicts (wide-rel-IB NULL max t=2.74, cross-NYSE NULL max t=2.15) fail the stricter Chordia t=3.79 bar regardless. So the with-theory carve-out is not decision-load-bearing.
- Any VERIFIED candidate sitting in 3.00 ≤ t < 3.79 band would need explicit "strong pre-registered economic theory support" argumentation anyway, which would require separate grounding beyond t-stat.

### When the indirection stops being acceptable

The moment ANY candidate discovery run produces:
- `t_statistic in [3.00, 3.79)`, AND
- pre-registered hypothesis cites "economic theory support" to invoke the with-theory carve-out

At that point the with-theory threshold becomes decision-load-bearing and the Chordia indirect reference is insufficient — acquire the PDF and complete this extract BEFORE promoting the candidate.

---

## Remediation checklist to close IMP-1

1. Acquire HLZ-2015 RFS PDF. Likely sources:
   - SSRN preprint (Harvey-Liu-Zhu October 2013 working paper, still circulating)
   - RFS published version (Jan 2016 issue) via institutional access
   - Duke CFRC working paper series
2. Place at `resources/harvey_liu_zhu_2015_cross_section.pdf` (or similar).
3. Extract page-anchored verbatim spans for:
   - The t ≥ 3 recommendation itself (quote + page)
   - The MHT correction method HLZ uses (Bonferroni vs BHY vs bespoke)
   - Whether HLZ distinguishes IS vs OOS application
   - Any explicit with-theory / without-theory distinction if present
4. Replace this stub's "Status: STUB PENDING PDF ACQUISITION" block with the extracted content.
5. Remove the "From training memory" block; keep only verbatim-verified content.
6. Update `docs/institutional/pre_registered_criteria.md:114` to remove the indirection note.

---

## Cross-references

- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md:20` — indirect reference that currently grounds the t ≥ 3 threshold
- `docs/institutional/literature/harvey_liu_2015_backtesting.md` — different paper (Harvey + Liu, JPM 2015, "Backtesting") that DOES have local extraction; do not confuse with HLZ-2015 RFS
- `docs/institutional/pre_registered_criteria.md:110-114` — Criterion 4 rule that currently depends on this stub
- `docs/audit/2026-04-18-grounding-audit-master.md` IMP-1 — audit finding that triggered this stub

## Related literature (all locally extracted)

- `bailey_et_al_2013_pseudo_mathematics.md` — MinBTL bound (upstream of multiple-testing thresholds)
- `bailey_lopez_de_prado_2014_deflated_sharpe.md` — DSR, alternative to t-stat MHT corrections
- `chordia_et_al_2018_two_million_strategies.md` — t ≥ 3.79 without-theory threshold (supersedes HLZ's t=3 recommendation at the 2M-strategy scale)
