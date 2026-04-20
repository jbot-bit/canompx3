# Phase 3 — per-claim heterogeneity audit results

**Date:** 2026-04-20
**Author:** retroactive heterogeneity audit (CURRENT-C)
**Phase 1 enumeration:** `docs/audit/results/2026-04-20-heterogeneity-audit-phase1-enumeration.md`

Every claim tested under RULE 14 (`.claude/rules/backtesting-methodology.md`):
HETEROGENEOUS if ≥25% of cells show sign opposite to pooled aggregate, with
per-cell N ≥ 30.

Claims tested:
- A1 — exchange_range / pit_range signal
- A2 — H2 Path C garch_vol_pct≥70 "universal"
- A3 — comprehensive scan 13 K_global BH-FDR survivors "universal volume confirmation"
- B3 — break_quality pooled N=18-27K "universal null"

B4 / B5 (exit-timing universal) intentionally DEFERRED — broader scope than
can be responsibly audited this session; logged in next-session mandates.

---

## A1 — exchange_range / pit_range signal

### Memory claim (before)
> `exchange_range_signal.md`: F5 pit range/ATR — FEATURE validated (T1-T8, +16pp,
> BH K=320), FILTER deployed (PIT_MIN >= 0.10). 3/3 inst, +17% WR, K=320 BH,
> 75-94% years.

### Verdict: **BLOCKED — infrastructure gap preempts heterogeneity audit**

### Evidence

1. `daily_features.pit_range_atr` schema: column exists (`init_db.py:344`)
2. `daily_features.pit_range_atr` **population: 0 rows across all 21 (instrument, orb_minutes) combos** (MNQ 2034×0, MES 2033×0, MGC 1120×0, all others 0)
3. `pipeline/ingest_statistics.py` docstring: "pit_range_atr column in daily_features is backfilled separately using the data stored here" — **the backfill step has not been run**
4. `trading_app/config.py:2304-2373` — `PitRangeFilter` registered as `ALL_FILTERS["PIT_MIN"]`
5. `trading_app/config.py:2346` — filter returns `pit_range_atr.notna() & (pit_range_atr >= min_ratio)` — with all NULL values this **always returns False** (fail-closed)
6. `validated_setups` — **zero rows use PIT_MIN** (neither active nor retired). No currently-live strategy would be affected.

### What this means

The memory claim was derived from `scripts/research/exchange_range_t2t8.py`
which reads DBN files directly (off-database computation). The feature was
added to schema but never materialized in canonical `daily_features`. The
PIT_MIN filter is **infrastructure-orphaned**: code path exists, registry
entry exists, but feature data does not.

**If PIT_MIN were activated into a new validated_setup without backfilling the
column first, the strategy would silently take zero trades** (fail-closed on
NULL values).

### Heterogeneity status: UNVERIFIABLE AT CANONICAL LAYER

The 3/3 inst / K=320 BH / 75-94% years claim cannot be re-evaluated against
canonical data because the feature data is not in the canonical DB. Any
"universal" framing or per-cell decomposition would require rerunning the
original DBN-reading script, outside this session's scope.

### Recommended action

1. **Do NOT activate PIT_MIN** on any new validated_setup until `pit_range_atr`
   is backfilled. Add to queue: run `pipeline/ingest_statistics.py --all` then
   the backfill step (whose location needs locating).
2. **Add drift-check** that flags any `ALL_FILTERS` entry whose required
   column is 0% populated in canonical `daily_features` — this is a
   latent silent-failure class.
3. **Rewrite memory entry** `exchange_range_signal.md` to state the feature
   is research-proven but canonical-layer-absent; REMOVE from "queued for
   activation" list until infrastructure is present.

---

## A2 — H2 Path C "garch_vol_pct≥70 universal"

### Memory claim (before)
> MEMORY.md line 45: "H2 BOOK CLOSED — Path C (Apr 15): garch_vol_pct≥70
> universal but NO synergy with rel_vol; ship garch alone if anything. NO CAPITAL."

### Verdict: **HETEROGENEOUS — the "universal" framing fails RULE 14**

### Evidence (from existing `docs/audit/results/2026-04-15-path-c-h2-closure.md`)

Per-session positive-cell ratio (garch_vol_pct≥70 better than <70) on
universality scan N=527 testable combos:

| Session | N cells | Positive | % positive | RULE 14 flip |
|---|---|---|---|---|
| BRISBANE_1025 | 18 | 17 | 94.4% | 5.6% |
| CME_PRECLOSE | 13 | 13 | 100.0% | 0.0% |
| CME_REOPEN | 46 | 28 | 60.9% | 39.1% FLIP |
| COMEX_SETTLE | 54 | 33 | 61.1% | 38.9% FLIP |
| EUROPE_FLOW | 54 | 43 | 79.6% | 20.4% |
| LONDON_METALS | 54 | 38 | 70.4% | 29.6% FLIP |
| NYSE_CLOSE | 18 | 9 | 50.0% | 50.0% FLIP |
| **NYSE_OPEN** | 54 | 23 | **42.6%** | **57.4% FLIP** |
| SINGAPORE_OPEN | 54 | 42 | 77.8% | 22.2% |
| TOKYO_OPEN | 54 | 50 | 92.6% | 7.4% |
| US_DATA_1000 | 54 | 32 | 59.3% | 40.7% FLIP |
| US_DATA_830 | 54 | 33 | 61.1% | 38.9% FLIP |

**Overall cell-flip rate: 166/527 = 31.5%** — EXCEEDS the 25% RULE 14 threshold.

### Key per-cell pattern

- **6 of 12 sessions (50%) exceed the 25% flip threshold** within the session.
- **NYSE_OPEN is the inverse** — garch_vol_pct≥70 is MORE OFTEN NEGATIVE than
  positive on NYSE_OPEN cells (23 pos / 31 neg). The existing per-(inst, session)
  table in `2026-04-15-garch-all-sessions-universality.md` confirms: MGC NYSE_OPEN
  shows 27 strong-neg at >0.15 inverse lift.
- **Asia sessions (BRISBANE_1025, TOKYO_OPEN, SINGAPORE_OPEN) dominate the
  universal signal** — 92-94% positive. These four sessions alone account for
  the bulk of the positive-cell count.

### Corrected framing

- NOT universal. Signal is Asia/closed-market heavy; flips on US sessions.
- "H2 validated for MNQ COMEX_SETTLE O5 RR1.0 long" (from
  `2026-04-15-path-c-h2-closure.md` § Step 3 composite) remains valid as a
  SINGLE-CELL finding at N=198 (K_global=14261 DSR=0.001, not robust).
- The existing "NO CAPITAL" status already dampens the impact, but the
  "universal" framing in MEMORY.md propagated session-agnostic.

### Recommended action

1. Rewrite MEMORY.md line 45 to: "H2 Path C: garch_vol_pct≥70 SESSION-HETEROGENEOUS
   (NYSE_OPEN inverse, Asia-heavy positive); not universal. NO CAPITAL."
2. Any future garch-overlay proposal must be per-session pre-reg, not universal.
3. `mechanism_priors.md` (if it mentions universal garch overlay) needs the
   same correction.

---

## A3 — comprehensive scan 13 K_global survivors "universal volume confirmation"

### Memory claim (before)
> MEMORY.md line 47: "Comprehensive scan (Apr 15): 14,261 cells; volume
> confirmation universal."

### Verdict: **PARTIALLY HETEROGENEOUS — single-feature, session-concentrated**

### Evidence (from `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`)

**All 13 K_global BH-FDR survivors use the SAME feature: `rel_vol_HIGH_Q3`.**
Not "universal volume" — universal presence of ONE specific rel_vol percentile
binning across multiple lanes.

Session distribution of the 13 survivors:

| Session | Count | Directions |
|---|---|---|
| SINGAPORE_OPEN | 4 | 2 short RR1.0, 1 short RR1.5, 1 long RR1.0, 1 long RR1.5 |
| CME_PRECLOSE | 3 | 3 short |
| LONDON_METALS | 2 | 2 short |
| COMEX_SETTLE | 2 | 2 short (MES + MNQ, Jaccard 0.491 → effectively 1 signal per 2026-04-19 overlap audit) |
| TOKYO_OPEN | 2 | 2 long |
| BRISBANE_1025 | 1 | 1 long |
| EUROPE_FLOW | 1 | 1 long |
| **NYSE_OPEN** | **0** | — |
| **NYSE_CLOSE** | **0** | — |
| **US_DATA_830** | **0** | — |
| **US_DATA_1000** | **0** | — |
| **CME_REOPEN** | **0** | — |

- **5 of 12 sessions (42%) have zero K_global survivors.** NYSE/US_DATA cluster
  is entirely absent.
- Sessions with any survivor: 7/12.
- Correcting for cross-instrument redundancy (MES×MNQ COMEX_SETTLE share 67% of
  fires per 2026-04-19 overlap audit), effective K_eff ≈ 11-12, not 13.

### Corrected framing

- NOT "universal volume confirmation" in the sense memory implies.
- IS "rel_vol_HIGH_Q3 broadly present in Asia/London/COMEX/CME_PRECLOSE sessions;
  absent at US equity-cash sessions".
- Mechanism interpretation: rel_vol (volume at ORB vs session-average-historical)
  likely captures "above-average engagement" which is a precondition for ORB
  follow-through. That it doesn't fire on NYSE cash sessions may reflect that
  those sessions ALREADY have consistently high volume, so rel_vol_HIGH_Q3
  doesn't discriminate.

### Recommended action

1. Rewrite MEMORY.md line 47 from "volume confirmation universal" to
   "rel_vol_HIGH_Q3 robust at 7/12 sessions; absent at US cash sessions".
2. Any future overlay proposal stating "volume confirmation universal" must
   cite this distribution, not the K_global count alone.
3. The rel_vol feature itself is still a valid filter candidate for the
   7-session subset; deployment decisions should pre-reg which session a
   new rel_vol overlay targets.

---

## B3 — break_quality universal pooled null

### Memory claim (before)
> `break_quality_research.md`: "Universal Pooled Results (N = 18,000-27,000)
> All instruments (MGC, MNQ, MES) + all sessions pooled: Compression p=0.522,
> Explosion p=0.450, Break/ORB p=0.579, Bar count p=0.869. None even approach
> significance. Break microstructure carries zero information about post-entry
> outcomes." Framed as NO-GO definitive.
>
> Memory framing line: "Blanket rules with massive pooled N are the only
> honest way to test microstructure. Session-specific findings from 164 tests
> are noise unless they survive universal pooling."

### Verdict: **NO-GO CONCLUSION HOLDS — memory FRAMING is outdated**

### Analysis

The NO-GO conclusion (break_quality microstructure features carry zero
information) is supported by pooled N=18-27K where even small effect sizes
would reach significance. Pooled p=0.5 at that N means the underlying effect
magnitude is genuinely ~0, not "obscured by heterogeneity."

RULE 14 heterogeneity flag: could break_quality have hidden session-specific
edge that pooling washes out? Two checks:

1. **Memory cites "narrow findings":** MNQ CME_CLOSE reversed explosion p=0.003
   N=118; MNQ 1100 compression p=0.006-0.010. These are tested at K=164 (164
   combinations). BH-FDR at q=0.05 requires p < 0.00031 for a single survivor
   → these narrow findings **do not survive multiple testing correction**.
2. **Conservative claim:** even without multiple-testing correction, N=118
   single-cell effects in a K=164 search are properly interpreted as noise
   until OOS-confirmed. Same methodological rigor as per-lane testing
   elsewhere in the project.

So: NO-GO stands. But the FRAMING in the memory file — "Blanket rules with
massive pooled N are the only honest way to test microstructure" — contradicts
today's RULE 14. The correct framing is: "Pooled N at this scale can detect
tiny effect sizes; pooled null plus BH-FDR on per-cell results closes the
hypothesis book. Pooling alone is not sufficient in general — see RULE 14
for hiding-of-heterogeneity caveat."

### Recommended action

1. Rewrite "Key Lesson" section of `break_quality_research.md` to cite RULE 14
   as modifier (pooled N is sufficient for null-confirmation on microstructure
   where effects are genuinely small; not a general methodology recommendation).
2. NO-GO status: CONFIRMED at original scope. No re-audit triggered.

---

## B4 / B5 — exit-timing universal — DEFERRED

`exit_rules_and_timing.md` cites pooled claims across 404K winning trades
spanning MGC+MNQ+MES × all sessions:
- "Recovery rates stable at 17-23% across all thresholds (robust mechanism)"
- "ALL sessions NET POSITIVE at current thresholds"
- "E3 is universally negative at 1000"

These would benefit from per-session breakdown under RULE 14, but the scope
is large (multiple timeout thresholds × multiple sessions × 3 instruments ×
3 entry models × cumulative recovery buckets). Responsible audit requires
its own pre-reg and at least half a session of dedicated work.

**DEFERRED to next-session mandate CURRENT-D.** Added to
`next_session_mandates_2026_04_20.md`.

---

## Summary of Phase 3 outcomes

| Claim | Verdict | Memory action |
|---|---|---|
| A1 exchange_range | **BLOCKED — feature unpopulated** | Remove from "queued for activation"; open infrastructure ticket |
| A2 garch_vol_pct≥70 universal | **HETEROGENEOUS (31.5% flip, NYSE inverse)** | Rewrite as session-heterogeneous |
| A3 volume confirmation universal | **SEMI-HETEROGENEOUS (0 survivors at 5 of 12 sessions)** | Rewrite as rel_vol_HIGH_Q3 robust at 7/12 sessions |
| B3 break_quality NO-GO | **NO-GO HOLDS, framing outdated** | Minor memory edit |
| B4/B5 exit timing | **DEFERRED to CURRENT-D** | Logged in next-session mandates |

**Zero impact on live trades.** No current deployment depends on any of these
"universal" framings. The corrections are doctrinal — preventing future
pigeon-holing — not operational.

**Three new drift-check candidates surfaced:**

1. Any `ALL_FILTERS` entry whose required column is 0% populated in canonical
   `daily_features` → block (A1 discovery)
2. Memory / doc assertions of "universal" must carry per-cell breakdown or
   they're advisory-only (RULE 14 enforcement)
3. PIT_MIN specifically — either backfill `pit_range_atr` and re-audit, or
   deregister the filter until backfill is complete
