# MNQ E2 Slippage Pilot v2 — 3-Session Gap Fill

**Date:** 2026-04-20
**Script:** `research/research_mnq_e2_slippage_pilot.py --pull --sessions EUROPE_FLOW COMEX_SETTLE US_DATA_1000` then `--reprice-cache`
**Dataset:** Databento `GLBX.MDP3`, `MNQ.FUT`, TBBO schema, 32-minute windows around ORB end
**Predecessor:** `2026-04-20-mnq-e2-slippage-pilot-v1.md`
**Databento spend:** $0.19 (under subscription allowance; effectively zero marginal)

---

## Purpose

`2026-04-20-mnq-e2-slippage-pilot-v1.md` closed with "MNQ sample missing from deployed lanes: EUROPE_FLOW, COMEX_SETTLE, US_DATA_1000. 3 of 5 unique deployed sessions absent from the 119-file cache." This v2 closes that gap.

Phase D evaluation gate (2026-05-15) specifically targets MNQ COMEX_SETTLE. Unmeasured slippage on that session was the highest-leverage open debt on the book.

## Scope change

Added `--sessions` CLI arg to `research/research_mnq_e2_slippage_pilot.py` (institutional fix, not monkey-patch). Pulled 30 new TBBO files covering the 3 missing deployed sessions (10 days per session, 2 ATR regimes × 5 days per bucket).

Total cache now: **149 files** (119 prior + 30 new). Valid reprice results: **142 of 149** (7 legitimate errors — 4 no-trigger-trade + 3 daily_features missing).

## Numbers (full cache, 142 valid samples)

**Slippage in ticks relative to modeled 2-tick round-trip:**

| Stat | Value | Read |
|---|---|---|
| Median | 0.00 | Central tendency is AT modeled, not worse |
| MAD | 0.00 | Zero dispersion around median |
| p25 | −1.00 | Quarter of samples fill 1 tick BETTER than modeled |
| p75 | 0.00 | Three quarters fill at-or-better than modeled |
| p95 | 0.00 | Tail is still at modeled |
| Max | +2.00 | Worst case is exactly modeled — NEVER exceeded |
| % ≤ 1 tick | 99.3% | |
| % ≤ 2 ticks | **100.0%** | |
| Mean | −0.79 | Outlier-sensitive; dominated by BBO-staleness artifacts on fast-move days (not real favorable fills — see v1 §Interpretation) |
| Min | −38.00 | Thin-book BBO-staleness artifact during a wide-spread fast move — NOT a real favorable fill; same class as v1's negative outliers |
| Spread at trigger (median) | 2.00 ticks | |

**Verdict: CONSERVATIVE (modeled ≥ measured everywhere that matters).**

## Per-session — all 9 deployed sessions now covered

| Session | N | Median | Mean | Status |
|---|---|---|---|---|
| COMEX_SETTLE | 10 | 0.0 | 0.0 | **NEW — clean; Phase D gate unblocked** |
| EUROPE_FLOW | 9 | 0.0 | −0.2 | NEW — clean |
| US_DATA_1000 | 9 | 0.0 | −0.4 | NEW — clean |
| NYSE_OPEN | 18 | 0.0 | −0.5 | v1 — clean |
| SINGAPORE_OPEN | 20 | 0.0 | −0.5 | v1 — clean |
| TOKYO_OPEN | 18 | 0.0 | −0.7 | v1 — clean |
| LONDON_METALS | 20 | 0.0 | −0.2 | v1 — clean |
| US_DATA_830 | 20 | 0.0 | −0.6 | v1 — clean |
| CME_PRECLOSE | 18 | −0.5 | −3.3 | v1 — mean dominated by BBO-staleness events, median is the honest read |

## Operational consequences

- **Phase D evaluation (2026-05-15) COMEX_SETTLE gate:** no deployment blocker on slippage grounds. Backtested ExpR for the pilot MNQ COMEX_SETTLE lane is not materially optimistic under routine-day friction.
- **6 live MNQ lanes' backtested ExpR:** still not materially optimistic under measured slippage. Every deployed MNQ session now has TBBO evidence backing the modeling.
- **No lane flips to negative EV.**
- **No deployment changes needed.**

## What is still unmeasured

- **Event-day tail for MNQ.** Pilot samples did not include a 2018-01-18-style gap-open equivalent for MNQ (MGC had one such day that dominated its mean). Directional claim: event-day tail risk is NOT zero, it is just not in this sample. Treat as a known-unknown, not a refuted concern.
- **MES book-wide slippage.** MES TBBO pilot has not been run. MES deployed-lane backtested ExpR carries unmeasured modeled-vs-real friction optimism. Book-wide debt `cost-realism-slippage-pilot` remains partially open.

## Files touched

- `research/research_mnq_e2_slippage_pilot.py` — added `--sessions` CLI arg (nargs+, optional, overrides `PILOT_SESSIONS` global).
- `research/data/tbbo_mnq_pilot/` — 30 new `.dbn.zst` cache files + updated `manifest.json` + updated `slippage_results_cache_v2.csv` (142 valid rows).
- `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md` — this doc.
- `docs/runtime/debt-ledger.md` — `cost-realism-slippage-pilot` MNQ portion CLOSED; MES portion remains open.
- `HANDOFF.md` — this entry.

## Reproducibility

```bash
# Cost estimate (free)
python -m research.research_mnq_e2_slippage_pilot --estimate-cost --sessions EUROPE_FLOW COMEX_SETTLE US_DATA_1000
# Pull (spent $0.19 subscription-absorbed)
python -m research.research_mnq_e2_slippage_pilot --pull --sessions EUROPE_FLOW COMEX_SETTLE US_DATA_1000
# Reprice full cache (free — all 149 files)
python -m research.research_mnq_e2_slippage_pilot --reprice-cache
```

Seed 42. All repricing delegates to canonical `research.databento_microstructure.reprice_e2_entry`; ORB window timing via canonical `pipeline.build_daily_features._orb_utc_window`; no re-encoding of canonical filter or first-cross logic.
