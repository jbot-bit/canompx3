# Stage: Repair MNQ live-bar volume corruption in bars_1m + harden tick guard

task: The live bot's BarPersister wrote 1037 rows into canonical bars_1m with source_symbol='MNQ' and garbage billion-scale volume (broker-feed volume field carries non-volume junk; no upper sanity bound in bar_aggregator). Delete the corrupt rows, re-ingest the only-corrupt days from Databento front-month, rebuild daily_features+orb_outcomes for affected days, re-run the Tokyo SR recheck on clean data, and add an upper volume sanity cap to bar_aggregator._validate_tick so a live-feed glitch can never again pass billion-scale volume into the canonical table.
mode: IMPLEMENTATION

## Scope Lock
- trading_app/live/bar_aggregator.py
- tests/test_trading_app/test_bar_aggregator.py
- gold.db
Note: gold.db edits = surgical DELETE of source_symbol='MNQ' corrupt bars_1m rows + daily_features/orb_outcomes rebuild for affected MNQ days. bar_aggregator.py = add upper volume sanity bound in _validate_tick.

## Blast Radius
- bars_1m: surgical DELETE of 1037 rows WHERE symbol='MNQ' AND source_symbol='MNQ'. Correct MNQM6 rows that coexist on 2026-04-24 (540 good) and 2026-05-25 (100 good) are UNTOUCHED. The 7 only-corrupt days (04-25, 05-27, 06-03, 06-05, 06-06, 06-08, 06-09) become gaps, then re-filled by Databento front-month re-ingest.
- trading_app/live/bar_aggregator.py: _validate_tick gains an upper volume ceiling (reject + log a tick whose volume exceeds a sane per-bar cap). Capital path — but strictly tightens validation (rejects garbage); cannot reject legitimate micro-futures volume (caps set ~100x the observed clean max of ~27k).
- Downstream rebuild (this stage): daily_features + orb_outcomes for affected MNQ days. C11/C12 envelopes invalidated for MNQ → regen tracked separately if needed.
- Re-ingest: cost-guarded Databento fetch (FREE-probed before any billable call) via refresh_data.py / ingest_dbn.
- Tokyo SR recheck (MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075) re-run on clean data → real verdict (prior verdict VOID — contaminated by 06-08/06-09 billion-volume bars passing the vol>=4000 filter).
- NOT in scope this stage (tracked follow-up): architectural separation of live-captured bars from canonical Databento bars_1m (source_symbol tagging / separate live_bars table) + a drift check rejecting bars_1m rows above the volume cap.

## ROOT CAUSE (proven 2026-06-12)
- bar_persister.py:106 writes source_symbol = self.symbol = 'MNQ' (the live symbol, not the dated contract).
- Every legit bars_1m row in 7yr history has source_symbol = a dated contract (MNQM6...) with volume <= ~27k. The ONLY source_symbol='MNQ' rows are the 1037 corrupt ones (2026-04-24..06-09, maxvol 1.63 BILLION).
- bar_aggregator._validate_tick rejects only NEGATIVE volume — no upper bound — so broker-feed garbage volume accumulates into the bar and lands in canonical bars_1m, silently corrupting the research layer (daily_features volume features → orb_outcomes filters).

## DONE CRITERIA
- bars_1m: zero source_symbol='MNQ' rows; affected days re-ingested or left as clean gaps; verify maxvol sane.
- daily_features + orb_outcomes rebuilt for affected MNQ days.
- bar_aggregator volume cap added + targeted test (billion-volume tick rejected, legit tick passes).
- Tokyo SR recheck re-run on clean data + verdict stated.
- check_drift.py passes.

## EXECUTION LOG (2026-06-12)
- Deleted 1037 source_symbol='MNQ' corrupt rows (maxvol 1.63B). Coexisting MNQM6 rows on 04-24/05-25 survived (verified).
- Re-downloaded MNQ.FUT ohlcv-1m 2026-04-24..06-11 via canonical download_dbn (front-month parent resolution); ingest_dbn --resume filled gaps. All days 04-24..06-11 now src=MNQM6, maxvol <=27k. Zero source_symbol='MNQ' remaining.
- Rebuilt 5m bars + daily_features (O5/O15/O30) + orb_outcomes (--force) for MNQ 2026-04-24..06-11.
- TOKYO SR RECHECK ON CLEAN DATA (MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15, N=805):
  last-10=-0.41R, last-20=-0.12R, last-30=-0.03R. VERDICT: SR-ALARM IS REAL — cold regime confirmed, Tokyo head stays PAUSED. (Pre-repair corrupt computation falsely showed last-10=+0.17R because 06-08/06-09 billion-volume junk passed the vol>=4000 filter; that flip was a data artifact.) NO downstream regen / NO live-route. Step 4 correctly NOT triggered.
- HARDENING: bar_aggregator._validate_tick now rejects per-tick volume > _MAX_TICK_VOLUME=1_000_000 (defense-in-depth below ProjectX _cum_to_delta Defect-A fix 2026-06-10). +2 regression tests (billion-vol rejected, 50k legit accepted). 24/24 bar_aggregator tests pass.
- ROOT CAUSE CONFIRMED: ProjectX feed emits cumulative session volume; pre-2026-06-10 it was fed to BarAggregator as per-tick delta → accumulated to billions. The 06-10 _cum_to_delta fix stopped it going forward but left historical bars corrupt and no aggregator-level bound. Corruption window 04-24..06-09 = exactly pre-fix.

## FOLLOW-UP (tracked, NOT this stage)
- Architectural: stop the live bot writing into canonical bars_1m (BarPersister source_symbol='MNQ'), OR tag live bars so the research layer excludes them, OR a separate live_bars table. A live-feed glitch should never silently corrupt backtests. (bar_persister.py:106)
- Drift check: reject any bars_1m row with per-bar volume above a sane cap (catch corruption at the table, not just the feed).

## STATUS: DONE pending drift pass + commit
