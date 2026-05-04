# HTF / Prior-Structure / Confluence Repo Audit

**Date:** 2026-04-23
**Scope:** higher-timeframe, prior-day, prior-session, confluence, conditioner, sizing, allocator, routing, and execution-translation research in repo canon
**Grounding:** `bars_1m`, `daily_features`, `orb_outcomes`; local rule files; local literature-backed criteria

## A. Verdict

**Partial**

- The repo has real **feature substrate** for HTF / prior-structure (`prev_week_*`, `prev_month_*`) and real **role-translation substrate** for routing / sizing / allocator work, but it does **not** have a broad validated HTF/confluence system already promoted into the current live book. Runtime truth here is the allocator + SR-liveness path, not the persisted JSON snapshot alone. See `pipeline/build_daily_features.py`, `pipeline/check_drift.py`, `trading_app/lane_allocator.py`, `trading_app/prop_profiles.py`, `docs/runtime/lane_allocation.json`, and `docs/audit/results/2026-04-20-allocator-scored-tail-audit.md`.
- No reviewed HTF / prior-structure / confluence survivor is currently `CONNECTED` end-to-end from canonical feature build -> clean research verdict -> runtime route -> active live auto-book deployment. The surviving branches are substrate-only, shelf-only, research-only, or runtime-blocked. See `trading_app/config.py`, `docs/runtime/decision-ledger.md`, `docs/runtime/lane_allocation.json`, `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`, and `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-preaudit.md`.
- The strongest prior-day geometry survivors are alive only as **exact shelf rows / hypothesis-scoped routes**, not as current auto-routed live lanes. See `trading_app/config.py`, `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`, `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`, and `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-preaudit.md`.
- Broad simple HTF families were fairly tested and killed; broad multi-timeframe chaining was structurally killed before empirical testing. See `docs/audit/results/2026-04-18-htf-path-a-prev-week-v1-scan.md`, `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md`, and `docs/audit/hypotheses/2026-04-15-multi-timeframe-chain-full-scope-nogo.md`.

## B. What is real

- **HTF feature infrastructure is real, but only as infrastructure.**
  `prev_week_*` / `prev_month_*` are canonically built and drift-guarded. This is `VALID` as substrate and `CONNECTED` only to the feature layer, not to any surviving live HTF edge. See `pipeline/build_daily_features.py` and `pipeline/check_drift.py`.
- **Exact prior-day geometry shelf survivors are real.**
  `PD_DISPLACE_LONG`, `PD_CLEAR_LONG`, and `PD_GO_LONG` are registered in `ALL_FILTERS` for hypothesis/routing use and are canonically routed only on exact MNQ parent lanes:
  - `MNQ US_DATA_1000` -> `PD_DISPLACE_LONG`, `PD_CLEAR_LONG`, `PD_GO_LONG`
  - `MNQ COMEX_SETTLE` -> `PD_CLEAR_LONG`
  This pipe is `CONNECTED` through feature logic and registry, but still `SHELF_ONLY` at live deployment. See `trading_app/config.py`, `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`, and `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`.
- **Allocator/routing translation is real as a repo capability.**
  The allocator stack, profile translation, and live profile loading from `lane_allocation.json` are real and currently active, but `lane_allocation.json` is only a persisted snapshot. Current routing truth is conditional on SR liveness and allocator replay. Even with that caveat, the reviewed runtime surfaces do not show HTF/prior-day geometry rows in the active MNQ auto book. See `trading_app/lane_allocator.py`, `trading_app/prop_profiles.py`, `docs/runtime/lane_allocation.json`, and `docs/audit/results/2026-04-20-allocator-scored-tail-audit.md`.
- **One exact deployed-lane confluence overlay remains alive enough to continue.**
  On the existing `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane, `H04_CMX_SHORT_RELVOL_Q3_AND_F6` is `CONTINUE`, while `H03_CMX_SHORT_RELVOL_Q3` and `H01_NYO_SHORT_PREV_BEAR` are `PARK`, not kill. This matters because it proves the repo has not honestly killed all overlay/confluence work beyond naive new-on-new stacking. See `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`.
- **One sizer-rule branch survived narrowly enough to matter.**
  PR48 frozen rel-vol sizing is still alive as a role question:
  - `MGC` = `SIZER_DEPLOY_CANDIDATE`
  - `MES` = `SIZER_ALIVE_NOT_READY`
  This is `RESEARCH_SURVIVOR`, not current live-book adoption. See `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`.
- **The surviving sizing story is role-translation, not broad per-lane proof.**
  Broad R5 cross-lane replication on 6 deployed lanes produced `0` BH-FDR survivors at `K=36`. Do not over-read the later sizing-replay wins as if the repo had already proved a universal per-lane GARCH conditioner. See `docs/audit/results/2026-04-15-r5-sizer-cross-lane-replication.md`.
- **Prior-session carry is not globally dead as an information channel.**
  What survived is narrower:
  - binary gate = dead
  - late-day `E1` / `COMEX_SETTLE` two-state carry signal = `CONDITIONAL`, local only
  See `docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md` and `docs/audit/results/2026-04-16-carry-encoding-exploration.md`.

## C. What is dead

- **Full-scope multi-timeframe indicator-on-indicator chaining is dead.**
  This is a structural `WRONG / NO-GO` verdict, not merely “not yet tested.” See `docs/audit/hypotheses/2026-04-15-multi-timeframe-chain-full-scope-nogo.md`.
- **Simple HTF Path A break-aligned families are dead as families.**
  - `prev-week v1` = `FAMILY KILL`
  - `prev-month v1` = `FAMILY KILL`
  These were fairly tested as clean simple HTF break filters and failed on the actual gates, not just on one weak cell. See `docs/audit/results/2026-04-18-htf-path-a-prev-week-v1-scan.md` and `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md`.
- **Binary prior-session carry hard-gate / veto framing is dead.**
  The veto-pair sign is directly contradicted on the validated shelf. This kills the binary hard-gate framing, not every possible carry role. See `docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md`.
- **E2 / E3 carry encodings are dead on the audited shelf.**
  Carry encoding follow-up explicitly parks `E2` and `E3`. See `docs/audit/results/2026-04-16-carry-encoding-exploration.md`.
- **New-on-new confluence AND-stacking is dead as a class.**
  The repo’s NO-GO is specific: new confluence feature pairs added little IS edge and collapsed OOS. This does **not** kill existing deployed filter + new overlay, veto-style use, or allocator-style use. See `docs/STRATEGY_BLUEPRINT.md`.
- **The exact bridge avoid cell is dead.**
  `F3_NEAR_PIVOT_50` exact avoid-state on the bridge lane is closed dead; the broader positive geometry families survived separately. See `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`.
- **The OVNRNG allocator-router line is dead and the early positive doc is superseded.**
  `ROUTER_HOLDS_OOS` was later retracted by rolling 4-fold CV. Current canonical truth is `ROUTER_BRITTLE — DEAD`, not the earlier single-fold result. See `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md` and `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md`.

## D. What is untested

- **`US_DATA_1000` same-session cross-aperture execution translation** is still the highest-value honest gap.
  The signal layer survived; the unresolved question is runtime coexistence / size-down / shadow translation under the actual live book. See `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-preaudit.md` and `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`.
- **Existing deployed filter + new HTF/structure overlay in veto/score form** remains uncleanly tested.
  The repo fairly killed naive new-on-new AND-stacking, but that is not the same question as residual edge on top of an existing deployed parent lane or a veto-style overlay. One exact deployed-lane branch is still `CONTINUE`, and two are `PARK`, which means this bucket is partially tested, not closed. See `docs/STRATEGY_BLUEPRINT.md` and `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`.
- **Prior-session carry as portfolio-context / sizing input** remains admissible and unclosed.
  W2e explicitly left portfolio-context, sizing-modifier, soft confluence, and narrow local context forms open. See `docs/audit/results/2026-04-16-garch-w2e-prior-session-carry-audit.md`.
- **`1H` / `4H` candle framing as a distinct feature class** is still `UNSUPPORTED`.
  I found no closed canonical result doc in this read set proving that 1H/4H candles are either genuinely new information or just a noisier restatement of already-killed structure.
- **MES/MGC as conditioner / confluence / allocator on MNQ** is still not honestly closed in the reviewed canonical result set.
  There is cross-instrument sizer and stale-filter residue (`PR48`, GARCH, `X_MES_ATR60` remnants in allocator-tail audits), but I did not find a clean closed-result branch proving this exact cross-instrument conditioner question either way. Treat it as `UNSUPPORTED / honest gap`, not as survivor or kill. See `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md` and `docs/audit/results/2026-04-20-allocator-scored-tail-audit.md`.

## E. What to trash / archive

- **Archive stale HTF handoff / queue surfaces that still imply Path A simple HTF is open.**
  `docs/handoffs/2026-04-19-htf-session-handover.md` and `docs/plans/2026-04-18-post-a4c-ev-queue-reset.md` now overstate “open HTF” compared with the later family-kill docs and the bridge-closure docs.
- **Do not treat `docs/runtime/lane_allocation.json` as unconditional live truth in audit narratives.**
  The allocator scored-tail audit shows replay-vs-persisted drift caused by SR-liveness updates. Keep the JSON, but stop using it as a standalone proof surface without the allocator replay context. See `docs/audit/results/2026-04-20-allocator-scored-tail-audit.md`.
- **Archive or explicitly relabel the MES EUROPE_FLOW HTF shadow ledger unless it is resumed with a fresh bounded pre-reg.**
  `docs/audit/shadow_ledgers/htf-mes-europe-flow-long-skip-rule-ledger.md` is easy to misread as an alive branch after the broader HTF family kills.
- **Relabel superseded OVNRNG routing claims so the repo cannot cite a retracted positive as current truth.**
  `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md` must not stand alone without its rolling-CV retraction in `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md`.
- **Archive stale cross-instrument `X_MES_ATR60` residue from allocator narratives unless a new bounded prereg revives it.**
  In the allocator scored-tail audit those rows are explicit `STALE` / `N=0`, which means they are not admissible evidence for current MES-as-conditioner claims. See `docs/audit/results/2026-04-20-allocator-scored-tail-audit.md`.
- **Trash the idea that “HTF/confluence” is one single bucket.**
  The repo’s own later docs show this was a framing error:
  - simple HTF filter family
  - prior-day geometry signal family
  - binary carry gate
  - confluence AND-stack
  - allocator / routing translation
  are different questions and should not live under one vague “HTF works / HTF dead” story.

## F. Best next step

- **One bounded audit only:** `US_DATA_1000` same-session cross-aperture execution translation / shadow test for the surviving prior-day geometry rows against the current MNQ live profile.
- **Why this is highest-EV:** the canonical signal work is already mostly done; the blocker is runtime truth, not another discovery pass. See `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`, `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`, and `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-preaudit.md`.
- **Shortest path to truth:** do not reopen broad HTF discovery, do not reopen full-scope chains, do not shop more filters. Answer the live translation question and either route, shelf, or park the surviving geometry rows.

## Verdict / decision

- **Decision:** treat the repo as `Partial`, not as an already-live HTF/confluence system
- **Operational implication:** do not reopen broad HTF discovery or confluence shopping from this audit
- **Action implication:** route follow-through only through the bounded `US_DATA_1000` execution-translation question

## Reproduction / outputs

- Grounding surfaces reviewed: `bars_1m`, `daily_features`, `orb_outcomes`, local rule files, and local result docs cited above
- Runtime surfaces reviewed: `trading_app/config.py`, `trading_app/lane_allocator.py`, `trading_app/prop_profiles.py`, `docs/runtime/lane_allocation.json`, `docs/runtime/decision-ledger.md`
- Output artifact: this repo audit document plus the supersedure note added to `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md`
- No DB writes, no `validated_setups` mutation, no runtime config mutation

## Caveats / limitations

- This is a document-grounded repo audit, not a new backtest run
- Claims are limited to the local canonical surfaces reviewed in this pass; where a clean closed result was not found, the label is `UNSUPPORTED` rather than inferred closed
- The audit distinguishes substrate, shelf, and live routing; absence from the active live auto-book should not be misread as proof that an information channel is globally dead
