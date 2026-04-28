---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Post-A4c EV Queue Reset

**Date:** 2026-04-18
**Trigger:** `HANDOFF.md` lines 50-52 post-A4c directive: *"Reset the queue. Next action is a fresh ranked list of highest-EV open items, NOT more garch allocator work. The garch-family allocator path (A4a / A4b / A4c) has produced three null stages with increasing rigor; declare garch allocator research paused until a meaningfully different mechanism is pre-registered."*
**Author:** Claude
**Scope:** planning doc only. No pre-reg, no code, no DB writes.

---

## Summary

Queue reset after closing 2026-04-18 grounding audit and A4c null verdict. **Top recommendation:** park active research for 2-3 weeks while Phase D D-1 signal-only shadow accumulates real-time forward OOS. Use the interim window to write the 2 highest-diversity candidate stubs (HTF-level features and a single horizon T0-T8 candidate) rather than run new scans. No live-capital changes; no new pre-reg fires.

---

## Inventory of open items post-A4c

Enumerated from `HANDOFF.md` active shelf + `MEMORY.md` open-queue entries + master audit's "trustworthy-right-now" list.

| # | Item | Source | Status today |
|---|---|---|---|
| 1 | **HTF level features (Path A)** — prior_week / prior_month break-level features | `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md` (stub); MEMORY line 18 "DEFERRED" | Blocked — `prev_week_*` / `prev_month_*` features not yet in `daily_features` |
| 2 | **Phase C E_RETEST entry model** | `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` (stub) | Blocked — requires `outcome_builder` schema extension (heavy, 2-3 week infra) |
| 3 | **Phase D D-2 live deployment** (volume size-scaling pilot) | `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`; D-1 contract-lock `4fd6c264` | Gated — D-1 signal-only shadow runs 4 weeks from 2026-04-17 → D-2 gate eval no earlier than 2026-05-15 |
| 4 | **Phase E non-ORB strategy class (SC2)** | `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md` (stub) | Blocked — requires Dalton/Murphy literature (not in `/resources`) |
| 5 | **Horizon T0-T8 candidates (non-volume)** — `ovn_range_pct_GT80` MES × {LONDON_METALS, EUROPE_FLOW, US_DATA_830}; `is_monday` MNQ BRISBANE_1025; `dow_thu` MNQ COMEX_SETTLE; `ovn_took_pdh` SKIP MES/MNQ COMEX_SETTLE | MEMORY lines 35, 22-27 "HORIZON T0-T8 — 5 non-volume candidates" | Available — each has untested T0-T8 battery; none deployed |
| 6 | **HLZ-2015 mechanical gate promotion** (closes audit IMP-1 class-level) | Master audit doc IMP-1 | Deferred — user directive: "doc-enforced for now; mechanical only if HLZ PDF arrives or 3.00 ≤ t < 3.79 candidate lands" |
| 7 | **Check 45 auto-refresh hook** (closes audit NFFU-5 class-level) | Master audit doc NFFU-5; review action item 3 | Deferred — manual refresh tool at `1a0a4a24` is idempotent; class-level gap noted but not burning |
| 8 | **Stale `experimental_strategies` rebuild** × 3 instruments | `project_pulse` `decaying` items | Maintenance — `/rebuild-outcomes MES/MGC/MNQ` |
| 9 | **SR ALARM review follow-through** on 3 WATCH lanes | `project_pulse` `sr_summary` | `reviewed_watch_count: 3, unresolved_alarm_count: 0` — no action today |
| 10 | **FX ORB reopen** | `docs/audit/results/2026-04-17-fx-orb-closure.md` | CLOSED — do not re-open without brand-new pre-reg + materially wider data surface + economic rationale |

---

## Ranking axes

Each item scored on four axes (qualitative):

1. **Mechanism diversity vs garch family** — the post-A4c directive is "meaningfully different mechanism". A5=high diversity, A1=same family.
2. **Infrastructure cost** — how much pipeline/schema work before first result. Cost1=no build, Cost5=multi-week build.
3. **Literature grounding today** — is it backed by a Tier 1 extract already? Grd1=no, Grd5=verbatim Tier 1 in hand.
4. **EV estimate** — HANDOFF/MEMORY-quoted rough magnitudes where available. Treat as directional, not citable.

| # | Item | Diversity | Cost | Grd | EV (rough) | Runnable today? |
|---|---|---:|---:|---:|---|:---:|
| 1 | HTF level features | 5 | 3-4 (new `prev_week_*` / `prev_month_*` build) | 3 (Fitschen intraday supports level-driven moves; no extract for HTF-level specifically) | Unknown — no scan yet | No (infra gated) |
| 2 | Phase C E_RETEST | 5 | 5 (outcome_builder rewrite) | 2 (no direct extract for limit-on-retest; concept from general price-action literature) | Unknown | No (infra gated) |
| 3 | Phase D D-2 live | 1 (same volume-sizing family as shadow) | 1 (already locked via D-1) | 5 (Carver Ch.9-10 in Tier 1) | MEMORY line 143: "+12-15% per-lane Sharpe. $10-12K/yr/ct → $13-14K/yr/ct on MNQ" realistic | Gated (≥2026-05-15) |
| 4 | Phase E non-ORB (SC2) | 5 | 5 (new entry model class + new backtest harness) | 1 (Dalton/Murphy not in `/resources`) | Unknown | No (lit gated) |
| 5a | **H1 MES LONDON_METALS O30 RR1.5 long `overnight_range_pct≥80` — signal-only shadow** | 4 (overnight range orthogonal to volume + garch) | 1 (shadow-log pattern already established by Phase D D-1) | 4 (Fitschen supports path-of-least-resistance; overnight-range literature partial) | H1 T0-T8 verdict = CONDITIONAL (7P/1F), only fail was T3 WFE=1.33 driven by N_OOS_on=11 (MEMORY line 23). Per-cell lift not quoted in MEMORY; **EV UNQUANTIFIED — requires re-query** before deployment decision. Not same cell as Phase D. | **Yes** (shadow only) |
| 5b | Horizon H3/H4/H5 re-audit | 3 (DOW / pivot-take orthogonal to volume but overlap with ORB timing) | 1 (reuse harness) | 2 (DOW effects literature is equity-specific; weak transfer) | All CONDITIONAL fail T3 thin-OOS only (MEMORY line 24). Re-running T0-T8 today would produce same thin-OOS fail — needs months of new OOS, not a re-scan. | No (waiting for OOS) |
| 5c | Composite rel_vol × garch_vol_pct on H2 cell | — | — | — | **Already closed.** `docs/audit/results/2026-04-15-path-c-h2-closure.md:104-105` verbatim: *"Synergy: ExpR(both) - max_marginal = -0.043 -> NO SYNERGY / SUBSUMED — composite is not additive."* MEMORY line 27 "next: composite" entry is stale. No action. | No (closed) |
| 6 | HLZ mechanical gate | 2 (audit housekeeping, not research) | 2 (drift check + tests) | 5 (grounded via IMP-1 + Chordia indirect) | Preventative only; no revenue lift | Yes (but user deferred) |
| 7 | Check 45 auto-refresh | 2 (audit housekeeping) | 2 (pipeline hook + tests) | 5 (Tier 1-irrelevant — pure engineering) | Preventative only; no revenue lift | Yes (but user deferred) |
| 8 | `/rebuild-outcomes` × 3 | 1 (pure maintenance) | 1 (compute time only) | 5 (N/A) | Nil; fixes stale read | Yes |
| 9 | SR ALARM re-review | 1 (monitoring) | 1 (review only) | 5 (institutional canon in place) | Nil — already reviewed, WATCH steady | No action today |
| 10 | FX ORB reopen | 3 (new surface, same ORB mechanism) | 3 (pre-reg + new data pull) | 3 (Fitschen supports ORB in general) | Twice-killed; low expected value without material scope change | Blocked by reopen criteria |
| 11 | **Deployment queue: 32 validated-active lanes NOT in `topstep_50k_mnq_auto` profile** | 2 (deployment decision, not research) | 1 (routing-only) | 5 (lanes already passed validation gates) | Unquantified as a whole; per-lane EV is in each lane's `validated_setups` row. Covers `CROSS_SGP_MOMENTUM` (3), `COST_LT12` (6), `ORB_G5` (7), `VWAP_MID_ALIGNED` (2), `X_MES_ATR60` (6), etc. | Yes (as individual routing decisions); blocked collectively by profile's correlation / capital budget |

---

## Recommendation

**Park explicit new research for ~3 weeks. Use the window to ship 2 low-cost stubs that close the highest-diversity near-term candidates.**

Rationale:
- Item 3 (Phase D D-2) is the highest-EV item already in-flight — lane contract locked 2026-04-17, D-1 signal-only shadow accumulating real-time forward OOS. Gate eval at ≥2026-05-15 per `MEMORY.md` phase-D daily runbook. **Don't disturb it. Forward OOS is the asset.**
- Items 1, 2, 4 all need infra/literature work before they can even start. Writing the scan today would be premature.
- Item 5 (Horizon T0-T8) is **the only high-diversity research item runnable today** with an existing harness. It's also the natural Path C successor — volume + garch + overnight_range were the three candidate axes in the rel_vol stress-test v2 horizon. But MEMORY says H2 is already VALIDATED and awaiting signal-only shadow; the other candidates (H1 is CONDITIONAL due to thin OOS, H3/H4/H5 CONDITIONAL similar) also need thicker OOS, not more scans.
- Items 6, 7 were explicitly user-deferred in the audit session.
- Items 8, 9, 10 are maintenance / monitoring / closed.

**Conclusion:** running new scans today would violate the `HANDOFF.md` freshness-bias warning and would compete with the Phase D D-1 signal-only shadow that's already earning forward OOS. **Best-EV action today is strategic patience** — but with one explicit exception:

**Exception — Item 5a (H1 MES LONDON_METALS signal-only shadow) is runnable TODAY** without competing with Phase D, because it's a *different cell* (MES LONDON_METALS ≠ MNQ COMEX_SETTLE). It sits at the same posture as H2 pre-Phase-D: CONDITIONAL verdict, waiting for a signal-only shadow to accumulate OOS. Shipping an H1 shadow now means H1 matures in parallel with Phase D D-1 rather than sequentially after it. Cost is low (shadow pattern already established), and it addresses MEMORY line 27's explicit "Next: signal-only shadow H2 + H1" directive whose H1 half was never completed. However, H1 per-cell EV is **not quoted in any live source** — would need a quick re-query against `orb_outcomes` before contract-locking.

### Cell-overlap dependency note (L-1 correction)

Items **3 (Phase D D-2 live)** and the H2 result from MEMORY's "signal-only shadow H2" are on the **same cell** (`MNQ COMEX_SETTLE O5 RR1.0 long garch_forecast_vol_pct≥70`). They are not parallel candidates — Phase D IS the productization path for H2. Don't double-count them.

## Concrete next actions (small, grounded, optional)

Rank-ordered. Each can be picked up individually without blocking the parked queue. Research items (#5) ship in PARALLEL with Phase D D-1, not sequential.

1. **5a — H1 MES LONDON_METALS signal-only shadow pre-reg.** Highest single-action EV on the list. Parallel to Phase D (different cell). Requires: quick `orb_outcomes` re-query to quantify H1 per-cell EV, then a D-1-style contract-lock yaml. ~1-2 hours. Docs-only until locked; then ambient daily-append for 4-12 weeks.
2. **Maintenance:** run `/rebuild-outcomes MES MGC MNQ` to clear the 3 `experimental_strategies` staleness flags. Pure compute; zero research impact. (Time: ~hours each instrument.)
3. **Stub writing:** fill in the two highest-diversity stubs with a minimal pre-reg skeleton each so the queue is ready when infra unblocks:
   - `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md` → add K_budget, kill criteria, expected-N per cell. Gated on `prev_week_*`/`prev_month_*` feature build.
   - `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` → add mechanism prose + outcome_builder interface spec. Gated on schema-extension design.
4. **Audit class-level closures (user-deferred, still available):** action items 3 (Check 45 auto-refresh) and HLZ mechanical gate. Both small code changes, both close existing class-level gaps.
5. **Phase D D-1 daily-append cadence:** per `MEMORY.md` phase-D daily runbook, append daily 2026-04-17 → 2026-05-15. No gate eval before 2026-05-15. Runs ambient; no action needed beyond daily-append script invocation.
6. **Deployment queue review (item 11):** 32 validated-active lanes sit outside `topstep_50k_mnq_auto`. Per-lane decisions are routing-level, not research-level. Worth a separate review pass against capital / correlation budget when an activation slot opens.

## Decision log for next session

If HANDOFF's "fresh EV queue" directive is re-raised:
- **Do NOT run a fresh scan today.** All scan-ready axes are either already validated (H2, going to Phase D D-2), already subsumed (composite rel_vol × garch), or waiting on OOS (H3 / H4 / H5).
- **Do NOT promote A4a/A4b/A4c derivative work.** User explicitly paused the garch-allocator family.
- **DO pick action 1 (H1 signal-only shadow)** if next session wants to ship a parallel, different-cell research stream while D-1 matures. Requires: `orb_outcomes` re-query for H1 per-cell EV → D-1-style contract-lock yaml.
- **DO pick action 3 (stub writing)** if next session wants docs-only progress while D-1 shadow runs.
- **DO pick action 4 (audit class-level closures)** if next session wants concrete engineering progress.
- **DO re-run this queue-reset pass** at ≥2026-05-15 when Phase D D-2 gate eval becomes available — that changes the EV calculus materially.

## Audit / self-correction log

- **2026-04-18 self-audit:** original doc committed at `c79aa395` had four findings corrected in follow-up commit:
  - H-1 (HIGH): removed unsourced "~+0.2R per-cell lift" training-memory claim from horizon candidates row. Replaced with **"EV UNQUANTIFIED — requires re-query"** label.
  - H-2 (HIGH): split original single-row "Horizon T0-T8" into 5a / 5b / 5c rows. 5a (H1 MES LONDON_METALS signal-only shadow) is runnable today as a parallel-to-Phase-D research stream. Was previously mislumped as "waiting for OOS".
  - M-1 (MEDIUM): flagged MEMORY line 27 "next: composite rel_vol × garch" entry as STALE — the composite was closed in `path-c-h2-closure.md` with verbatim "NO SYNERGY / SUBSUMED".
  - M-2 (MEDIUM): added item 11 — deployment queue of 32 validated-but-not-deployed lanes. Routing-level decision, separate from research queue.
  - L-1 (LOW): added cell-overlap dependency note — Phase D D-2 IS the H2 productization path; not a separate candidate.

## Cross-refs

- `HANDOFF.md` lines 7-52 (A4c NULL verdict + queue-reset directive)
- `MEMORY.md` lines 7-19 (Path C H2 book closed), 22-27 (horizon candidates), 29-36 (rel_vol v2), 41-45 (Phase D spec)
- `docs/audit/2026-04-18-grounding-audit-master.md` (audit session just closed)
- `docs/audit/results/2026-04-17-garch-a4c-routing-selectivity-replay.md` (A4c NULL verdict)
- `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md` (Phase D pilot)
- `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md` (HTF Path A stub)
- `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` (Phase C stub)

---

**Verdict:** strategic patience. Do not disturb Phase D D-1 shadow. Write low-cost stubs during the wait if cycles allow. Re-evaluate the queue at 2026-05-15 D-2 gate.
