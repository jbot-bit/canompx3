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
| 5 | Horizon T0-T8 candidates | 3-4 (overnight_range, DOW, pivot-take are mechanism-orthogonal to garch) | 1 (reuse `t0_t8_audit_horizon_non_volume.py` harness) | 4 (Fitschen + overnight-range lit partial) | H2 was VALIDATED (8P/0F) per MEMORY line 22; H1 CONDITIONAL (7P/1F) line 23. Per-cell lift ~+0.2R on thin OOS | **Yes** |
| 6 | HLZ mechanical gate | 2 (audit housekeeping, not research) | 2 (drift check + tests) | 5 (grounded via IMP-1 + Chordia indirect) | Preventative only; no revenue lift | Yes (but user deferred) |
| 7 | Check 45 auto-refresh | 2 (audit housekeeping) | 2 (pipeline hook + tests) | 5 (Tier 1-irrelevant — pure engineering) | Preventative only; no revenue lift | Yes (but user deferred) |
| 8 | `/rebuild-outcomes` × 3 | 1 (pure maintenance) | 1 (compute time only) | 5 (N/A) | Nil; fixes stale read | Yes |
| 9 | SR ALARM re-review | 1 (monitoring) | 1 (review only) | 5 (institutional canon in place) | Nil — already reviewed, WATCH steady | No action today |
| 10 | FX ORB reopen | 3 (new surface, same ORB mechanism) | 3 (pre-reg + new data pull) | 3 (Fitschen supports ORB in general) | Twice-killed; low expected value without material scope change | Blocked by reopen criteria |

---

## Recommendation

**Park explicit new research for ~3 weeks. Use the window to ship 2 low-cost stubs that close the highest-diversity near-term candidates.**

Rationale:
- Item 3 (Phase D D-2) is the highest-EV item already in-flight — lane contract locked 2026-04-17, D-1 signal-only shadow accumulating real-time forward OOS. Gate eval at ≥2026-05-15 per `MEMORY.md` phase-D daily runbook. **Don't disturb it. Forward OOS is the asset.**
- Items 1, 2, 4 all need infra/literature work before they can even start. Writing the scan today would be premature.
- Item 5 (Horizon T0-T8) is **the only high-diversity research item runnable today** with an existing harness. It's also the natural Path C successor — volume + garch + overnight_range were the three candidate axes in the rel_vol stress-test v2 horizon. But MEMORY says H2 is already VALIDATED and awaiting signal-only shadow; the other candidates (H1 is CONDITIONAL due to thin OOS, H3/H4/H5 CONDITIONAL similar) also need thicker OOS, not more scans.
- Items 6, 7 were explicitly user-deferred in the audit session.
- Items 8, 9, 10 are maintenance / monitoring / closed.

**Conclusion:** running new scans today would violate the `HANDOFF.md` freshness-bias warning and would compete with the Phase D D-1 signal-only shadow that's already earning forward OOS. **Best-EV action today is strategic patience.**

## Concrete next actions (small, grounded, optional)

Rank-ordered, all docs-only unless flagged. Each can be picked up individually without blocking the parked queue.

1. **Maintenance:** run `/rebuild-outcomes MES MGC MNQ` to clear the 3 `experimental_strategies` staleness flags. Pure compute; zero research impact. (Time: ~hours each instrument.)
2. **Stub writing:** fill in the two highest-diversity stubs with a minimal pre-reg skeleton each so the queue is ready when infra unblocks:
   - `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md` → add K_budget, kill criteria, expected-N per cell. Gated on `prev_week_*`/`prev_month_*` feature build.
   - `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` → add mechanism prose + outcome_builder interface spec. Gated on schema-extension design.
3. **Audit class-level closures (user-deferred, still available):** action items 3 (Check 45 auto-refresh) and HLZ mechanical gate. Both small code changes, both close existing class-level gaps.
4. **Phase D D-1 daily-append cadence:** per `MEMORY.md` phase-D daily runbook, append daily 2026-04-17 → 2026-05-15. No gate eval before 2026-05-15. Runs ambient; no action needed beyond daily-append script invocation.

## Decision log for next session

If HANDOFF's "fresh EV queue" directive is re-raised:
- **Do NOT run a fresh scan today.** All scan-ready axes are either already validated (H2) or waiting on OOS (H1 / H3 / H4 / H5 / SGP).
- **Do NOT promote A4a/A4b/A4c derivative work.** User explicitly paused the garch-allocator family.
- **DO pick item 2 (stub writing)** if next session wants docs-only progress while D-1 shadow runs.
- **DO pick item 3 (audit class-level closures)** if next session wants concrete engineering progress.
- **DO re-run this queue-reset pass** at ≥2026-05-15 when Phase D D-2 gate eval becomes available — that changes the EV calculus materially.

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
