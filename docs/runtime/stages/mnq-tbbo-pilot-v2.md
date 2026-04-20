---
task: MNQ TBBO slippage pilot fix — Phase 2 follow-through from MGC adversarial re-examination (HANDOFF 2026-04-20)
mode: IMPLEMENTATION
scope_lock:
  - research/research_mnq_e2_slippage_pilot.py
  - pipeline/cost_model.py
  - docs/runtime/debt-ledger.md
  - docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md
  - HANDOFF.md
  - tests/test_research/test_reprice_e2_entry_regression.py
  - tests/test_research/test_mnq_pilot_caller.py
  - docs/runtime/stages/mnq-tbbo-pilot-v2.md
blast_radius: "pipeline/cost_model.py comment-only; 139 importers read COST_SPECS numeric values (unchanged). research/research_mnq_e2_slippage_pilot.py has 0 importers; 2 print-string references (cost_model.py:146, scripts/tools/slippage_scenario.py:159) remain accurate. reprice_e2_entry (canonical) NOT touched; 10-test coverage in test_databento_microstructure.py protects against drift. Full audit via blast-radius agent 2026-04-20."
acceptance:
  - Stage 1 canonical regression tests green (3/3)
  - Stage 2 caller tests green (9/9)
  - Pilot runs --reprice-cache clean (114 valid / 5 legitimate error rows)
  - cost_model.py TODO comment updated; no numeric COST_SPECS changes
  - debt-ledger mnq-tbbo-pilot-script-broken entry closed
  - Result doc 2026-04-20-mnq-e2-slippage-pilot-v1.md complete
  - python -m pytest tests/ -x -q all green
  - python -m pipeline.check_drift all green
agent: claude
---

# Stage — MNQ TBBO Slippage Pilot Fix (v2)

**Status:** IMPLEMENTATION
**Started:** 2026-04-20
**Owner:** claude-opus-4-7
**Plan:** `C:\Users\joshd\.claude\plans\go-full-autonomous-next-jaunty-eclipse.md` (v2 approved)
**Parent audit:** `docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md`
**Pre-reg N/A:** This is a pilot MEASUREMENT (cost-realism audit), not a discovery run; 0 MinBTL trials.

## Purpose

Fix broken `research/research_mnq_e2_slippage_pilot.py` so it produces a credible MNQ slippage measurement from the existing 119-file tbbo cache (no Databento spend). Update `pipeline/cost_model.py` MNQ TODO comment with measured median/mean/MAD/p95/max. Close `mnq-tbbo-pilot-script-broken` in debt-ledger.

## Scope lock

Files this stage MAY touch:
- `research/research_mnq_e2_slippage_pilot.py` (rewrite `reprice_entries` + add `--reprice-cache` mode)
- `pipeline/cost_model.py` (comment-only change to lines 144-164 MNQ TODO block; NO numeric COST_SPECS changes)
- `docs/runtime/debt-ledger.md` (close `mnq-tbbo-pilot-script-broken`; refresh `cost-realism-slippage-pilot`)
- `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md` (new result doc)
- `HANDOFF.md` (top-entry update)
- `tests/test_research/test_reprice_e2_entry_regression.py` (new)
- `tests/test_research/test_mnq_pilot_caller.py` (new)
- `docs/runtime/stages/mnq-tbbo-pilot-v2.md` (this file, delete on completion)

Files this stage MUST NOT touch (canonical):
- `research/databento_microstructure.py`
- `research/research_mgc_e2_microstructure_pilot.py`
- `pipeline/cost_model.py` numeric COST_SPECS values (comment only)
- `trading_app/entry_rules.py`, `trading_app/outcome_builder.py`
- Any `validated_setups`, `edge_families`, `live_config` writes

## Blast-radius (verified via blast-radius agent 2026-04-20)

- `pipeline/cost_model.py` comment: 139 importers; none read the TODO comment; zero runtime impact
- `research/research_mnq_e2_slippage_pilot.py`: 0 importers; 2 file-path references in print strings only (cost_model.py:146, scripts/tools/slippage_scenario.py:159); zero callers break on rewrite
- `reprice_e2_entry` (canonical, NOT touched): 10-test coverage in `tests/test_research/test_databento_microstructure.py`; protects against canonical drift

## MGC baseline provenance verification (2026-04-20)

Source: `research/output/mgc_e2_slippage_analysis.json` + `research/output/mgc_e2_repriced_entries.csv`.

MGC.FUT n=40: mean=6.75, **median=0.0, p75=1.0, p95=2.05**, std=41.57, max=263.0, pct_above_2_ticks=5.0%.

**Critical: the mean=6.75 is dominated by ONE day (2018-01-18 long, 263 ticks = gap-open event; all other days ≤ 3 ticks).** Trim that one day → mean ≈ 0.18 ticks. The "MGC 3.4× modeled" claim in `cost_model.py:145` and parent audit §4 is outlier-driven. Honest central tendency is `median=0, p95=2.05`. MNQ pilot result must use median + trimmed-mean + per-day outlier investigation to avoid the same framing bias.

Clean MGC days for Stage 1 regression fixture:
- **2017-04-26 long MGC**: orb_level=1270.8, trigger=1270.8, fill=1270.8, slippage=0.0 ticks, 72 tbbo records (non-event, chosen for Stage 1 regression).
- **2018-01-18 long MGC**: orb_level=1328.1, trigger=1354.4, slippage=263.0 ticks (regression guard — protects against silent canonical changes that would hide this outlier).

## Done criteria (per stage-gate protocol)

- [ ] 2 new test files committed; both GREEN
- [ ] `research_mnq_e2_slippage_pilot.py` runs `--reprice-cache` end-to-end with output CSV produced
- [ ] `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md` complete with per-session breakdown + deployed-lane-weighted aggregate + outlier investigation + all methodology caveats
- [ ] `pipeline/cost_model.py:144-164` TODO comment updated with measured stats
- [ ] `docs/runtime/debt-ledger.md` entries updated
- [ ] `python -m pytest tests/ -x -q` all GREEN
- [ ] `python -m pipeline.check_drift` all GREEN
- [ ] `HANDOFF.md` top entry updated
- [ ] Code-review skill grade ≥ B on full diff
- [ ] This stage file deleted

## Red-flag stops (from plan v2 §Risks)

- Stage 1 canonical regression tests FAIL → STOP; surface as canonical bug affecting MGC parent audit
- >10% of MNQ days report \|slippage\| > 50 ticks without event-day explanation → STOP; per-day investigation
- MNQ median ≥ 5 ticks with no mechanism → STOP; surface
- Phase D MNQ COMEX_SETTLE baseline materially changes → STOP; flag for user decision
- Any of 6 deployed MNQ lanes flips negative EV under measured median-friction → STOP; flag
