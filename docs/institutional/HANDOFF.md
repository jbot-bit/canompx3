# HANDOFF — Phase 0 Literature Grounding (2026-04-07)

> **🟢 SUPERSEDED 2026-04-07.** This file was the mid-session handoff written when Phase 0 was partially complete (6 of 9 literature extracts). **Phase 0 is now FULLY COMPLETE.** All 7 literature extracts, the finite-data framework, the v2 criteria (with 5 Codex amendments), the hypothesis registry template, the `docs/audit/hypotheses/` infrastructure, the Codex finite-data audit, the rule file integrations, and the CLAUDE.md pointers are all committed on `main`. See:
>
> - **Current state of Phase 0:** `docs/institutional/README.md` § "v2 status (as of 2026-04-07)"
> - **Phase 2-5 plan:** `docs/plans/2026-04-07-canonical-data-redownload.md` (committed `63c3c28`)
> - **Memory file:** `~/.claude/projects/C--Users-joshd-canompx3/memory/institutional_phase0_grounding.md` (commit chain + cost verification)
> - **Top-level HANDOFF.md:** session-baton entries from 2026-04-07
>
> Commit chain for Phase 0: `0028333` (jbot-bit, literature + rules + criteria v1) → `63c3c28` (jbot-bit, redownload plan) → `48d4e2d` (jbot-bit, View B closure + HANDOFF Codex section) → `390e408` (mine, Codex audit file + CLAUDE.md pointers) → `7bc47a7` (mine, audit/hypotheses README) → `f471b54` (mine, finite_data_framework v2 reconciliation).
>
> Phase 2 cost verified $0.00 (free on Standard plan, 398.1 MB total). Phase 2-5 execution blocked on `e2-canonical-window-fix` worktree merge (scope_lock on `pipeline/check_drift.py`, `pipeline/build_daily_features.py`, `trading_app/outcome_builder.py`).
>
> **Historical mid-session content preserved below for audit history.** Do not act on the "What's PENDING for next session" list — that work is done.

---

**Status:** Partial — 6 of 9 literature extracts written, framework/criteria/template pending.
**Context clear:** Safe to close terminal. Files on disk, git committed.
**Next session:** Start by reading `docs/institutional/README.md` and this HANDOFF.

---

## What triggered this work

April 2026 audit revealed that `strategy_discovery.py` tested ~35,616 MNQ combinations + ~26,000 MGC combinations on limited real micro data (2.2 years clean MNQ, 16 years parent proxy). Per Bailey et al (2013) Minimum Backtest Length theorem, the safe bound is ~45 independent trials on 5 years of data. We over-tested by approximately 600x. The goal of this work is to establish a canonical, literature-grounded framework to fix the discovery methodology without throwing out real edge that may exist in the current 5 deployed MNQ/MGC lanes.

---

## What's DONE in this session

### Directory structure created
```
docs/institutional/
├── README.md                                 ✅ master index
├── HANDOFF.md                                ✅ this file
└── literature/
    ├── bailey_et_al_2013_pseudo_mathematics.md          ✅ MinBTL theorem
    ├── bailey_lopez_de_prado_2014_deflated_sharpe.md    ✅ DSR formula
    ├── lopez_de_prado_bailey_2018_false_strategy.md     ✅ False Strategy Theorem
    ├── harvey_liu_2015_backtesting.md                   ✅ BHY haircut, Exhibit 4
    ├── chordia_et_al_2018_two_million_strategies.md     ✅ t=3.79 threshold
    └── pepelyshev_polunchenko_2015_cusum_sr.md          ✅ Shiryaev-Roberts monitoring
```

### PDFs successfully read and verbatim-extracted
1. `resources/Pseudo-mathematics-and-financial-charlatanism.pdf` — pages 1-15 (full)
2. `resources/deflated-sharpe.pdf` — pages 1-20 (full)
3. `resources/false-strategy-lopez.pdf` — pages 1-7 (full)
4. `resources/backtesting_dukepeople_liu.pdf` — pages 1-17 (full)
5. `resources/Two_Million_Trading_Strategies_FDR.pdf` — pages 1-20
6. `resources/real_time_strategy_monitoring_cusum.pdf` — pages 1-14 (full)
7. `resources/prop-firm-official-rules.md` — read (text file)
8. `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` — pages 1-28 (Chapter 1 in full, plus bibliography confirmed pp 29-45). **Note: this is the Cambridge Elements short-form version, only 45 pages total, containing Chapter 1 "Introduction" as substantive content. Not a truncation.**

### Core findings already captured in literature files
- **Bailey et al 2013 Theorem 1 (MinBTL):** `MinBTL < 2·Ln[N] / E[max_N]²`. With 5 years of data, max ~45 independent trials. Our discovery tested 35,000+. Over-tested by ~600x.
- **Bailey-LdP 2014 DSR (Eq. 2):** formula that corrects Sharpe ratio for both selection bias AND non-normality. Threshold DSR > 0.95. Our project does not currently compute this.
- **LdP-Bailey 2018 False Strategy Theorem:** under zero true edge, E[max SR] ≈ (1-γ)Z⁻¹[1-1/K] + γZ⁻¹[1-1/(Ke)]. At K=35,000, expected max SR ≈ 3.87. Our best deployed lane Sharpe: 1.23. We are below the noise floor.
- **Harvey-Liu 2015 Exhibit 4:** at 240 monthly obs, 10% vol, 300 tests: BHY hurdle is 7.4% annual. Our current implied hurdle is higher because data is shorter.
- **Chordia et al 2018 t-statistic threshold:** finance MHT threshold is t ≥ 3.79 (for FDP approach with correlation). Only MNQ COMEX_SETTLE OVNRNG_100 clears this among our 5 deployed lanes.
- **Pepelyshev-Polunchenko 2015:** Shiryaev-Roberts procedure recursion `R_n = (1 + R_{n-1}) · Λ_n` is multi-cyclic optimal for live drift detection. Superior to CUSUM for repeated use.

---

## What's PENDING for next session

### Files still to write
1. `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` — **Chapter 1 content already read (pp 6-28), need to write verbatim extract**. Key content: Lesson 1 (theory, not backtests), Lesson 2 (ML helps discover theories), two types of overfitting, CPCV reference, Monte Carlo on synthetic data, "backtests are not a research tool" quote.
2. `docs/institutional/finite_data_framework.md` — synthesized methodology. Cross-references literature files. Covers: MinBTL calculation for our N and T, effective N via correlation adjustment, DSR application, CPCV for short data, provisional sizing, CUSUM monitoring.
3. `docs/institutional/pre_registered_criteria.md` — LOCKED thresholds. BH FDR q, DSR threshold, WFE threshold, sample size minimums, OOS requirements, era stability, pre-registered trial count budget. No post-hoc relaxation allowed.
4. `docs/institutional/hypothesis_registry_template.md` — template for pre-registered hypothesis entries.

### Integration points still to wire
5. Update `CLAUDE.md` Document Authority section to reference `docs/institutional/` for research methodology.
6. Update `.claude/rules/research-truth-protocol.md` to require reading `docs/institutional/pre_registered_criteria.md` before any discovery run.
7. Update `.claude/rules/institutional-rigor.md` to reference `docs/institutional/literature/` as the canonical citation source.
8. Optional: create `.claude/rules/pre-registered-discovery.md` — new rule file that auto-triggers on `strategy_discovery.py` edits to warn about trial count.
9. Optional: add drift check in `pipeline/check_drift.py` that verifies `validated_setups` are consistent with locked criteria.

### Bigger Phase 4+ work (not started)
10. Refactor `strategy_discovery.py` to accept a pre-registered hypothesis list (YAML or JSON) instead of brute-force enumeration.
11. Compute DSR for all 5 currently-deployed lanes using Bailey-LdP Eq. 2 with skewness/kurtosis from raw trades.
12. Compute effective N̂ via correlation matrix of trial Sharpe ratios.
13. Re-run discovery with pre-registered ~100-300 hypothesis budget after Phase 0-3 complete.
14. Implement Shiryaev-Roberts drift monitor in `trading_app/live/` for deployed lanes.

---

## Key facts to re-verify in next session (don't trust stale memory)

- MGC discovery task `be02hwule` completed earlier in session — results in `gold_snap.db.experimental_strategies` table (snapshot file, not live gold.db)
- `gold_snap.db` is a 6.3GB copy of gold.db with `promoted_from` nulled and FK-blocked tables cleared to permit holdout discovery runs
- Memory index already updated with `MNQ_CUT = DATE '2024-02-05'` for era-split queries — do not re-discover this
- `paper_trades` table in gold.db has 161 rows but NONE match the 5 currently-deployed strategy_ids — there is no real forward paper trade data for the deployed set
- Current deployed profile: `topstep_50k_mnq_auto` with 5 lanes at 2 copies, all price-filter based (no volume filters)
- 4 of 5 deployed MNQ lanes had ZERO or NEGATIVE edge in pre-2020 data per era-split query earlier in session

---

## Reading rule that must be followed (CLAUDE.md line 79)

**Extract-before-dismiss rule:** Before characterizing a PDF's content as "bibliography only" or "nothing relevant", extract the table of contents AND at least 3 sample pages from the middle. A single-keyword grep can miss whole chapters when terminology differs. This session verified compliance: each literature file above is backed by pages actually read via the Read tool, with page numbers cited.

---

## Pointer to related docs

- `CLAUDE.md` — project guidance, updated with PDF reading rule on 2026-04-07
- `.claude/rules/institutional-rigor.md` — updated with extract-before-dismiss corollary
- `docs/specs/research_modes_and_lineage.md` § 9.2 — prior self-caught error re: LdP ML characterization, commit `aec7730`
- `resources/` — 16 PDFs, including 8 that were read this session and 8 that were NOT yet read (Carver Systematic Trading, Algorithmic Trading Chan, Quantitative Trading Chan 2008, Evidence Based TA Aronson, BH 1995, Man Group Overfitting 2015, Building Reliable Trading Systems, Prado Optimization of Trading Strategies)

---

## How to resume (checklist for next session)

1. Read `CLAUDE.md` top-to-bottom (new PDF rule at line 79)
2. Read this HANDOFF.md
3. Read `docs/institutional/README.md`
4. `TaskList` to see remaining tasks (#9-13 pending as of handoff)
5. Read already-written literature files to get context on what's already extracted
6. Continue with pending work — start with LdP ML extract (task #9), then framework/criteria/template
7. After files are written: wire integration points (CLAUDE.md Document Authority, rules files, drift check)
8. Commit in logical chunks
9. Only then start Phase 4 (pre-registered rediscovery)
