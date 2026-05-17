---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "scripts/**"
  - "research/**"
  - "docs/institutional/**"
  - "docs/specs/**"
---

# Institutional Rigor — Working-Style Hard Rule

**Non-negotiable.** The user has been explicit: we do the proper long-term institutional-grounded fix. We do not band-aid. We do not skip. We review our own work before claiming done.

This rule supersedes "just ship it" when they conflict.

## The Non-Skip Rules

### 1. Self-review before claim-of-done — MANDATORY

Before marking any stage complete, run the code-review skill (or a structured self-review) on the work just done. Produce line citations, not narrative. Execute the code to verify claims, do not rely on reading.

The first review of eligibility Phase 0+1 caught a HALF_SIZE bug that would have silently blocked real trades. Without that review, the bug would have shipped. Review is load-bearing.

### 2. After any fix, review the fix

Fixes introduce new bugs. The eligibility hardening commit closed seven findings and introduced four new ones. Expect this. Plan for it. Do not declare "done" after a fix without a second review pass.

**For CRIT/HIGH fixes in truth-layer paths, the review pass is formalized as the adversarial-audit gate — see `.claude/rules/adversarial-audit-gate.md`.** The gate requires an independent-context `evidence-auditor` pass before the next phase dispatches. The C1 kill-switch race (iter 174 F4 fix, caught by audit in 2026-04-25) is the proof case for why single-agent self-review is insufficient on exposure-creating paths.

### 3. Refactor when you see a pattern of bugs

If review cycles keep finding new divergences, the architecture is wrong — stop patching. Name the root cause, propose the structural change, present options with blast radius and trade-offs. Do not offer "just ship it" as a realistic option — frame it honestly as debt.

### 4. Delegate to canonical sources — never re-encode

If `trading_app/config.py` has filter logic, new code must CALL the canonical implementation, not re-implement it. Parallel models drift. Every divergence is a silent failure waiting to happen.

Examples of this rule in action:
- Eligibility builder calls `StrategyFilter.matches_row()` directly, not a hand-coded copy.
- Session times come from `pipeline.dst.SESSION_CATALOG` resolvers, never hardcoded.
- Cost specs come from `pipeline.cost_model.COST_SPECS`, never inlined.
- DB path from `pipeline.paths.GOLD_DB_PATH`, never `/tmp/gold.db`.

### 5. No dead code — remove or populate

- Dead fields (initialized, never written) are lies about the data model.
- Dead enum values are silent mislabels waiting to happen.
- Dead parameters passed for "future use" are drift bait.
- `_ = unused_var` to silence linters is forbidden — fix the underlying structure.

### 6. No silent failures

- Every `except Exception` must record the exception (build_errors / logger / propagate).
- Every "if X is None return default" must be explicit.
- NaN, NaT, pd.NA must be treated as missing — `is None` alone is not sufficient.
- Fail-open is acceptable only when the canonical source fails open AND the behavior is documented on the canonical source.

### 7. Ground in local resources before training memory

- `resources/` has 15+ institutional PDFs (raw sources).
- **`docs/institutional/literature/`** is the CANONICAL CITATION SOURCE for research methodology claims — it contains verbatim extracts from the resources PDFs with page numbers and metadata. **Cite from there, not from training memory.** Established 2026-04-07 (Phase 0 literature grounding); refreshed 2026-05-12 (27 extracts). **When citing any of these topics, fetch the extract via the `research-catalog` MCP (`mcp__research-catalog__get_literature_excerpt`) — do not paraphrase from memory.** Inventory:

  | Topic | Extract file | What it grounds |
  |---|---|---|
  | Multi-testing (FDR) | `benjamini_hochberg_1995_fdr.md` | Primary BH-1995 source for BHY haircut math |
  | Multi-testing (MinBTL) | `bailey_et_al_2013_pseudo_mathematics.md` | MinBTL bound — caps brute-force at 300 trials |
  | Multi-testing (DSR) | `bailey_lopez_de_prado_2014_deflated_sharpe.md` | Deflated Sharpe Ratio core formula |
  | Multi-testing (DSR-selection) | `bailey_lopezdeprado_2014_dsr_sample_selection.md` | DSR sample-selection correction |
  | Multi-testing (FST) | `lopez_de_prado_bailey_2018_false_strategy.md` | False Strategy Theorem |
  | Multi-testing (haircut) | `harvey_liu_2015_backtesting.md` | BHY Sharpe haircut method |
  | Multi-testing (cross-section) | `harvey_liu_zhu_2015_cross_section.md` | t ≥ 3.0 threshold for cross-sectional anomalies |
  | Multi-testing (empirical) | `chordia_et_al_2018_two_million_strategies.md` | t ≥ 3.79 empirical bound (2M strategies) |
  | Data-snooping | `aronson_2007_ebta_data_snooping.md` | EBTA / data-snooping bias mechanics |
  | Backtest method (ML) | `lopez_de_prado_2020_ml_for_asset_managers.md` | Theory-first ML, CPCV for asset managers |
  | Backtest method (AFML) | `lopez_de_prado_2018_afml_ch_3_7_8.md` | AFML Ch 3 labeling, Ch 7 CV, Ch 8 feature importance, Ch 12 CPCV |
  | Backtest method (lookahead) | `chan_2013_ch1_backtesting_lookahead.md` | Look-ahead bias prevention |
  | Backtest method (TOC) | `chan_2013_toc_determination.md` | Backtest TOC determination |
  | Mechanism (microstructure) | `harris_2002_trading_exchanges_microstructure.md` | Stop-cascade (E2 entry); adverse selection; 17.8× commodity-pool Sharpe deflation |
  | Mechanism (auction theory) | `topstep_2026_auction_market_theory_intro.md` | AMT introduction (Topstep) |
  | Mechanism (auction formal) | `tolusic_2026_amt_inventory_dynamics.md` | Formal mathematical AMT + inventory dynamics |
  | Mechanism (value area) | `howard_2026_value_area_breakouts_es.md` | ES value-area breakout empirics |
  | Mechanism (ORB premise) | `fitschen_2013_path_of_least_resistance.md` | Fitschen Ch 3 intraday trend-follow — CORE ORB premise |
  | Mechanism (ORB-NQ) | `yordanov_2026_nq_orb_value_area_breakouts.md` | NQ ORB value-area breakout study |
  | Regime / sessions | `chan_2009_ch1_intraday_session_handling.md` | Intraday session handling |
  | Regime switching | `chan_2008_ch7_regime_switching.md` | Regime-switching models |
  | Regime (currencies/futures) | `chan_2013_ch5_currencies_futures_meanreversion.md` | FX/futures mean-reversion |
  | Regime (intraday momentum) | `chan_2013_ch7_intraday_momentum.md` | Intraday momentum empirics |
  | Sizing (vol target) | `carver_2015_volatility_targeting_position_sizing.md` | Volatility targeting / Kelly-linked sizing |
  | Sizing (portfolio) | `carver_2015_ch11_portfolios.md` | Forecast combination / portfolio construction |
  | Sizing (speed/size) | `carver_2015_ch12_speed_and_size.md` | Trading speed × size trade-offs |
  | Monitoring | `pepelyshev_polunchenko_2015_cusum_sr.md` | Shiryaev-Roberts SR-monitor mathematics |
- **`docs/institutional/mechanism_priors.md`** (added 2026-04-15) is the LIVE TRADING-LOGIC DOCUMENT — captures what we think drives ORB edge, maps each signal to implementation roles (R1 FILTER → R8 PORTFOLIO allocator), and stages deployment (binary → sizing → geometry → portfolio). Read before proposing any new level-based or HTF filter so you don't pigeonhole the test. Priors listed there are testable hypotheses, not validated facts — every claim still requires pre_registered_criteria.md gate pass before deployment.
- **`docs/institutional/pre_registered_criteria.md`** is the LOCKED threshold file — 12 criteria for strategy validation derived from the literature extracts. No post-hoc relaxation allowed. Read before any discovery or deployment decision.
- Design decisions resting on literature claims must cite the specific extract file in `docs/institutional/literature/`. If the extract file does not yet exist for a paper you need to cite, write it first (extract the relevant passage from the PDF), then reference it. Training-memory citations must be labeled "from training memory — not verified against local PDF."
- **Extract-before-dismiss rule:** before characterizing a PDF's content (e.g., "bibliography only", "front matter only", "no relevant content"), extract the table of contents AND at least 3 sample pages from the middle. A single-keyword grep can miss whole chapters when the terminology is different — e.g., on 2026-04-07 a self-review caught a MEDIUM factual error in `docs/specs/research_modes_and_lineage.md` § 9.2 where `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` was incorrectly characterized as "bibliography only" because the only `walk.forward` grep hit was in a bibliography entry, when in fact pp 6-28 of the local PDF are Chapter 1 "Introduction" containing substantive backtest-overfitting content including an explicit CPCV reference. The fix is at commit `aec7730`.
- **Corollary:** keyword matches against PDFs are a starting point, not a conclusion. Always read the surrounding paragraphs before reporting a finding. "`walk`" often means "random walk" (Monte Carlo), "`half`" often means "half-life" (mean reversion) — not walk-forward testing or the 50% Sharpe discount.

### 8. Verify before claiming

- "Done" means: tests pass (show output) + dead code swept (`grep -r`) + drift check passes + self-review passed. All four required.
- "It should work" is not acceptable. Run it.
- "The test passes" is not acceptable. Confirm the test exercises the new code path.

### 9. Discovery-Loop Tells — produce an artifact before editing

"Reading remaining files before patching" / "let me check more" / "isolating weak spots" are discovery-loop tells. Stop. Produce ONE of three artifacts before editing `pipeline/` or `trading_app/`:

  (a) `REPRO:` failing command + actual vs expected
  (b) `python scripts/tools/context_resolver.py --task "<x>" --format markdown` output
  (c) `TRIVIAL:` declaration with file list and diff <100 lines

Enforced by `.claude/hooks/pre-edit-discovery-marker.py` (PreToolUse, fail-open). Manual escape: `.claude/scratch/discovery-marker.json` with `{"valid_until": "<ISO timestamp>"}`. See `docs/plans/discovery-loop-hardening.md` for the full forcing-function design.

### 10. Canonical Sources — Authority Table (merged from integrity-guardian.md § 2, 2026-05-17)

Import from the single source of truth. Never inline lists or magic numbers.

| Data | Source |
|------|--------|
| Active instruments | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| All instrument configs | `pipeline.asset_configs.ASSET_CONFIGS` |
| Session catalog | `pipeline.dst.SESSION_CATALOG` |
| ORB window timing (UTC) | `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` |
| Entry models / filters | `trading_app.config` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| DB path | `pipeline.paths.GOLD_DB_PATH` |
| Holdout policy (Mode A) | `trading_app.holdout_policy` — `HOLDOUT_SACRED_FROM`, `HOLDOUT_GRANDFATHER_CUTOFF`, `enforce_holdout_date()` |

### 11. Never Trust Metadata — Always Verify (merged from integrity-guardian.md § 7, 2026-05-17)

Metadata, comments, docstrings, bundle fields, and config labels are NOT evidence.
- Never trust a model bundle's `rr_target_lock` field without querying what data it trained on
- Never trust a check's `PASSED` label without confirming the check actually tests what it claims
- Never trust a function docstring's description of behavior without reading the code
- Never trust row counts from memory — execute the query and read the output
- When inspecting ML models, trace FROM the database query THROUGH the code TO the model output
- When verifying drift checks, inject a known violation and confirm it's caught
- **Reading code is not verifying code. Verifying requires execution + output inspection.**

### 12. Seven-Sins Bias Defense (reminder from quant-agent-identity.md, 2026-05-17)

Behavioral reminder on every change. Full bias-class index lives in `quant-agent-identity.md` (now auto-loads only on research / strategy_* / audit edits). Quick reference:

| Sin | What to watch for |
|-----|-------------------|
| Look-ahead bias | Future data as predictor; see `backtesting-methodology.md` § 1.1 + § 6.3 banned-column list |
| Data snooping | Multiple-testing without BH-FDR at the appropriate K framing |
| Overfitting | High Sharpe, low N (< 30) or one good year |
| Survivorship bias | Dropping dead instruments (MCL/SIL/M6E/MBT/M2K) or E0 from base rates |
| Storytelling bias | Narrative around noise; p > 0.05 is observation, not finding |
| Outlier distortion | One extreme day driving aggregate; year-by-year breakdown required |
| Transaction cost illusion | Always use `pipeline.cost_model.COST_SPECS` |

---

## The Treadmill Signal

If you find yourself saying "oh and also fix X" more than twice in a session, stop. The architecture is wrong. Propose a refactor. Do not keep patching.

## Enforcement

- Post-work verification: before any `git commit` on non-trivial work, mentally run through the 8 rules.
- If any fail, fix them before committing. If the fix reveals a pattern, propose a refactor instead of patching.
- If the user's task is ambiguous ("continue", "go", "next"), default to the more thorough interpretation, not the faster one.

## What this rule forbids

- "I'll just fix this one thing real quick"
- "Close enough" / "TODO later" / "we can revisit"
- "It probably works"
- Re-encoding logic that already exists in a canonical source
- Dead fields, dead enums, dead parameters left in code
- Silencing linter warnings with `_ = x` instead of fixing the structure
- Offering "just ship it" as a realistic option in design proposals
- Moving to a new task with known bugs in the current one
