# Institutional Rigor — Working-Style Hard Rule

**Non-negotiable.** The user has been explicit: we do the proper long-term institutional-grounded fix. We do not band-aid. We do not skip. We review our own work before claiming done.

This rule supersedes "just ship it" when they conflict.

## The Non-Skip Rules

### 1. Self-review before claim-of-done — MANDATORY

Before marking any stage complete, run the code-review skill (or a structured self-review) on the work just done. Produce line citations, not narrative. Execute the code to verify claims, do not rely on reading.

The first review of eligibility Phase 0+1 caught a HALF_SIZE bug that would have silently blocked real trades. Without that review, the bug would have shipped. Review is load-bearing.

### 2. After any fix, review the fix

Fixes introduce new bugs. The eligibility hardening commit closed seven findings and introduced four new ones. Expect this. Plan for it. Do not declare "done" after a fix without a second review pass.

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
- **`docs/institutional/literature/`** is the CANONICAL CITATION SOURCE for research methodology claims — it contains verbatim extracts from the resources PDFs with page numbers and metadata. **Cite from there, not from training memory.** Established 2026-04-07 (Phase 0 literature grounding). Covered papers: Bailey et al 2013 (MinBTL), Bailey-LdP 2014 (DSR), LdP-Bailey 2018 (False Strategy Theorem), Harvey-Liu 2015 (BHY haircut), Chordia et al 2018 (t ≥ 3.79), Pepelyshev-Polunchenko 2015 (Shiryaev-Roberts monitoring), LdP 2020 ML for Asset Managers (theory-first, CPCV), Fitschen 2013 Ch 3 (grounds intraday trend-follow for commodities + stock indices — CORE ORB premise), Carver 2015 Ch 9-10 (added 2026-04-15: grounds volatility targeting + Kelly-linked position sizing + forecast combination for Stage 2+ sophisticated signal deployment).
- **`docs/institutional/mechanism_priors.md`** (added 2026-04-15) is the LIVE TRADING-LOGIC DOCUMENT — captures what we think drives ORB edge, maps each signal to implementation roles (R1 FILTER → R8 PORTFOLIO allocator), and stages deployment (binary → sizing → geometry → portfolio). Read before proposing any new level-based or HTF filter so you don't pigeonhole the test. Priors listed there are testable hypotheses, not validated facts — every claim still requires pre_registered_criteria.md gate pass before deployment.
- **`docs/institutional/pre_registered_criteria.md`** is the LOCKED threshold file — 12 criteria for strategy validation derived from the literature extracts. No post-hoc relaxation allowed. Read before any discovery or deployment decision.
- Design decisions resting on literature claims must cite the specific extract file in `docs/institutional/literature/`. If the extract file does not yet exist for a paper you need to cite, write it first (extract the relevant passage from the PDF), then reference it. Training-memory citations must be labeled "from training memory — not verified against local PDF."
- **Extract-before-dismiss rule:** before characterizing a PDF's content (e.g., "bibliography only", "front matter only", "no relevant content"), extract the table of contents AND at least 3 sample pages from the middle. A single-keyword grep can miss whole chapters when the terminology is different — e.g., on 2026-04-07 a self-review caught a MEDIUM factual error in `docs/specs/research_modes_and_lineage.md` § 9.2 where `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` was incorrectly characterized as "bibliography only" because the only `walk.forward` grep hit was in a bibliography entry, when in fact pp 6-28 of the local PDF are Chapter 1 "Introduction" containing substantive backtest-overfitting content including an explicit CPCV reference. The fix is at commit `aec7730`.
- **Corollary:** keyword matches against PDFs are a starting point, not a conclusion. Always read the surrounding paragraphs before reporting a finding. "`walk`" often means "random walk" (Monte Carlo), "`half`" often means "half-life" (mean reversion) — not walk-forward testing or the 50% Sharpe discount.

### 8. Verify before claiming

- "Done" means: tests pass (show output) + dead code swept (`grep -r`) + drift check passes + self-review passed. All four required.
- "It should work" is not acceptable. Run it.
- "The test passes" is not acceptable. Confirm the test exercises the new code path.

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
