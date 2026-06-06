# Max-Return Own-Capital Book — Design Record (2026-06-06)

**Status:** report built. Read-only analysis, outside the capital path.
**Script:** `scripts/reports/report_max_return_own_capital.py`
**Branch:** `feature/own-capital-return-report` (NOT main).

---

## Question

What is the **max-total-return** trading book for a **$30,000 personal-capital
account** with a **$6,000 hard-stop drawdown**, built from the best strategies in
`validated_setups` — ranked by *annualized dollar expectancy*, not RR headline,
and sized **risk-first** (no prop-firm contract caps)?

This is explicitly NOT the deployed Topstep prop book. Per
`.claude/rules/self-funded-sizing-doctrine.md`, personal-capital sizing is bounded
by drawdown tolerance + volatility-targeting + broker margin + liquidity — never
by a prop firm's 5/10/15 lot ladder or MLL logic.

## Hard guardrails (operator-set)

- Account = **$30,000**.
- Hard shutdown DD = **$6,000**.
- Design-target modeled DD = **≤ $4,000** so a 1.5× stress still fits inside $6k.
- No prop-firm caps, no MLL, no scaling-plan lot ceilings.
- Rank by **annualized dollar expectancy** = `ExpR × trades/year × risk_dollars`,
  NOT RR headline.
- CORE first (sample ≥ 100). REGIME (30–99) only as flagged overlay, never
  standalone.
- Deduplicate by edge family.
- Penalize correlation / concentration.
- Slippage stress at 1×, 1.5×, 2×.
- Do NOT use max-contracts-to-DD-limit as the primary sizer (too aggressive for a
  first real-capital deployment).

## Tier taxonomy (canonical)

`trading_app/config.py` defines exactly two sample tiers:
- `CORE_MIN_SAMPLES = 100` → **CORE**
- `REGIME_MIN_SAMPLES = 30` → **REGIME** (30–99)

There is **no canonical PRELIM tier**. The operator's "CORE + PRELIM book" maps to
**CORE + `REGIME_NEAR_CORE`** — the upper REGIME band — and every such lane is
flagged provisional. We do not invent a PRELIM label that the validator doesn't
recognize.

## Truth discipline (load-bearing)

`validated_setups` is a DERIVED layer (`research-truth-protocol.md`). Its
`trades_per_year`, `median_risk_dollars`, `expectancy_r` are **provisional
snapshots** — used ONLY to seed and rank candidates.

The **binding numbers** — drawdown, realized frequency, realized risk dollars, and
the sizing decision — are **recomputed from canonical replay** via
`trading_app.account_survival._load_lane_trade_paths` (live from `orb_outcomes`,
with `stop_multiplier` applied) and `_max_observed_rolling_drawdown`. The report
prints both and flags any snapshot field as `[snapshot]`.

## Sizing model — Carver-grounded (official best practice)

**Literature grounding (page-cited, on-disk extracts — NOT training memory):**
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
  (Carver 2015, *Systematic Trading*, Ch 9–10, pp 135–163)
- `docs/institutional/literature/carver_2015_ch11_portfolios.md` (Ch 11, pp 165–176)

Carver's framework (ex-AHL PM) is **risk-first**: size to a volatility target;
returns are an OUTPUT, not the objective. This is deliberately NOT the
"max-contracts-to-DD-limit" sizer the operator rejected as primary.

**PRIMARY sizer — Carver vol-target half/quarter-Kelly chain:**
1. Per-lane **realistic Sharpe** = backtest annualized Sharpe (replay) × **0.75**
   OOS-decay discount (Carver p143-144).
2. **Vol target** = realistic Sharpe, then **half-Kelly** (halve); **halve AGAIN
   for negative skew** (Carver p146, Table 25). Our top CORE lanes have skew
   −0.2 to −0.6 (negative) → **quarter-Kelly** applies. Vol target clipped to
   Carver's recommended band and a hard 50% ceiling.
3. `annual_cash_vol = equity × vol_target`; `daily_cash_vol = annual/16`
   (16 = √256; Carver p137-138).
4. Subsystems are **vol-standardised** then combined with **equal instrument
   weights** (Carver p168 explicitly discourages Sharpe-weighting) × a
   **diversification multiplier D = 1/√(w'Cw)**, **HARD-CAPPED at 2.5** (Carver
   p170 — a crisis-correlation-jump robustness cap, not a statistical one).
5. Convert each lane's cash-vol budget to integer contracts via its replay daily
   $-volatility; round to integer blocks.

**SECONDARY cross-check — DD-budget feasibility (NOT the objective):**
- After Carver sizing, recompute modeled rolling-90d DD on the sized book.
- Assert it sits under the **$4,000 design target**. If Carver sizing exceeds the
  design target, **trim** (risk-first: the DD limit is a backstop the vol target
  must respect, never a target to fill).
- Report the legacy greedy-to-$4k sizing alongside for comparison ONLY, clearly
  labelled "return-max reference (not recommended primary)".

**Slippage stress** (1× / 1.5× / 2×): recompute DD with a per-fill $ haircut;
assert the 1.5× DD stays ≤ the $6k hard stop. A book whose 1.5× DD breaches $6k
is reported FAIL-stress.

## Dedup + concentration

- **Edge-family dedup:** one lane per `family_hash` (prefer `is_family_head`, else
  highest annualized dollar expectancy).
- **Concentration penalty:** cap how many lanes share a single
  `(instrument, orb_label)` session so the book is not one correlated bet. The
  Carver diversification multiplier also penalizes correlation directly (a
  correlated book has lower `D`, hence less size).

## Three books compared

1. **CORE-only** (sample ≥ 100) — institutionally clean.
2. **CORE + REGIME_NEAR_CORE** — adds upper-REGIME lanes, all flagged provisional.
3. **Aggressive + REGIME_THIN overlays** — only the thin (N 30–99 below near-core)
   band. If none qualify after dedup, the report says so explicitly rather than
   re-printing an identical book.

## Outputs

- Ranked candidate table (annualized dollar expectancy, tier, snapshot vs replay).
- Per-book: expected $/yr, modeled max DD, 1.5× stress DD, 2× stress DD,
  contracts/lane.
- Top rejected high-ExpR lanes and why (REGIME-thin / family-dup / correlated /
  DD-trim).
- Recommended live-safe `self_funded_30k` profile **sketch** (lanes + contracts +
  projected $/yr + modeled & stress DD). This is a SKETCH printed to stdout — the
  report writes NO profile, no `prop_profiles.py` edit, no live state.

## Selection-inflation honesty (DSR finding — added after grounding pass)

Grounding the Sharpe input against `bailey_lopez_de_prado_2014_deflated_sharpe.md`
surfaced a material finding: the lanes' raw backtest Sharpes (2.2–2.4) are
**selection-inflated**. The project's own canonical Deflated-Sharpe Z-score
(`validated_setups.sharpe_haircut`, computed by
`trading_app.strategy_discovery._compute_haircut_sharpe`) is **negative for nearly
every CORE lane** — but that DSR is deflated against `n_trials_at_discovery`
≈ 36,000 GLOBAL discovery trials, which is the **discredited brute-force K** that
`research-truth-protocol.md` explicitly rejects. The lanes were FDR-validated under
the correct per-family K.

**Resolution (operator-approved "conservative haircut chain"):**
- **Sizing input** = REPLAY Sharpe × Carver's documented **0.75** OOS discount
  (`carver_2015_volatility_targeting_position_sizing.md` p143-144) — the
  literature-grounded haircut that is NOT contaminated by the global-K problem.
- **DSR Z-score is surfaced per-lane** (`DSRz` column) for honesty so the
  selection-inflation is VISIBLE, but is NOT used to size (it would zero the book
  for the wrong reason).

This is why the deliverable was **reframed risk-first**: return is the OUTPUT of a
defensible vol-target, never the objective. The earlier return-max greedy-to-DD
sizer was removed — it produced 29–54-contract books that are exactly the
overconfidence Carver Ch12 p179 warns against.

## Survival check

Per `pre_registered_criteria.md` Criterion 11 (90-day account-death Monte Carlo,
≥70% survival), the report prints a **survival PROXY** = historical-worst 1.5×-stress
DD headroom to the $6k hard stop, and flags that the **canonical Monte Carlo
(`account_survival.evaluate_profile_survival`) must run on a real profile before any
capital**. The report writes no profile, so it cannot run the binding gate itself.

## Known limitations (honest)

- `trades_per_year` for grandfathered rows is a Mode-B-era snapshot; the realized
  replay frequency is the truth and may differ. Ranking uses the snapshot (that's
  its purpose); sizing/DD uses replay.
- Annualized $/yr is directional, bounded by each lane's available history window.
  Books with different common windows are **NOT** directly comparable on $/yr — the
  report prints a cross-book window-mismatch caveat and prefers the longer-window
  figure as more bankable.
- Slippage-at-scale is modeled as a haircut sweep, not measured per-fill.
- No correlation matrix is estimated parametrically for sizing; the diversification
  multiplier IS computed from the empirical correlation matrix
  (`D = 1/√(w'Cw)`, capped 2.5 per Carver Ch11 p170), and residual concentration is
  bounded by a session cap.
- Hydration is **session-stratified** (top-N per `(instrument, orb_label)`) so a
  single high-rank session cannot starve book diversity — a flat top-N rank cap did.

## Scope boundary

NEW files only: this doc + `scripts/reports/report_max_return_own_capital.py`.
NO edits to `pipeline/`, `trading_app/`, schema, `account_survival.py`, C11, or any
profile. Read-only against `gold.db`.
