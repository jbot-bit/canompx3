# TradingView AI Backtesting Engine Teardown

**Date:** 2026-05-11
**Mode:** clean-room reverse engineering from user-supplied transcript and public docs
**Scope:** product/workflow assessment only; no proprietary ZIP, prompts, or source code used

## Verdict

This is likely a small, easily buildable **AI workflow bundle**, not a fundamentally new backtesting technology.

What the seller appears to offer:

1. A folder layout that Codex/Claude can understand.
2. A Python OHLCV backtesting script or package.
3. A few sample CSV datasets across crypto/stocks/timeframes.
4. Prompt templates for converting Pine indicators to strategies, improving strategies, adding indicators, and fetching data.
5. A variant-search loop that asks the LLM to generate many parameter/logic variants, runs local backtests, then emits PineScript for TradingView.
6. A claim that the engine can reproduce TradingView Strategy Tester closely enough for user confidence.

The build itself is not hard. The hard part is keeping it honest. The transcript's strongest claims are exactly where bias risk is highest: "6,713 variants", "best variant", "avoid overfitting", and "same KPIs as TradingView".

## Reproduction / Outputs

This was a documentation and intake audit, not a backtest or strategy-validation run.

Inputs reviewed:

- User-supplied transcript/description of the product workflow.
- Public TradingView Pine strategy documentation cited below.
- Repo-local institutional methodology anchors: `docs/institutional/pre_registered_criteria.md` Criterion 2 and `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`.

Outputs produced:

- This teardown result, with no strategy-performance claim.
- `docs/audit/intake/2026-05-11-autotrading-school-tv-ai-engine.yaml`, classifying the source as `DOC_ONLY`.
- `scripts/tools/external_strategy_intake_check.py`, a read-only validator for future external-intake YAML records.
- `docs/prompts/external_strategy_intake_prompt.md`, a prompt contract for converting external trading ideas into bounded intake records before preregistration.

Reproduction commands:

```bash
python scripts/tools/external_strategy_intake_check.py docs/audit/intake/2026-05-11-autotrading-school-tv-ai-engine.yaml
pytest -q tests/test_tools/test_external_strategy_intake_check.py
```

Expected output:

- `PASS external strategy intake validation (1 file(s))`
- `20 passed` for the external-intake validator tests.

## Disconfirming Checks / Limitations

- This audit did not inspect the seller's proprietary ZIP, prompt files, or source code. The implementation-shape claims are INFERRED from the supplied transcript and public product behavior, not MEASURED from source.
- No live, paper, or historical trade was replayed. Any statement about strategy profitability, edge quality, or deployability is UNSUPPORTED.
- The phrase "easily buildable" refers to software construction difficulty, not to research validity or expected trading performance.
- Public TradingView documentation can establish broker-emulator and Pine semantics, but it does not establish parity for any private script or dataset.
- The external-intake gate is a screening guard only. A `DOC_ONLY` or `PREREG_CANDIDATE` intake record is not a preregistration, validation result, deployment approval, or live-trading permission.

## What Is Actually Being Sold

| Claimed feature | Likely concrete implementation | Assessment |
|---|---|---|
| "AI Backtesting Engine" | Python backtester plus prompts that tell Codex how to use it | Straightforward to recreate |
| "Convert TradingView indicator to strategy" | LLM rewrites `indicator()` Pine into `strategy()` Pine and invents entries/exits | Useful, but subjective and non-deterministic |
| "Improve strategy" | Generate/search many variants, rank on backtest metrics, export best Pine | Easy mechanically; dangerous statistically |
| "Avoid overfitting" | Prompt instructs IS/OOS and cross-market validation | Not enough by itself; still leak-prone |
| "Same as TradingView" | Simulator tries to mimic TradingView broker emulator assumptions | Possible for simple scripts; fragile for complex Pine |
| "Works on crypto, stocks, commodities" | CSV fetchers from exchanges/Yahoo/other feeds | Data parity is the problem, not code |
| "No coding skills" | The LLM edits files and runs scripts | Plausible UX, but user cannot audit failure modes |

The marketing wrapper is bigger than the code. A usable v1 would be a few modules:

- `data/`: OHLCV CSVs with symbol, timeframe, source, timezone, start/end.
- `strategies/`: generated `.pine` plus Python equivalents.
- `backtester/`: bar simulator, costs, order model, metrics.
- `optimizer/`: variant generator, trial ledger, ranking, train/test split.
- `prompts/`: convert, improve, combine-indicator, fetch-data workflows.
- `reports/`: backtest table, trade list, equity curve, rejected variants.

## Bias And Gap Audit

### 1. The "6,713 variants" claim is a red flag

Testing thousands of variants is not proof that the final version is best. It is a textbook multiple-testing setup. This repo's own doctrine cites Bailey, Borwein, Lopez de Prado, and Zhu: as trial count rises, the best observed backtest can be produced by noise. Harvey and Liu also warn that iterative OOS revision is not truly out-of-sample.

For this product to be honest, it must log every attempted variant and adjust conclusions for the trial count. A prompt saying "avoid overfitting" does not do that.

Minimum honest controls:

- immutable trial ledger with every variant, seed, params, metric, and rejection reason
- pre-declared trial budget before the optimizer runs
- deflated Sharpe / multiple-testing haircut or equivalent
- no hidden reruns after seeing OOS or cross-market failures
- final report states "selected from N variants", not "best strategy"

### 2. IS/OOS can still leak

The transcript says the prompt tells Codex to use in-sample, out-of-sample, and cross-market validation. That is only valid if OOS is used once. If Codex sees OOS results, edits the strategy, and reruns until OOS improves, OOS becomes part of the training loop.

Minimum honest controls:

- nested split: train for variant generation, validation for model choice, final untouched holdout for one-shot audit
- locked split dates before search
- no parameter or logic changes after final holdout
- explicit "OOS consumed" marker in the output

### 3. Cross-market validation is easy to fake accidentally

Cross-market validation only helps if the markets/timeframes are declared before optimization. If the user or LLM picks the 10 assets after seeing results, it becomes another selection layer.

Minimum honest controls:

- pre-declared asset universe and timeframes
- source and start/end date for every dataset
- pass/fail threshold defined before the run
- report all failures, not just the symbols where the strategy works

### 4. TradingView parity does not prove edge

TradingView itself uses a broker emulator, not real fills. TradingView's official Pine docs say strategies simulate trades using available chart data and broker-emulator assumptions. The docs also describe default intrabar path assumptions and Bar Magnifier as a lower-timeframe override with limits.

Matching TradingView KPIs proves only that the local simulator matches TradingView's historical simulation for that case. It does not prove:

- the strategy survives live execution
- the data source matches the exchange/broker
- intrabar TP/SL order is knowable
- slippage/liquidity is realistic
- the selected strategy is not a multiple-testing artifact

Official reference points:

- TradingView strategy FAQ: https://www.tradingview.com/pine-script-docs/faq/strategies/
- TradingView strategy concepts / broker emulator: https://www.tradingview.com/pine-script-docs/concepts/strategies/
- TradingView `request.security()` and lookahead docs: https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/

### 5. OHLC bars cannot fully resolve TP/SL order

The seller admits Bar Magnifier cannot be simulated without lower-timeframe/tick data. That limitation matters. If both target and stop are inside the same OHLC bar, a bar-only simulator must choose an assumption. TradingView has its own emulator assumptions; a conservative research engine should either use lower timeframe data or score ambiguous bars pessimistically.

Minimum honest controls:

- configurable fill model: TradingView-like, conservative, and lower-timeframe magnified
- same-bar TP/SL ambiguity count in every report
- reject strategies whose performance depends heavily on ambiguous bars

### 6. Pine-to-Python conversion is fragile

The seller's limitation around `security()` and custom libraries is real, but understated. Fragility includes:

- `request.security()` lookahead/repainting behavior
- higher-timeframe confirmation delays
- `var` state and recursive series logic
- pyramiding and partial exits
- `strategy.exit()` bracket behavior
- `calc_on_every_tick`, `process_orders_on_close`, and fill settings
- non-standard charts such as Heikin Ashi
- imported/custom libraries

An MVP should not promise arbitrary Pine conversion. It should support a restricted Pine subset or generate Python and Pine from the same internal strategy spec.

### 7. Data mismatch can dominate the result

The transcript tells users to align TradingView's chart with the data source and start date. That is correct, but it reveals the core weakness: if the local data does not match TradingView's symbol, session, adjustment, timezone, or missing-bar policy, the comparison is not meaningful.

Minimum honest controls:

- dataset manifest for every CSV
- timezone normalization
- duplicate/missing bar checks
- adjusted vs unadjusted price flag
- exchange/session calendar declaration
- first/last timestamp surfaced in the report

## Honest Clone Blueprint

An honest in-house version should be built as a research harness, not a "profit engine".

### MVP Capability

- Load OHLCV CSVs with strict schema validation.
- Run a small strategy interface: `on_bar(history, state, params) -> orders`.
- Simulate market, limit, stop, stop-loss, and take-profit orders.
- Support TradingView-like historical fill assumptions and a conservative fallback.
- Calculate trade list, equity curve, net profit, drawdown, win rate, expectancy, Sharpe, exposure, and ambiguity stats.
- Generate Pine from a limited internal template, not arbitrary Python.
- Run bounded parameter sweeps with a trial ledger.
- Produce a report that separates train, validation, and untouched holdout.

### Non-Negotiable Bias Controls

- Pre-run config locks asset universe, date splits, metric, max trials, and allowed parameters.
- Every trial is recorded.
- OOS is one-shot; no iterative "fix after OOS".
- Trial count feeds a multiple-testing warning or formal haircut.
- Cross-market validation reports all tested markets.
- Ambiguous intrabar fills are counted and stress-tested.
- Final output says "candidate", never "validated edge", unless it passes a separate forward/OOS protocol.

### What Not To Build First

- Full Pine parser.
- Arbitrary `request.security()` conversion.
- Live exchange automation.
- "AI picks strategy from scratch" without a bounded search protocol.
- Marketing metrics like raw net profit optimized across leverage/drawdown without trial correction.

## Comparison To This Repo

This repo already has stricter research standards than the product appears to describe:

- sacred holdout discipline
- BH/FDR and MinBTL framing
- DSR support
- walk-forward/OOS validation
- cost and slippage modeling
- lookahead/feature-timing doctrine
- backtest/live parity checks
- mechanism-first interpretation

The product's useful part is not the statistical discipline. It is the convenience loop: "LLM edits strategy -> local backtest -> export Pine -> compare in TradingView." That loop is valuable if treated as an idea generator. It is dangerous if treated as proof.

## Bottom Line

Yes, this is easily creatable. A competent engineer can build the core workflow quickly.

The defensible version is not "AI finds great strategies". It is:

> A local strategy prototyping harness that lets an LLM generate bounded candidates, backtest them reproducibly, export simple Pine strategies, and produce an audit trail showing exactly how much overfit risk remains.

If we build it, the edge is not matching TradingView. The edge is refusing to let the LLM hide selection bias.
