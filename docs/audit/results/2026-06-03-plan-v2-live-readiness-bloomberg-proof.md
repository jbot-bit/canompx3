# Plan v2 Execution - Live Readiness, Bias Gaps, and Professional Proof

Date: 2026-06-03
Profile: `topstep_50k_mnq_auto`
Checkout: `main` at `c12f43f6`
Status scope: measured on a dirty Windows checkout; do not treat as clean-main proof.

## Scope

This report executes plan v2 for `topstep_50k_mnq_auto`: DB availability, Criterion 11, Criterion 12, telemetry maturity, paused-lane reason plumbing, execution attribution dry-run, dashboard static smoke, drift, and phase-7 live audit.

## Question

Is the profile strictly live-ready after the plan-v2 blocker review, and are the prior no-DB, silent-paused-lane, attribution, dashboard, and drift gaps resolved with measured evidence?

## Reproduction

Run the commands listed in each measured section. The minimum rerun set is:

- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto`
- `python scripts\tools\live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`
- `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`
- `python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory`
- `python scripts\audits\run_all.py --phase 7`
- `python -m pytest tests\test_trading_app\test_account_survival.py tests\test_tools\test_live_readiness_report.py -q`

## Outputs

Primary output is this report. Code/test outputs are the strict-C11 launch-blocking behavior in `scripts/tools/live_readiness_report.py`, focused coverage in `tests/test_tools/test_live_readiness_report.py`, and the durable closeout note in `docs/plans/2026-06-03-strict-c11-readiness-closeout.md`.

## Bottom Line

- MEASURED: `gold.db` exists at `C:\Users\joshd\canompx3\gold.db` and was readable for readiness/account-survival checks.
- MEASURED: Criterion 11 operational survival passes at 73.3% against the 70% threshold.
- MEASURED: Criterion 11 strict prop-account diagnostics fail: 7 historical daily-loss breach days and max observed 90-day drawdown `$2,788` against strict budget `$1,600`.
- MEASURED: Criterion 12 SR state is valid, age 1 day, all three deployed lanes are `CONTINUE`.
- MEASURED: strict C11 diagnostics are now treated as launch-blocking strict-readiness warnings.
- MEASURED: telemetry maturity is 9/30 profile-scoped trading days; advisory for express/funded profile in current policy.
- MEASURED: `paper_trade_logger --sync --dry-run` retried successfully and found zero backfillable trades.
- MEASURED: dashboard static smoke passed 42 tests; phase-7 live audit passed 11 checks; fast drift passed 137 checks with 15 advisories.

## Disconfirming Checks First

### No-DB fallback

MEASURED: Not applicable in this Windows checkout. DB exists and is 7,628,664,832 bytes.

Command:
`python -c "from pipeline.paths import GOLD_DB_PATH; ..."`

Result:
`C:\Users\joshd\canompx3\gold.db`, exists `True`; `live_journal.db` exists `True`.

### Dirty-checkout caveat

MEASURED: The checkout is dirty in live-readiness/account-survival and MNQ research files before this report was created. This audit therefore measures the working tree, not pristine `main`.

Dirty files include:
- `trading_app/account_survival.py`
- `scripts/tools/live_readiness_report.py`
- `tests/test_trading_app/test_account_survival.py`
- `tests/test_tools/test_live_readiness_report.py`
- MNQ single-leg replacement research/result files

## Blocker Taxonomy

### `BLOCKED_CRITERION_11_STRICT_DIAGNOSTICS`

MEASURED.

Command:
`python -m trading_app.account_survival --profile topstep_50k_mnq_auto`

Evidence:
- Source days: 2048, `2019-05-31 -> 2026-06-03`
- Horizon: 90d
- Paths: 10000
- Operational pass: 73.3%
- Daily-loss breach probability: 26.5%
- Strict diagnostics: fail
- Historical daily-loss breach days: `2022-05-12`, `2025-04-04`, `2025-04-07`, `2025-04-09`, `2025-04-23`, `2026-02-20`, `2026-05-19`
- Max observed 90-day DD: `$2,788`
- Effective strict DD budget: `$1,600`

Conclusion:
Criterion 11 is operationally green under the current gate, but a professional risk read should not ignore the strict account diagnostic failure.

### `CRITERION_12_VALID`

MEASURED.

Command:
`python scripts\tools\live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`

Evidence:
- `criterion12.latest_verdict=valid`
- `criterion12.age_days=1`
- Deployed lanes:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, SR `CONTINUE`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, SR `CONTINUE`
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`, SR `CONTINUE`

### `EXECUTION_ATTRIBUTION_DRY_RUN_CLEAR`

MEASURED.

Command:
`python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`

Result:
Dry-run completed after the DB lock cleared.

Evidence:
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`: 0/0 trades pass filter
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`: 0/2 trades pass filter
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`: 0/0 trades pass filter
- Total dry-run backfill: 0 trades, +0.00R

Conclusion:
No mutating sync was needed from this dry-run.

### `PAUSED_REASON_PLUMBING_DRIFT`

MEASURED.

The proof pack previously reported 844/845 paused lanes as `unspecified`, but the allocation JSON uses the field name `reason`, not `status_reason`, `lifecycle_block_reason`, or `pause_reason`. The report now preserves `reason`.

Direct allocation parse:
- `paused_count=845`
- reason counts:
  - 781: `strict live gate: SR status UNKNOWN is not CONTINUE for fresh live allocation`
  - 35: `Session regime COLD (-0.1011)`
  - 12: `live tradeability gate: E2 deployment-unsafe: filter_type 'PD_CLEAR_LONG'...`
  - 5: `live tradeability gate: E2 deployment-unsafe: filter_type 'PD_GO_LONG'...`
  - 2: `strict live gate: SR status ALARM is not CONTINUE for fresh live allocation`

Conclusion:
The plan's concern was valid, but the measured defect was report reason-plumbing drift, not 845 unexplained operational pauses.

### `TELEMETRY_ADVISORY`

MEASURED.

Profile-scoped telemetry:
- Verdict: `UNVERIFIED_INSUFFICIENT_TELEMETRY`
- Days: 9/30
- Qualifying records: 57/119
- Policy: advisory for express/funded profile in current readiness report.

## Dashboard and Drift

MEASURED:
- `python -m pytest tests\test_trading_app\test_bot_dashboard.py -q`: 42 passed, 3 pytest config warnings, ignored Windows temp cleanup `PermissionError`.
- `python scripts\audits\run_all.py --phase 7`: phase 7 passed, 11 checks.
- `python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory`: `SUMMARY: clean passed=137 advisory=15`.
- `ruff check ... --quiet`: passed.
- `python -m py_compile scripts\tools\live_readiness_report.py trading_app\account_survival.py`: passed.
- `git diff --check`: passed with line-ending warnings only.
- `python -m pytest tests\test_trading_app\test_account_survival.py tests\test_tools\test_live_readiness_report.py -q`: 50 passed, 3 pytest config warnings, ignored Windows temp cleanup `PermissionError`.

Runtime dashboard smoke:
- MEASURED: port 8080 is listening under PID 53672.
- SKIPPED: no browser/runtime smoke was performed because this task did not start or control that running dashboard process.

## Literature and Source Grounding

Local methodology support:
- `RESEARCH_RULES.md` requires local literature grounding for MinBTL, DSR, multiple-testing, theory-first research, and live SR monitoring.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` grounds MinBTL and overfit risk.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` grounds DSR and multiple-testing deflation.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` grounds the high t-stat threshold and false-discovery risk.
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` grounds theory-first discipline.
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` grounds SR live drift monitoring.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` grounds position sizing and low effective vol targets under tight prop-firm drawdown.

Coverage checks:
- `python scripts\tools\check_pdf_tooling.py`: PyMuPDF and OCR tooling available; sample PDF opened.
- `python scripts\tools\check_literature_coverage.py`: 6 indexed resources, 4 with curated extracts, 3 raw resource files present locally, 3 raw PDFs missing locally. Use tracked extracts where raw PDFs are absent.

Official/primary prop-firm support:
- Topstep official MLL page currently lists `$2,000` MLL for 50K Trading Combine / Express Funded accounts and says trading stops immediately if balance hits that level.
- Topstep official DLL page currently lists `$1,000` daily loss limit for 50K when applied and says breach auto-liquidates, cancels pending orders, and blocks new trades until the next trading day.
- Topstep official Express Funded rules currently list 50K max loss `$2,000`, 100K `$3,000`, 150K `$4,500`, and warn that rules may update.

## Research / Live Review Verdicts

RESEARCH METHOD REVIEW
- Verdict: UNVERIFIED for any new edge, promotion, allocation, or ASX-open claim.
- Reason: no new preregistered scan was run; 2026 remains holdout/monitoring; this pass measured live-readiness and report plumbing only.
- Minimum next evidence for ASX: prereg first, K declared, ASX cash open distinct from existing `SESSION_CATALOG`, no 2026 selection, costs, BH/FDR, era splits, holdout policy.

LIVE RISK AUDIT
- Capital impact: deploy-readiness/account-risk.
- Decision: BLOCK strict live readiness until strict C11 diagnostics clear or an explicit capital-risk exception is approved; execution attribution sync is also blocked by DB lock.

## Final Classification

- `NEED_DB_EVIDENCE`: CLEARED for Windows checkout; DB exists and was used.
- `BLOCKED_CRITERION_11`: BLOCKING FOR STRICT LIVE READINESS. Operational gate passes; strict account diagnostic fails.
- `BLOCKED_CRITERION_12`: CLEARED. Current SR state valid.
- `BLOCKED_TELEMETRY`: ADVISORY under current express/funded policy; 9/30 days.
- `PAUSED_UNSPECIFIED`: RECLASSIFIED as report reason-plumbing drift.
- `CONFIG_DISCONNECTED`: NOT MEASURED as a blocker; active allocation has 3 lanes and profile match is intact in proof pack.
- `EXECUTION_ATTRIBUTION`: DRY-RUN CLEAR; no backfillable trades found.

## Next Smallest Fixes

1. Clear strict C11 diagnostics by reducing account risk or explicitly record an approved exception before any strict live launch.
2. If ASX is still desired, create prereg only; do not add session catalog entries or scan without the DB-backed prereg front door.
