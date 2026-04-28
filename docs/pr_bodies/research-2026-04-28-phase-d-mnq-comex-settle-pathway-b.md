# Phase D D4/D5 + Amendment 3.2 MinTRL doctrine

Bundles the Phase D D4/D5 research closeout with the doctrine that emerged from auditing it. D4 PARK stands; D5 KILLED honestly by locked KILL_PAIRED_P + KILL_C9 gates; Amendment 3.2 MinTRL gate now binds every Pathway B pre-reg.

## Evidence

**Canonical re-derivations** (all from `bars_1m`-grounded `daily_features` JOIN `orb_outcomes` via `pipeline.paths.GOLD_DB_PATH`, scratch policy `COALESCE(pnl_r, 0.0)`, IS `< 2026-01-01`):

- **D4 cell** (MNQ COMEX_SETTLE O5 RR1.0 long ORB_G5 garch>70): N_IS_on=199, ExpR_IS_on=+0.2453, ΔIS=+0.2199, t=3.18 — verified exact match to D4 result file.
- **D5 cohort** (deployed RR1.5 BOTH-SIDES ORB_G5 N=1577): SR_pt_flat=+0.0825, SR_pt_D5=+0.1005, SR_ann_flat=+1.2381, SR_ann_D5=+1.5086, abs SR diff=+0.2705, rel uplift=+21.85% — runner reproduces pre-reg's `independent_sql_baseline` exactly.
- **D5 paired test** (per-trade R differences): paired t=−1.45, p=0.148, block-bootstrap paired p=0.136 — KILL_PAIRED_P fires.
- **D5 C9 era stability**: 2019 ExpR_D5=−0.108 < −0.05 floor — KILL_C9 fires.
- **Cross-session K=12 BH-FDR** (D5 framing): best q=0.0707 at COMEX_SETTLE; nothing survives q<0.05. The garch>70 effect is COMEX_SETTLE-specific, not universal.
- **Empirical sd_monthly_sharpe_diff** (80 month-pairs, IS): 0.0475 (rough estimate was 0.30; ~6.3× too conservative). MinTRL recomputed: 7 OOS months = 0.59 years → STANDARD per Amendment 3.2.

## Claims

1. **D4 PARK_PENDING_OOS_POWER stands** — IS edge real, OOS dir-matched, but N_OOS_on=17 < 50 power floor (Amendment 3.1).
2. **D5 conditional half-size DEAD** — KILL verdict from locked pre-reg. Sharpe gain was variance-driven, not expectancy-driven (paired-p exposed it). 2019 era instability (C9 fail) confirms half-sizing the off-cohort amplifies negative-skew eras rather than damping them.
3. **Carver Ch12 p.192 mechanism is cell-specific, not universal** — addendum (commit `25ed6f09`) self-corrected (commit `2e858956` → `416855cd`) after K=12 BH-FDR showed only COMEX_SETTLE survives. Mechanism reading remains valid for the one cell.
4. **Amendment 3.2 MinTRL gate binds every Pathway B pre-reg** — must compute `min_trl_years` at write time using empirical (not estimated) sd_R, classify STANDARD/EXTENDED_PARK/NOT_OOS_CONFIRMABLE. Bare PARK_PENDING_OOS_POWER label forbidden when MinTRL > 2 years.
5. **Lock-before-run discipline caught D5** — without locked KILL_PAIRED_P, the +21.85% rel uplift headline would have been promoted CANDIDATE_READY. Pre-reg gates locked the honesty.

## Disconfirming Checks

- **Independent evidence-auditor pass** (separate context, no prior framing): re-ran 10 load-bearing claims; flagged 3 errors that are all retracted in `416855cd` — q=0.391 fabricated (no script reproduces it; correct=0.0707), MGC SR_ann=0.094 used wrong years divisor (correct=0.124 for 4-year MGC span), `sd_monthly_sharpe_diff=0.30` was rough estimate (empirical=0.0475). Substantive conclusions all unchanged; evidence numbers corrected.
- **Pre-reg lock-before-run discipline**: D5 runner committed AFTER pre-reg + MinTRL amendment landed (`52ea845d` < `416855cd` < `8270ec8d`). No post-hoc threshold rescue possible.
- **KILL_BASELINE_SANITY**: D5 runner reproduces every pre-reg `expected_*` value to ≤0.0001 absolute diff. Verifies the runner is reading the same canonical data the pre-reg author cited.
- **Pre-commit gauntlet 8/8 PASSED on every judgment commit** — drift (114 checks, 0 skipped, 10 advisory), claim hygiene, behavioral audit, checkpoint guard, syntax, lint, format, fast tests.
- **MGC sister-instrument coherence claim withdrawn** — small-sample illusion; full N=66 both-sides gives p=0.805, SR_ann=+0.124 (noise).

## Grounding

All literature citations use verbatim extracts from `docs/institutional/literature/` (which extracts from `resources/` PDFs with page numbers):

- **Carver Ch 9-10** (`carver_2015_volatility_targeting_position_sizing.md`) — vol-targeting position sizing framework; `Position ∝ forecast/10`. Grounds D5's sizing-conditional-on-forecast premise.
- **Carver Ch 12 p.192** (`carver_2015_ch12_speed_and_size.md` addendum 2026-04-28) — holding-period × stop-level matching. Verbatim p.192 quote. Self-corrected to cell-specific scope after Stone 1 K=12 BH-FDR refuted universality.
- **Chan Ch 7 p.155-157** (`chan_2013_ch7_intraday_momentum.md`) — stop-cascade mechanism on 5-30-min horizon. Verbatim p.155 + p.157 quotes at lines 22 + 40.
- **Bailey-LdP 2014** (`bailey_lopezdeprado_2014_dsr_sample_selection.md`) — Sharpe ratio confidence intervals are sample-size dependent. Grounds Amendment 3.1 OOS power floor + Amendment 3.2 MinTRL gate.
- **Bailey 2013 p.7** (`bailey_et_al_2013_pseudo_mathematics.md`) — "high Sharpe IS, zero Sharpe OOS." Grounds the OOS-positive deployment principle that motivates Amendment 3.2 NOT_OOS_CONFIRMABLE classification.

**Canonical sources delegated, never re-encoded:** ORB_G5 ≥5pt threshold from `trading_app/config.py:2980`; `HOLDOUT_SACRED_FROM` from `trading_app/holdout_policy`; cost specs from `pipeline/cost_model.py`; DB path from `pipeline/paths.py`; GARCH feature from `pipeline/build_daily_features.py:1468-1497`.

**Memory updated** (cross-session): `feedback_d5_lock_before_run_2026-04-28.md` captures the three lessons (lock-before-run discipline, auditor-fabrication catch, Amendment 3.2 binding). Indexed in `MEMORY.md` under Audit playbooks.

## Not done in this PR

- D2 (B-MES-LON) and D3 (B-MNQ-NYC) pre-regs — same Amendment 3.2 MinTRL math will apply; defer until needed
- Continuous-scaling D5 v2 (Carver Ch9 forecast-proportional sizing as continuous function) — would require new theory citation
- Real-money flip on D4 — D5 KILLED; deployed lane unchanged; Amendment 3.2 NOT_OOS_CONFIRMABLE per-trade-R framing unresolved
- RULE 14/15/16 numbering doctrine drift — separate housekeeping task
- Branch hygiene (`live_signals_*.jsonl`, `live_session.stop`, `bias-grounding-guard.py`) — separate concern
