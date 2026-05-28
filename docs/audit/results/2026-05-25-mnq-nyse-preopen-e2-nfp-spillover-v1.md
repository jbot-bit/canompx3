# NYSE_PREOPEN MNQ E2 NFP-spillover v1 — Stage 4b verdict

**Prereg file:** `docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml`
**Prereg SHA (locked):** `40f032aa3ecc99d3fdf5721e6af09d0de13885a94cbea54861c13a60f2be6436`
**Runner:** `research/mnq_nyse_preopen_e2_nfp_spillover_v1.py` (Stage 4a)
**Emitter:** `scripts/research/emit_nyse_preopen_verdict.py` (Stage 4b)
**Result CSV:** `docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`
**Generated:** 2026-05-28T14:17:49.889077+00:00

## Scope

Strict-Chordia ``t >= 3.79`` (no-theory) family verdict for
``MNQ_NYSE_PREOPEN_E2_CB1`` at K_family = 27
(3 ORB apertures x 3 RR targets x 3 NFP-day splits). Mode A IS/OOS at
``HOLDOUT_SACRED_FROM = 2026-01-01``. NYSE-holiday
contamination excluded upstream via ``pipeline.market_calendar.is_nyse_holiday``
(Stage 2 wiring).

## Headline

**0 of 27 cells PASS strict-Chordia at t >= 3.79 + BH-FDR q < 0.05.**

## Verdict breakdown

| Verdict | Count |
|---|---:|
| `CONDITIONAL_OOS_UNDERPOWERED` | 6 |
| `FAIL_CHORDIA_STRICT` | 12 |
| `UNVERIFIED_DST_IMBALANCE` | 9 |

## Passing cells

_No cells cleared every gate (strict-Chordia + BH-FDR + DST-balance + OOS power floor)._

## Held back by OOS power floor (RULE 3.3)

Cells where IS strict-Chordia clears (t_IS >= 3.79, N >= 100, ExpR_IS > 0,
BH q < 0.05, DST-balanced) but OOS dir-flip at an insufficient power tier
prevents promotion. These are not refutations -- the OOS slice cannot
confirm OR refute the IS signal at actionable power.

- `O30_RR1.0_all_days` — t_IS=4.534 N_IS_on=1674 ExpR_IS=0.1034 BH q=0.000042; OOS N=84 ExpR_OOS=-0.0635 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=556 N_EDT=1118 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`
- `O30_RR1.0_non_nfp_days` — t_IS=4.001 N_IS_on=1600 ExpR_IS=0.0934 BH q=0.000178; OOS N=81 ExpR_OOS=-0.0288 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=530 N_EDT=1070 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`
- `O30_RR1.5_all_days` — t_IS=4.649 N_IS_on=1674 ExpR_IS=0.1339 BH q=0.000042; OOS N=84 ExpR_OOS=-0.1384 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=556 N_EDT=1118 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`
- `O30_RR1.5_non_nfp_days` — t_IS=4.057 N_IS_on=1600 ExpR_IS=0.1194 BH q=0.000176; OOS N=81 ExpR_OOS=-0.1065 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=530 N_EDT=1070 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`
- `O30_RR2.0_all_days` — t_IS=4.401 N_IS_on=1674 ExpR_IS=0.1488 BH q=0.000052; OOS N=84 ExpR_OOS=-0.1088 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=556 N_EDT=1118 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`
- `O30_RR2.0_non_nfp_days` — t_IS=3.873 N_IS_on=1600 ExpR_IS=0.1337 BH q=0.000252; OOS N=81 ExpR_OOS=-0.0758 dir_match=NO tier=STATISTICALLY_USELESS; DST N_EST=530 N_EDT=1070 (BALANCED) -> `CONDITIONAL_OOS_UNDERPOWERED`

## Per-cell K=27 table

Sorted in prereg lock order (ORB x RR x SPLIT).

| Cell | N_IS | ExpR_IS | t_IS | BH q | N_OOS | ExpR_OOS | dir_match | OOS pwr tier | N_EST | N_EDT | DST | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|:---:|---|
| `O5_RR1.0_all_days` | 1673 | -0.0270 | -1.237 | 0.182489 | 84 | 0.0334 | NO | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O5_RR1.0_nfp_days_only` | 74 | 0.0770 | 0.718 | 0.342781 | 3 | -0.3809 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O5_RR1.0_non_nfp_days` | 1599 | -0.0319 | -1.427 | 0.148331 | 81 | 0.0488 | NO | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O5_RR1.5_all_days` | 1673 | -0.0158 | -0.580 | 0.379466 | 84 | -0.0163 | YES | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O5_RR1.5_nfp_days_only` | 74 | 0.0651 | 0.480 | 0.388028 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O5_RR1.5_non_nfp_days` | 1599 | -0.0195 | -0.703 | 0.342781 | 81 | 0.0201 | NO | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O5_RR2.0_all_days` | 1673 | -0.0420 | -1.335 | 0.163818 | 84 | 0.1136 | NO | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O5_RR2.0_nfp_days_only` | 74 | 0.1263 | 0.790 | 0.342781 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O5_RR2.0_non_nfp_days` | 1599 | -0.0498 | -1.553 | 0.125380 | 81 | 0.1548 | NO | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR1.0_all_days` | 1673 | -0.0018 | -0.078 | 0.469091 | 84 | 0.0771 | NO | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR1.0_nfp_days_only` | 74 | 0.1723 | 1.607 | 0.125380 | 3 | 0.2939 | YES | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O15_RR1.0_non_nfp_days` | 1599 | -0.0098 | -0.423 | 0.394434 | 81 | 0.0691 | NO | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR1.5_all_days` | 1673 | 0.0143 | 0.506 | 0.388028 | 84 | 0.0009 | YES | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR1.5_nfp_days_only` | 74 | 0.2457 | 1.777 | 0.097825 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O15_RR1.5_non_nfp_days` | 1599 | 0.0036 | 0.124 | 0.468165 | 81 | 0.0379 | YES | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR2.0_all_days` | 1673 | 0.0081 | 0.249 | 0.433943 | 84 | -0.0389 | NO | STATISTICALLY_USELESS | 556 | 1117 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O15_RR2.0_nfp_days_only` | 74 | 0.3795 | 2.286 | 0.033998 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O15_RR2.0_non_nfp_days` | 1599 | -0.0091 | -0.272 | 0.433943 | 81 | -0.0033 | YES | STATISTICALLY_USELESS | 530 | 1069 | BALANCED | `FAIL_CHORDIA_STRICT` |
| `O30_RR1.0_all_days` | 1674 | 0.1034 | 4.534 | 0.000042 | 84 | -0.0635 | NO | STATISTICALLY_USELESS | 556 | 1118 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |
| `O30_RR1.0_nfp_days_only` | 74 | 0.3198 | 3.077 | 0.004971 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O30_RR1.0_non_nfp_days` | 1600 | 0.0934 | 4.001 | 0.000178 | 81 | -0.0288 | NO | STATISTICALLY_USELESS | 530 | 1070 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |
| `O30_RR1.5_all_days` | 1674 | 0.1339 | 4.649 | 0.000042 | 84 | -0.1384 | NO | STATISTICALLY_USELESS | 556 | 1118 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |
| `O30_RR1.5_nfp_days_only` | 74 | 0.4488 | 3.283 | 0.003046 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O30_RR1.5_non_nfp_days` | 1600 | 0.1194 | 4.057 | 0.000176 | 81 | -0.1065 | NO | STATISTICALLY_USELESS | 530 | 1070 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |
| `O30_RR2.0_all_days` | 1674 | 0.1488 | 4.401 | 0.000052 | 84 | -0.1088 | NO | STATISTICALLY_USELESS | 556 | 1118 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |
| `O30_RR2.0_nfp_days_only` | 74 | 0.4755 | 2.899 | 0.007410 | 3 | -1.0000 | NO | STATISTICALLY_USELESS | 26 | 48 | EST_THIN | `UNVERIFIED_DST_IMBALANCE` |
| `O30_RR2.0_non_nfp_days` | 1600 | 0.1337 | 3.873 | 0.000252 | 81 | -0.0758 | NO | STATISTICALLY_USELESS | 530 | 1070 | BALANCED | `CONDITIONAL_OOS_UNDERPOWERED` |

## Framings NOT tested by this prereg

The prereg locks one framing: NYSE_PREOPEN MNQ E2 CB1 as a STANDALONE session
trade, partitioned by NFP-day class, with one tail-OOS binary ``dir_match``
gate. That is what the K=27 table above answers. The result does NOT speak to
any of the following adjacent framings, which would require their own prereg:

1. **Overlay / filter on adjacent US-cash lanes.** The 09:00 ET order-imbalance
   prior may condition US_DATA_830 / NYSE_OPEN / US_DATA_1000 lanes (each
   within 90 min of NYSE_PREOPEN). Adjacent sessions have their own larger OOS
   slices, so the same signal viewed at a different lane is not bottlenecked
   by this prereg's 4-month tail-OOS.
2. **Portfolio-level forecast for combining adjacent lanes** (Carver Ch 11
   forecast combination — ``docs/institutional/literature/
   carver_2015_ch11_portfolios.md``). The IS strength on O30 cells
   (t = 3.87-4.65 across 6 cells, BH q < 3e-4 at K=27) is allocator-grade
   evidence; it could weight adjacent-lane allocations even if it does not
   stand alone as a trade signal.
3. **Different OOS framing.** This prereg locks a single tail-OOS binary
   ``dir_match`` gate. Per RULE 3.3 + ``research.oos_power``, the binary gate
   is misspecified when OOS power is below ``CAN_REFUTE``; the 6 held-back
   O30 cells are at ``STATISTICALLY_USELESS``. Alternative OOS pathways
   (Harvey-Liu Sharpe haircut from ``harvey_liu_2015_backtesting.md``; CPCV
   per ``lopez_de_prado_2018_afml_ch_3_7_8.md`` Ch 12) would treat the OOS
   as a discount multiplier or a multi-path resampling rather than a veto.
   Neither is implemented in this repo today.

A 0/27 PASS verdict on this prereg does NOT refute any of the above. It only
says: the standalone-session + tail-OOS + NFP-split framing does not certify
NYSE_PREOPEN for live trading.

## Where the data suggests the edge lives

The headline NFP-split hypothesis is NOT what the data supports. NFP-only
cells are uniformly DST-imbalanced (N_EST = 26 < 30 floor) -- verdict deferred,
not concluded. The unconfounded signal sits at the **O30 aperture across all
RR targets, regardless of NFP status**: 6 cells with t_IS in [3.87, 4.65],
BH q in [4e-5, 2.5e-4] at K=27, all positive ExpR_IS, all DST-balanced. The
short-aperture (O5/O15) cells are real-null (t near zero or negative). This
is consistent with a slow-prior 09:00-ET cash-imbalance regime that needs the
fuller 30-minute aperture to express -- it is NOT consistent with an NFP-day
spillover specifically.

## Method notes

- Canonical source only: ``orb_outcomes`` joined to ``daily_features`` upstream
  (the runner reads ``orb_outcomes`` directly; ``daily_features`` is consulted
  only by ``pipeline.build_daily_features`` when materialising the NYSE_PREOPEN
  rows).
- Sacred holdout boundary: ``trading_day < 2026-01-01`` for
  IS, ``>=`` for descriptive OOS.
- Strict-Chordia threshold: t >= 3.79 (no-theory; prereg ``theory_grant: false``).
- BH-FDR composition at K_family = 27; NaN p-values treated as p=1.0.
- DST imbalance: ``N_EST < 30 OR N_EDT < 30`` -> ``UNVERIFIED_DST_IMBALANCE``
  (verdict deferred, NOT killed) per prereg.
- OOS power floor (RULE 3.3): ``dir_match=False`` only kills when OOS power
  tier == ``CAN_REFUTE``. ``DIRECTIONAL_ONLY`` -> ``CONDITIONAL_OOS_UNDERPOWERED``;
  ``STATISTICALLY_USELESS`` -> ``CONDITIONAL_OOS_UNDERPOWERED``.
- Promotion gate: ``PASS_CHORDIA_STRICT`` requires ALL of t_IS >= 3.79,
  N_IS_on >= 100, ExpR_IS > 0, BH q < 0.05, DST-balanced, NOT killed by OOS.

## Reproduction

```
python scripts/research/emit_nyse_preopen_verdict.py
```

Outputs (refuses to overwrite without ``--force``):

- ``docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.md``
- ``docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.csv``

## Caveats

- This run answers ONLY the prereg's family. It does NOT certify the NYSE_PREOPEN
  session for live trading; the prereg's ``execution_gate.next_phase_blockers``
  remain in force for any post-verdict promotion decision.
- DST-imbalance verdicts (``UNVERIFIED_DST_IMBALANCE``) are NOT death certificates;
  they declare the prereg cannot conclude on the cell with the current EST/EDT
  sample mix and must be re-evaluated once the imbalance closes.
- No write to ``experimental_strategies`` is performed by this script regardless
  of the verdict. The prereg's single-use SHA gate is armed but not pulled here.
