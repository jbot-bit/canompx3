# H0 Adaptive-Stops — Independent Audit + NEED-EVIDENCE Resolution

- **Date:** 2026-05-31
- **Worktree:** `canompx3-adaptive-stops-audit` (branch `session/joshd-adaptive-stops-audit`, off `origin/main` 7782a573)
- **Audited artifact:** H0 MFE/MAE-symmetry diagnostic, committed `4db9b5dd` in sibling worktree `canompx3-adaptive-stops-h0` (NOT pushed)
  - Results: `docs/audit/results/2026-05-31-adaptive-stops-h0-symmetry-diagnostic-results.md`
  - Raw CSV: `docs/audit/results/2026-05-31-adaptive-stops-h0-raw-per-lane.csv`
  - Pre-reg (LOCKED): `docs/audit/hypotheses/2026-05-31-adaptive-stops-h0-mfe-mae-symmetry-diagnostic-v1.yaml`
  - Script: `research/adaptive_stops_h0_symmetry_diagnostic.py`
- **Auditor:** independent `research-methodologist` (fresh context, did not author H0). Verdict reproduced verbatim in § Independent audit verdict below.
- **Resolution evidence:** this session — per-year N breakdown (from script stdout) + live `get_strategy_fitness` query. No strategy logic changed; no `experimental_strategies`/`validated_setups` write.

---

## Scope

Independent adversarial audit of the H0 adaptive-stops MFE/MAE-symmetry diagnostic, and resolution of the one NEED-EVIDENCE item it raised (`MES_CME_PRECLOSE` E2 year-stability). Question: does the broad fixed tighter price-stop multiplier survive, and if a narrow branch survives, is it the highest-EV next action vs the untested H4 time-exit?

## Why this audit ran

The original H0 doc issued a single clean PROCEED (`MES_CME_PRECLOSE` E2) and marked the price-stop family PRE-KILL on 4/7 deployed E2 lanes. An independent adversary was dispatched to falsify rather than confirm. The auditor's single most important challenge:

> The `MES_CME_PRECLOSE` "clean PROCEED" rests on **N_IS=289, N_OOS=12, OOS delta NEGATIVE (-0.0327)**. "year-stability CONSISTENT" may be an artifact of too few powered years (N≥30 floor) to disagree — the per-year N breakdown is **not in the committed CSV**, so CONSISTENT was unverifiable as written.

Resolution gate (operator-set): **if <3 powered years (N≥30) OR fitness=DECAY → mark H1/H3 PARK/KILL, redirect trial budget to H4.**

---

## Resolution evidence (this session)

### 1. Per-year N breakdown — `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` [E2]

Source: `research/adaptive_stops_h0_symmetry_diagnostic.py --instrument MES --allow-draft` **stdout** (the per-year `{year: ev_delta(n)}` is printed but collapsed to a flag in the CSV — no logic change needed to extract it). `YEAR_STABILITY_MIN_N = 30`. Per-year EV-delta evaluated at the pooled best multiplier (m=0.5):

| Year | N | EV-delta @m=0.5 | Powered (N≥30)? | Sign |
|---|---|---|---|---|
| 2019 | 2  | -0.6719 | ✗ excluded | (neg, N=2 noise) |
| 2020 | 60 | +0.0291 | ✓ | + |
| 2021 | 25 | +0.0661 | ✗ excluded | + |
| 2022 | 87 | +0.0255 | ✓ | + |
| 2023 | 12 | +0.2500 | ✗ excluded | + |
| 2024 | 33 | +0.1844 | ✓ | + |
| 2025 | 70 | +0.1428 | ✓ | + |

**Powered years = 4 (2020, 2022, 2024, 2025), all positive.** The N≥30 floor correctly excludes the 2019 N=2 outlier that would otherwise dominate the sign.

**→ Auditor's worry FALSIFIED.** CONSISTENT does NOT rest on ≤2 powered years; it rests on 4, all positive. Gate condition "<3 powered years" is **NOT met** (4 ≥ 3).

### 2. Live fitness — `get_strategy_fitness` (gold-db MCP, this session)

```
strategy_id        : MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8
fitness_status     : FIT
fitness_notes      : "Positive rolling ExpR with stable recent Sharpe"
full_period_exp_r  : 0.1729   full_period_sample : 287
rolling_exp_r      : 0.0687   rolling_sample     : 86   rolling_window_months : 18
rolling_win_rate   : 0.5698
recent_sharpe_30   : -0.0016  sharpe_delta_30 : -0.192
recent_sharpe_60   : +0.0355  sharpe_delta_60 : -0.155
```

**→ FIT, not DECAY.** Gate condition "fitness=DECAY" is **NOT met.** (Note for the record: `sharpe_delta_30/60` are mildly negative — a soft watch signal, not a DECAY classification. Does not trip the gate.)

### Gate outcome

Neither redirect trigger fired (4 powered years ≥ 3; fitness FIT ≠ DECAY). **H1/H3 for `MES_CME_PRECLOSE` E2 is NOT PARK/KILL.** The lane survives the falsification gate honestly.

---

## Reproduction

All read-only against canonical `gold.db`; no logic changed. From worktree `canompx3-adaptive-stops-h0`:

```
# Per-year N breakdown (printed to stdout, collapsed to a flag in the CSV):
python research/adaptive_stops_h0_symmetry_diagnostic.py --instrument MES --allow-draft
#   → MES_CME_PRECLOSE [E2]: per-year EV-delta @best-m line; year-stability: CONSISTENT
#   (--allow-draft required: the script's draft-gate is a hardcoded flag check at L516,
#    NOT a promotion-state inspection; the pre-reg is in fact LOCKED+committed 4db9b5dd.)

# Live fitness (gold-db MCP):
get_strategy_fitness(instrument="MES", strategy_id="MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8")
#   → fitness_status: FIT; rolling_exp_r: 0.0687; rolling_win_rate: 0.5698
```

Outputs captured under `docs/audit/results/evidence/`:
- `2026-05-31-mes-preclose-per-year-stdout.txt` — full diagnostic stdout (per-year N at L28)
- `2026-05-31-mes-preclose-detail.csv` — MES per-stratum rows

## But the verdict is PROCEED-**but-DEFER**, not PROCEED-now — budget dominates

Surviving the gate ≠ being the highest-EV next action. The auditor's strongest finding stands and is confirmed against the **canonical pre-reg** (`...v1.yaml:208-224`, live-verified via research-catalog `estimate_k_budget` 2026-05-31):

- Downstream family K=21 (H1≤6 + H2≤8 + H3≤6 + **H4=1**).
- MinBTL requires **6.09yr**; available clean-data headroom **0.56yr** — the family is **already near the MinBTL ceiling**.
- **H1+H3 for this one lane cost up to 12 trials; H4 costs 1.** H4 (time-based no-progress exit) applies to **all 7 lanes**, is a different mechanism unaffected by every H0 verdict, and carries the **stronger external evidence**: Howard 2026 §5.4 time-stack beat price-stops **11/13 months (sign-test p=0.02)** (`docs/institutional/literature/howard_2026_value_area_breakouts_es.md`).

Portfolio EV check (auditor, back-of-envelope — flagged as such): +0.081R on 289 MES trades at MES $5/pt ≈ negligible book impact vs. the cost of spending ~12/21 of the MinBTL budget on one thin lane.

**Conclusion:** MES_CME_PRECLOSE H1/H3 is a *legitimate* future draft (it survived), but it is **not budget-rational to draft first**. H4 is the budget-rational first branch.

---

## Other audit findings carried forward (not gated this session, recorded for the H1/H3 drafter)

1. **EV gate has no materiality floor.** `MNQ_EUROPE_FLOW` E2 issues PROCEED on a **+0.0034R** margin (~1 tick tighter than baseline, no t-test) — indistinguishable from noise on N=1583. It should be treated as effective PRE-KILL; leaving H1/H3 "LIVE" for it would waste trial budget. (CSV row 18; gate is `best_tighter_ev > baseline` with no min delta, script ~L377.)
2. **Post-hoc multiplier selection** (argmax over 6 tighter multipliers on IS) is load-bearing only for the two SIGN_INCONSISTENT PROCEEDs (`US_DATA_1000`, `EUROPE_FLOW`); harmless on PRE-KILL lanes. Year-stability is annotation-only and does not flip the gate — correct for K=0, but means those two PROCEEDs should NOT be drafted without first demonstrating year-stability.
3. **OOS sign flip on MES_CME_PRECLOSE E2** (IS +0.064 → OOS -0.033, N_OOS=12) is correctly STATISTICALLY_USELESS per RULE 3.3 (cannot refute at this power) but is the highest-risk single number in the dataset and was buried in a CSV column. It does not block H1/H3 but the drafter must foreground it.
4. **Howard citation is legitimate** — symmetry→PRE-KILL mechanism supported (extract lines 88, 94); the ES-specific 75% breakeven is correctly NOT used as a gating constant.

---

## Verdict

| Item | Verdict | Basis |
|---|---|---|
| Broad fixed tighter price-stop multiplier (pooled) | **PARK / largely DEAD** | 4/7 deployed E2 lanes PRE-KILL (symmetric MFE/MAE); 2 SIGN_INCONSISTENT PROCEEDs are noise |
| `MES_CME_PRECLOSE` E2 H1/H3 (narrow branch) | **PROCEED-but-DEFER** (survives gate; NOT KILL) | 4 powered positive years + FIT; but ~12/21 trial cost vs H4's 1 |
| `MNQ_US_DATA_1000` / `MNQ_EUROPE_FLOW` E2 | **NEED-EVIDENCE** (effective PRE-KILL until year-stable) | SIGN_INCONSISTENT; EUROPE_FLOW margin +0.0034R = noise |
| H4 (time-based / no-progress exit) | **the live branch** | K=1, all lanes, Howard 11/13 months p=0.02; unaffected by H0 |

**The broad idea is largely dead. The narrow price-stop branch (MES_CME_PRECLOSE) is alive but should not be drafted first. The genuinely highest-EV alive branch is H4.**

---

## Highest-EV next action (single, executable)

Pre-register **H4 (time-based no-progress exit)** before drafting MES_CME_PRECLOSE H1/H3. H4 is K=1, applies to all 7 lanes, has the strongest external grounding, and does not consume the price-stop trial budget. It is the budget-rational first branch given 0.56yr MinBTL headroom.

Concrete:
```
python scripts/tools/propose_hypothesis.py \
  --slug 2026-06-01-h4-time-based-no-progress-exit \
  --grounding docs/institutional/literature/howard_2026_value_area_breakouts_es.md
```
(or invoke the `propose-hypothesis` skill) — then human-review the `.draft.yaml`, citing Howard §5.4 (time-stack beat price-stops 11/13 months, sign-test p=0.02) and the H0 gate result that H4 is UNAFFECTED by every E2 price-stop verdict. Do NOT draft MES_CME_PRECLOSE H1/H3 until H4 is registered and the EUROPE_FLOW/US_DATA_1000 PROCEEDs are demonstrated year-stable.

**Forbidden:** parameter fishing, OOS tuning, post-hoc threshold rescue, drafting H1/H3 for the two regime-fragile PROCEEDs without year-stability proof.

---

## Limitations

- **No logic changed.** The per-year breakdown was extracted from existing script **stdout** (already printed at L468-474); the CSV merely collapses it to a flag. `--allow-draft` was required only because the script's draft-gate is a hardcoded `if not args.allow_draft` (L516) that never inspects promotion state — the pre-reg is in fact LOCKED + committed (`4db9b5dd`). Run was `read_only=True` against canonical `gold.db`.
- **Numbers verified, not claimed:** per-year table read directly from `mes_stdout.txt` L28; fitness read directly from the `get_strategy_fitness` MCP return; K-budget read from pre-reg `...v1.yaml:208-224`.
- **Auditor's own claims held to the same standard:** the auditor's `--instrument MES --csv` invocation was verified to exist (script L509-511) before running; its "$0.40/trade" figure is explicitly a back-of-envelope and is labeled as such above.
- **What's still MISSING / residual risk:** the OOS slice (N_OOS=12) cannot confirm the IS effect for MES_CME_PRECLOSE — this is inherent to the <5-month-old sacred holdout, not a defect, and resolves only with time. `sharpe_delta_30/60` mildly negative = soft watch, monitor on next fitness pass.
