---
date: 2026-05-12
verdict: 3 REJECTED on Criterion 8 (Mode A OOS)
audit_lock_sha: 82ab4c06
validator_fix_sha: 56f69f51
---

# LLM-drafted prereg trio — audit result

## Scope

Three strategies surfaced by `compute_lane_scores(date.today())` on 2026-05-11
as `chordia_verdict=MISSING` with raw t≥3.79, OOS>0, N≥50. Drafted via
PR #259 LLM hypothesis proposer (`anthropic/claude-opus-4` via OpenRouter),
human-reviewed (date_locked corrected, `orb_minutes` verified against
canonical `validated_setups`), committed at `82ab4c06`, and audited via
the prereg-locked Pathway-A discipline.

| # | strategy_id                                    | t    | N   | ExpR_IS | ExpR_OOS (validated_setups) |
|---|------------------------------------------------|-----:|----:|--------:|----------------------------:|
| 1 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105`    | 4.19 | 339 | 0.196   | 0.177                       |
| 2 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15`| 4.17 | 217 | 0.260   | 0.246                       |
| 3 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15`    | 4.01 | 173 | 0.278   | 0.266                       |

## Audit chain

1. **Drafted:** `python scripts/research/llm_hypothesis_proposer.py --slug … --model anthropic/claude-opus-4 …` (3 invocations).
2. **Human-reviewed:** dates corrected, `orb_minutes` cross-checked against `validated_setups` canonical row (TOKYO_OPEN was O5, not O15 as initially generated).
3. **Locked:** committed at `82ab4c06` (`prereg: lock 3 LLM-drafted hypotheses for chordia-MISSING audit unlock`).
4. **Discovery:** `prereg-loop.sh --execute` ran `strategy_discovery` for each. Phase 4 enforcement passed for all 3; BH FDR at K=1 passed for all 3; rows written to `experimental_strategies`.
5. **Validator-gate parity bug surfaced:** filename pattern mismatch between LLM proposer (`<date>-llm-<slug>.yaml`) and validator's expected glob (`<date>-<instr>-*.yaml`). Fixed in `56f69f51` — validator now matches by `scope.instruments` content via canonical `hypothesis_loader.extract_scope_predicate`.
6. **Validation:** `python -m trading_app.strategy_validator --instrument MNQ --testing-mode individual` ran end-to-end without `--allow-legacy-prereg`. All 3 candidates loaded; all 3 REJECTED.

## Results

| # | strategy_id                                    | OOS_ExpR (Mode A strict) | IS_ExpR (validator) | OOS/IS | N_OOS | Rejection reason                                         |
|---|------------------------------------------------|-------------------------:|--------------------:|-------:|------:|----------------------------------------------------------|
| 1 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105`    | —                        | —                   | —      | 13    | criterion_8: N_oos<30 (Pathway-B no exemption)           |
| 2 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15`| +0.0511                  | +0.2601             | 0.196  | 53    | criterion_8: OOS/IS=0.196 < 0.40 floor                   |
| 3 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15`    | +0.0511                  | +0.2779             | 0.184  | 53    | criterion_8: OOS/IS=0.184 < 0.40 floor                   |

## Verdict

**ALL 3 STRATEGIES REJECTED.** Zero new lanes promoted to `validated_setups`. Live portfolio unchanged. The institutional pipeline (prereg-locked Pathway-A discipline + Mode A holdout enforcement) worked as designed; the candidates failed Criterion 8 honestly.

## Honest framing

**Predicted vs actual:** the pre-validation prediction (3 likely PASS based on
`validated_setups` numbers) was wrong. The error class is documented in
`.claude/rules/research-truth-protocol.md` § "Mode B grandfathered baselines":

> NEVER use `validated_setups.expectancy_r` as a positive-control or
> comparison baseline for a Mode A scan without first recomputing the value
> against strict Mode A IS from canonical `orb_outcomes`.

The `oos_exp_r` values in `validated_setups` (0.177–0.266) for these strategies
were computed under Mode B grandfathered baselines (rows have
`last_trade_day` in the post-2026-01-01 sacred-OOS window). Strict Mode A
recomputation produces OOS=+0.0511 for both CME_PRECLOSE strategies — a
~5x reduction that puts them below the +0.40×IS Criterion 8 floor.

**The institutional pipeline worked exactly as designed.** Three strategies
that looked auditable from the cited stats failed honest Mode A OOS. Zero
new lanes promoted to `validated_setups`; live portfolio unchanged
(3 deployed lanes per MEMORY.md `live_lanes_2026_05_03_three_alarm_one_pause.md`).

## What this implies for the wider 58-MISSING surface

The 58 lanes meeting raw `t≥3.79 + OOS>0 + N≥50` from yesterday's session
ranking were sorted by Mode-B-contaminated `oos_exp_r`. That ranking cannot
be trusted for prereg targeting. Before drafting more preregs in this
direction, the candidate-ranking query needs to be re-derived against
strict Mode A `orb_outcomes JOIN daily_features` per the rule, not against
`validated_setups`.

This re-derivation is filed as a separate follow-up; it is not within the
scope of this audit.

## Carry-forwards

- **Stage 2 (drift Check 145 for prereg/validator parity):** filed as P3
  follow-up per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`. n=1 failure;
  Stage 1 closes the runtime risk; preemptive meta-tooling deferred.
- **Stage 4.4 (adversarial audit):** required by
  `.claude/rules/adversarial-audit-gate.md` for the truth-layer change in
  `56f69f51`. Dispatched in this session via `evidence-auditor` subagent.
- **Mode B baseline contamination:** the entire 58-lane MISSING ranking
  produced 2026-05-11 should be discarded as a target list. Re-derive under
  strict Mode A before next prereg cycle.

## Limitations

- **Mode B baseline contamination in candidate ranking.** The 58 lanes meeting raw `t≥3.79 + OOS>0 + N≥50` from the 2026-05-11 session ranking were sorted by `validated_setups.oos_exp_r`, which for rows with `last_trade_day` post-2026-01-01 is computed under Mode B grandfathered baselines. Strict Mode A recomputation produced ~5x lower OOS for the 2 CME_PRECLOSE candidates here (Mode-B 0.246/0.266 → Mode-A 0.0511 each). The wider 58-lane ranking is not trustworthy for prereg targeting until re-derived under strict Mode A from canonical `orb_outcomes JOIN daily_features`.
- **Single-instrument scope only.** All 3 preregs are MNQ; MES and MGC have separate Criterion 1 violations from a prior session that this audit does not address.
- **N=13 OOS for TOKYO_OPEN** is intrinsic to the lane's ORB_5 firing rate during the 2026 holdout window and cannot be remediated without waiting for more OOS data or adopting CPCV (López de Prado 2020) — outside this audit's scope.
- **Pre-validation prediction was wrong** — pre-audit text predicted 3/3 PASS based on Mode-B-rooted numbers. This is a documented failure mode (`research-truth-protocol.md` § "Mode B grandfathered baselines") and an instance of the rule firing exactly as intended.

## Reproduction

```bash
# Discovery (idempotent — re-runs deduplicate against existing rows):
for slug in tokyo-open-atr-vel-ge105 cme-preclose-orb-vol-16k-o15 cme-preclose-atr-p30-o15; do
  CANOMPX3_PYTHON=.venv/Scripts/python bash scripts/infra/prereg-loop.sh \
    --hypothesis-file docs/audit/hypotheses/2026-05-11-llm-${slug}.yaml --execute
done

# Validation:
.venv/Scripts/python -m trading_app.strategy_validator \
  --instrument MNQ --testing-mode individual

# Read rejection reasons:
.venv/Scripts/python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
print(con.execute('''
  SELECT strategy_id, validation_status, rejection_reason
  FROM experimental_strategies
  WHERE strategy_id IN (
    \"MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105\",
    \"MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15\",
    \"MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15\")
''').fetchdf().to_string(index=False))
"
```

## Audit-trail commits

| commit     | role                                                    |
|------------|---------------------------------------------------------|
| `aee9daba` | LLM proposer hardening (markdown-fence strip, 4xx body) |
| `82ab4c06` | Lock the 3 prereg yamls (audit lock SHA)                |
| `56f69f51` | Validator content-aware match + 7 companion tests       |
| `cecec1db` | Unrelated: AccountProfile dead-field removal            |
