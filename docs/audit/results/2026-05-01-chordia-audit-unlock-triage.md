# Chordia Audit Unlock — Triage (2026-05-01)

**Status:** ORIENT step complete. No audits run. No pre-regs written.
**Next decision:** user choice on per-strategy theory-grant feasibility scan.
**Action queue item:** `chordia_audit_unlock_pass_chordia_strategies` (P1, open).
**Authority:** `docs/runtime/chordia_audit_log.yaml` doctrine + `docs/institutional/pre_registered_criteria.md` Criterion 1, 3, 4.

## Scope

**What this triage tests:** Which of the 8 PASS_CHORDIA-without-audit strategies surfaced by `compute_lane_scores` are even auditable as PASS_PROTOCOL_A (theory-granted) vs need to clear strict t≥3.79 standalone (PASS_CHORDIA), and which are likely correlated and should NOT be audited as independent.

**What this triage does NOT do:** It does NOT pre-register any hypothesis. It does NOT run any Pathway-A revalidation. It does NOT make a theory-grant decision. It does NOT modify `chordia_audit_log.yaml`. Those are downstream user decisions.

---

## Live ground truth (queried 2026-05-01 via `compute_lane_scores` + `validated_setups`)

8/59 strategies are PASS_CHORDIA-without-audit. All clear t≥3.79.

| # | strategy_id                                            | t    | N    | ExpR (IS) | ExpR (OOS) | RR  | filter                |
|---|--------------------------------------------------------|-----:|-----:|----------:|-----------:|----:|-----------------------|
| 1 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`   | 4.58 |  701 |     0.210 |      0.207 | 1.5 | VWAP_MID_ALIGNED      |
| 2 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`            | 4.57 |  596 |     0.170 |      0.160 | 1.0 | X_MES_ATR60           |
| 3 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100`             | 4.32 |  520 |     0.173 |      0.149 | 1.0 | OVNRNG_100            |
| 4 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`            | 4.32 |  673 |     0.151 |      0.163 | 1.0 | X_MES_ATR60           |
| 5 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`   | 4.27 |  744 |     0.149 |      0.151 | 1.0 | VWAP_MID_ALIGNED      |
| 6 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12`              | 4.26 | 1247 |     0.110 |      0.103 | 1.0 | COST_LT12             |
| 7 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`             | 4.15 |  513 |     0.215 |      0.203 | 1.5 | OVNRNG_100            |
| 8 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075`         | 3.82 |  380 |     0.157 |      0.113 | 1.0 | COST_LT10 + S075      |

All 8 have `validation_pathway = family` → BHY FDR correction required per Criterion 3, not raw p<0.05. OOS ExpR is close to IS for all 8 — no obvious decay.

---

## Triage tiers

### Tier A — likely-correlated pairs (audit ONE per pair, not both)

- VWAP_MID_ALIGNED_O15 RR1.0 vs RR1.5 (#1 & #5) — same setup, different RR
- COMEX_SETTLE OVNRNG_100 RR1.0 vs RR1.5 (#3 & #7) — same setup, different RR
- X_MES_ATR60 in COMEX_SETTLE (#4) and CME_PRECLOSE (#2) — same filter, different sessions; less correlated, can audit both

### Tier B — needs theory-grant decision

`chordia_audit_log.yaml` doctrine: `has_theory=True` requires citation in `docs/institutional/literature/`. Without theory, strategy must clear strict t≥3.79 standalone (PASS_CHORDIA, not PASS_PROTOCOL_A).

| Filter class      | Likely theory ground (read to confirm)              | Notes |
|-------------------|------------------------------------------------------|-------|
| VWAP_MID_ALIGNED  | Carver Ch 9-10? Fitschen Ch 3? — UNVERIFIED         | Mechanism: VWAP-anchor as fair-value reference for breakout direction |
| OVNRNG_100        | Fitschen Ch 3 (likely) — UNVERIFIED                 | Mechanism: prior overnight range as session-context regime tag |
| X_MES_ATR60       | UNSUPPORTED in current literature/                  | Cross-asset MES vol — no obvious local citation |
| COST_LT12         | NOT a mechanism — execution-friction floor         | Cannot earn theory grant; needs t≥3.79 standalone (currently 4.26 ✓) |

### Tier C — barely clears

- #8 (MES CME_PRECLOSE COST_LT10 S075) at t=3.82 with +0.113R OOS. Marginal. Audit cost ≈ same; reward smaller. Defer until A/B audited.

---

## Honest framing on `+0.4R+` user goal

Highest single-strategy OOS ExpR here is **+0.207R** (VWAP_MID_ALIGNED_O15 RR1.5). Not +0.4R. A blended portfolio of 4-6 of these at proper weights could plausibly land **+0.10–0.18R after costs/correlation drag**. **+0.4R is not a single-lane target** — it would need correlated-but-additive lane stacking or a structurally new mechanism.

---

## What an audit run requires (per institutional rigor)

For EACH strategy retained from the triage:

1. **Pre-registered hypothesis file** at `docs/audit/hypotheses/2026-MM-DD-<slug>.yaml` per Criterion 1 (numbered hypotheses, theory citation per hypothesis, filter columns + thresholds + sessions + instruments + RR explicit, kill criteria).
2. **Pathway-A adversarial-split revalidation** per Criterion 3 (NOT a sweep — single-strategy retest with split discipline).
3. **Theory-grant decision** — append to `docs/runtime/chordia_audit_log.yaml` `theory_grants:` row with required `theory_ref` to a file in `docs/institutional/literature/` (or set `has_theory=False` and rely on strict t≥3.79).
4. **Audit entry** under `audits:` with `audit_date`, `verdict`, `t_stat`, `threshold`, `sample_size`. PASS only.

This is **per-strategy multi-day research**, not a single PR.

---

## Verdict

**ORIENT-ONLY VERDICT:** 8 strategies confirmed PASS_CHORDIA-without-audit. 3 of 8 are likely-correlated pairs (Tier A) — collapse to ~5-6 distinct audits. 1 of 8 is a marginal candidate (Tier C — defer). Theory-grant feasibility for the 4 distinct filter classes (VWAP_MID / OVNRNG / X_MES_ATR / COST_LT) is UNVERIFIED until literature extracts are read. Action queue item REMAINS OPEN. No deploy implications until per-strategy audits land.

## Reproduction

Live ground truth was queried fresh (volatile-data rule, never cite from memory):

```python
from datetime import date
from trading_app.lane_allocator import compute_lane_scores
scores = compute_lane_scores(date.today())
targets = [s.strategy_id for s in scores
           if s.chordia_verdict == "PASS_CHORDIA"
           and s.chordia_audit_age_days is None]
# 8 strategies returned; full list in the table above.
```

t-stats / N / ExpR pulled from `validated_setups` directly (the `compute_lane_scores` LaneScore object does not carry these fields):

```python
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute("""
    SELECT strategy_id, sample_size,
           round(sharpe_ratio,3) AS sharpe,
           round(sharpe_ratio * sqrt(sample_size),2) AS t_stat,
           round(expectancy_r,3) AS expR,
           round(oos_exp_r,3) AS oos_expR,
           round(p_value,4) AS p_val,
           validation_pathway
    FROM validated_setups
    WHERE strategy_id IN (?, ?, ?, ?, ?, ?, ?, ?)
    ORDER BY t_stat DESC
""", targets).fetchdf()
```

Outputs / artefacts produced by this orientation:
- This doc: `docs/audit/results/2026-05-01-chordia-audit-unlock-triage.md`
- Memory entry: `memory/chordia_audit_unlock_triage_2026_05_01.md` (auto-memory dir)
- HANDOFF.md "2026-05-01 EVE" section

No code committed, no DB writes, no pre-reg files, no chordia_audit_log entries.

## Limitations

- **Theory-grant feasibility per filter class is UNVERIFIED.** The Tier B matrix in this doc lists "likely supports" / "UNSUPPORTED" guesses based on filter-class semantics, NOT on actual reads of `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` or `docs/institutional/literature/carver_2015_systematic_trading_ch9_10.md`. Those reads are the recommended next step.
- **Live data is a snapshot.** `compute_lane_scores(date.today())` will produce a different list as new strategies are validated or as the Chordia gate's freshness threshold (90 days) expires existing audits. The 8-strategy count is true at 2026-05-01; re-query before acting.
- **`validation_pathway = family` for all 8** means BHY FDR correction is required per Criterion 3, not raw p<0.05. Even strategies with t=4.58 may not survive a fresh BHY pass — depends on the family K used at original discovery time.
- **OOS ≈ IS does not prove no decay.** The OOS window for these strategies is the post-validation forward period, which for `family`-pathway strategies discovered before 2026-04-08 includes Mode-B-grandfathered data. Per `research-truth-protocol.md` § "Mode B grandfathered baselines", `validated_setups.expectancy_r` may need re-derivation under strict Mode A.
- **Correlation pruning in Tier A is structural, not measured.** The "RR1.0 vs RR1.5 same setup" pairs are obviously correlated by construction; the `X_MES_ATR60` cross-session pair is "less correlated" by inspection only. Actual Pearson r between trade returns is not computed in this doc.
- **+0.4R framing is an estimate.** "Blended portfolio plausibly +0.10-0.18R" is an order-of-magnitude expectation based on single-strategy ExpR + typical correlation drag for similar lanes. No portfolio bootstrap was run.

## Recommended first step (NOT yet executed)

**Theory-grant feasibility scan** — read Fitschen Ch 3 and Carver Ch 9-10 extracts in `docs/institutional/literature/` to determine which of the 4 filter classes (VWAP_MID, OVNRNG, X_MES_ATR, COST_LT) have citable theory grounds. One read pass, one memo, no code, no commits. Output: a `feasibility-scan.md` annex to this triage doc indicating which strategies are PASS_PROTOCOL_A-eligible vs need-strict-t-threshold.

This step is the cheapest step that informs everything else and directly satisfies the "literature grounding BEFORE writing a prereg" rule (`memory/feedback_literature_before_prereg.md`).

---

## Why this isn't a single-PR ship

- 8 strategies × per-strategy pre-reg + audit + theory grant = 8 separate research artifacts
- Tier A correlation pruning can collapse to ~5-6 distinct audits
- Each audit takes ~half-day with proper theory grant + adversarial split
- Total: ~3-4 days of focused research work

Action queue item should remain `open` until the feasibility scan tells us which strategies are auditable at all.
