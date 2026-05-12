# Deployed Lanes — Strict Chordia 3.79 Exposure Audit (Path E)

**Date:** 2026-05-12
**Trigger:** `chordia-loader-has-theory-silent-downgrade` debt entry filed during MGC LONDON_METALS Stage 1 K=1 Mode-A revalidation (commit `9e10b62f`).
**Scope:** Read-only exposure audit. No DB mutation, no loader code change, no allocator/pause changes.
**Question:** Were the currently-deployed lanes (`lane_allocation.json:lanes[]`) graded against the strict t≥3.79 threshold their prereg declared, or silently downgraded to t≥3.00?
**Authority:** `docs/institutional/pre_registered_criteria.md` Criterion 4; `trading_app/hypothesis_loader.py:262-269`.

## Live-state inventory (verified 2026-05-12 against `docs/runtime/lane_allocation.json`)

- `lanes[]` = **2** (active deployed)
- `paused[]` = **52**
- `rebalance_date: 2026-05-11`

The 2026-05-03 memory entry citing 3 deployed lanes is stale — `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` was paused 2026-05-12 per a SR-alarm diagnostic. Audited here as informational (4th row in table below) because the loader-bug hypothesis would apply equally.

## Exposure table

| # | Lane | Prereg declared threshold | Prereg `theory_citation` field | Runtime applied | Runtime `has_theory` | t_IS | N_IS_fired | Pass declared? | Capital action |
|---:|---|---:|---|---:|:---:|---:|---:|:---:|---|
| 1 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` (LIVE) | **3.79** (no-theory) | OMITTED on hypotheses[0] | **3.79** | False | **4.256** | 522 | **PASS** | **NO ACTION** — clears strict threshold by t≥0.46 margin |
| 2 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (LIVE) | **3.79** (no-theory) | OMITTED on hypotheses[0] | **3.79** | False | **5.158** | 806 | **PASS** | **NO ACTION** — clears strict threshold by t≥1.37 margin |
| 3 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` (PAUSED 2026-05-12) | **3.00** (theory-backed Chan Ch 7) | `chan_2013_ch7_intraday_momentum.md` | **3.00** | True | **3.600** | 1532 | **PASS** (pooled) | **NO ACTION** — pooled clears theory threshold by t≥0.60 margin; pause is due to SR-alarm + mechanism falsification per `lane_allocation.json:paused[]:reason`, NOT a Chordia issue |

Reference files audited:
- `docs/audit/hypotheses/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.yaml` + `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`
- `docs/audit/hypotheses/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.yaml` + `docs/audit/results/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.md`
- `docs/audit/hypotheses/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml` + `docs/audit/results/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.md`

## Loader behavior — verified

`trading_app/hypothesis_loader.py:262-269`:

```python
# has_theory is True iff at least one hypothesis cites a theory_citation
# field. Used by the Chordia gate to pick the 3.00 (with theory) vs 3.79
# (without theory) threshold per Criterion 4.
has_theory = False
for h in hypotheses:
    if isinstance(h, dict) and h.get("theory_citation"):
        has_theory = True
        break
```

The loader sets `has_theory=True` iff `hypotheses[i].theory_citation` is **any truthy value** for at least one hypothesis. Unambiguous and correctly documented.

## Root cause of the original alarm — RECLASSIFIED

The MGC LONDON_METALS Stage 1 result emitted `MEASURED loader has_theory: True` and applied threshold 3.00 instead of the declared 3.79. I filed this as a loader class-bug to debt-ledger. **That classification is wrong.**

The MGC L_M prereg I authored declared:

```yaml
hypotheses:
  - id: 1
    theory_citation: >
      No filter-mechanism theory citation available for ORB_VOL_8K on
      LONDON_METALS. ...
```

The `theory_citation` field contains prose **saying** there is no theory citation, but the field itself is a non-empty truthy string. The loader correctly interprets the field's *presence* as the signal (per its documented contract) and emits `has_theory=True`. The downgrade to threshold 3.00 was a direct consequence of my prereg authoring error, not a loader silent-downgrade.

The deployed lanes' preregs (`2026-05-02-mnq-comex-ovnrng100-*` and `2026-05-02-mnq-usdata1000-vwapmid-*`) **omit the `theory_citation` field entirely** on their hypotheses — and the loader correctly emits `has_theory=False`. The NYSE_OPEN lane's prereg cites `chan_2013_ch7_intraday_momentum.md` and correctly emits `has_theory=True`.

The loader does what its docstring says. The bug was in my MGC L_M prereg authoring, not in the loader.

## MGC LONDON_METALS Stage 1 verdict — STANDS

t=2.930 fails BOTH the strict t≥3.79 AND the theory-backed t≥3.00 thresholds. The Stage 1 KILL_COHORT verdict (commit `0f6d4a26`) is unaffected by which threshold was applied. The cohort park stands.

## Capital action summary

**ZERO exposure on currently-deployed capital from any Chordia-threshold misapplication.**

Both `lanes[]` entries cleared the strict no-theory hurdle (t≥3.79) under correct loader behavior, with safety margins of t≥0.46 (OVNRNG_100) and t≥1.37 (VWAP_MID_O15). No lane was silently promoted.

## Heterogeneity addendum (informational)

The paused NYSE_OPEN COST_LT12 lane shows a known per-direction heterogeneity (long t=2.591, short t=2.498 both fail t≥3.00 unilaterally; only pooled t=3.600 clears). This is explicitly flagged in `docs/audit/results/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.md` front-matter `heterogeneity_note`. Not in scope for this audit (pause is governing decision), but recorded for completeness.

## Next step

Path F (loader fix) is **NOT WARRANTED on the original premise** (loader silent downgrade). However a related, smaller hardening is justifiable:

- **Drift check candidate (not blocking):** assert that every result MD's `MEASURED threshold applied` matches the prereg's declared `chordia_threshold_basis` numeric value. This would catch authoring errors like my MGC L_M case at commit time rather than at run time. Class of value: low; the run-time emission of `MEASURED loader has_theory:` already exposes the mismatch in the result MD.
- **Documentation update (recommended):** add a sentence to `docs/prompts/prereg-writer-prompt.md` § OUTPUT SCHEMA noting that `hypotheses[i].theory_citation` must be **omitted entirely** for no-theory preregs, NOT set to a "no citation available" prose string. The has_theory flag is field-presence-based.

The original `chordia-loader-has-theory-silent-downgrade` debt-ledger entry (commit `9e10b62f`) is **misclassified** and should be retracted or amended to reflect the corrected analysis above.

## Verdict

**ZERO live-capital exposure** to a Chordia threshold misapplication. Both currently-deployed lanes (`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` and `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`) cleared the strict no-theory hurdle `t≥3.79` by margins of `t≥0.46` and `t≥1.37` respectively, under correct loader behavior. The paused `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` lane carried legitimate theory citation (Chan Ch 7) and cleared the theory-backed `t≥3.00` by `t≥0.60` pooled. The original `chordia-loader-has-theory-silent-downgrade` debt-ledger entry is misclassified — the loader behavior matches its documented contract.

## Reproduction

Read-only audit, no code execution. Cross-references:
- `docs/runtime/lane_allocation.json` — `lanes[]` (2 active deployed) + `paused[]` (1 relevant)
- `docs/audit/hypotheses/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.yaml` + result MD
- `docs/audit/hypotheses/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.yaml` + result MD
- `docs/audit/hypotheses/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml` + result MD
- `trading_app/hypothesis_loader.py:262-269` — `has_theory` field-presence semantics
- `docs/institutional/pre_registered_criteria.md` — Criterion 4 threshold table

## Caveats / limitations

- **Snapshot-in-time, 2026-05-12.** Lane inventory changes on rebalance; re-verify before any future decision touches Chordia thresholds.
- **The loader has_theory contract IS field-presence based.** Authoring a no-theory prereg with a prose `theory_citation:` string is the documented failure mode — addressed via prereg-writer guidance, not a loader change.
- **Heterogeneity addendum on lane 3 is informational.** The paused NYSE_OPEN COST_LT12 lane's per-direction t-stats (long 2.591, short 2.498) both fail t≥3.00 unilaterally; only pooled clears. Pause decision governs; this audit does not re-litigate it.
- **Does NOT cover paused or research-only strategies.** Only `lanes[]` (currently routable to live broker) was audited for capital exposure.
- **Does NOT verify loader behavior at runtime.** Verification is by reading the canonical source + result MD `MEASURED` annotations, not by executing the loader against fresh inputs.
