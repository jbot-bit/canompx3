---
task: PR #257 audit-trail corrections — Criterion 13 scratch_policy declaration, sr_review_registry OOS/IS figure correction, Chordia result MD pooled_finding front-matter, Criterion 12 WATCH-continue doctrine amendment, candidate-proposal MD bootstrap-path disclosure, pulse test line bound bump
mode: IMPLEMENTATION
scope_lock:
  - docs/audit/hypotheses/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml
  - trading_app/sr_review_registry.py
  - tests/test_trading_app/test_sr_review_registry.py
  - docs/audit/results/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.md
  - docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.md
  - docs/institutional/pre_registered_criteria.md
  - tests/test_tools/test_pulse_integration.py
  - HANDOFF.md
---

## Blast Radius

- `docs/audit/hypotheses/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml` — adds `scratch_policy:` field (Criterion 13 BINDING for deployment). No runner change required; canonical scratches already realized-eod (verified 2026-05-11 vs gold.db: 69 scratches, 0 NULL pnl_r, mean scratch R = +0.183).
- `trading_app/sr_review_registry.py` — corrects the NYSE_OPEN RR1.5 entry's "OOS/IS 61%" to canonical "112%" (verified vs validated_setups: oos_exp_r/expectancy_r = 0.1179/0.105 = 1.1229). Adds directional-breakdown recheck trigger. Reads-by callers: `trading_app/lifecycle_state.py`, `scripts/tools/refresh_control_state.py`. No new keys, no behavior change beyond corrected figure + extended recheck-trigger string.
- `tests/test_trading_app/test_sr_review_registry.py` — companion test for the new figure (one assertion update).
- `docs/audit/results/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.md` — adds `pooled_finding: true` YAML front-matter + heterogeneity_ack note. Required by `.claude/rules/pooled-finding-rule.md` for any pooled-claim audit MD dated ≥2026-04-20. Drift check `check_pooled_finding_annotations` may enforce.
- `docs/audit/results/2026-05-11-mnq-profile-candidate-proposal.md` — adds one sentence under Verdict + Reproduction sections documenting the `--bootstrap-runtime-control` in-band path that wrote the SR WATCH review.
- `docs/institutional/pre_registered_criteria.md` — adds "Operational extension" subsection under Criterion 12 formalizing the WATCH-continue path (registry as canonical, WFE 0.50 / OOS-IS 0.40 deploy floors repurposed, sibling-selection-pathway disclosure rule).
- `tests/test_tools/test_pulse_integration.py:214` — `len(lines) <= 60` → `<= 65` for the new lane row.
- `HANDOFF.md` — one-line session-baton update post-merge.

Reads: gold.db (read-only via duckdb), validated_setups (read-only). Writes: only the files listed above.

## Truth-state of figures used in this stage (all verified 2026-05-11)

| Claim | Verified figure | Source |
|---|---|---|
| WFE for new lane | 1.7986 → "1.80" | `validated_setups.wfe` |
| expectancy_r | 0.105 | `validated_setups.expectancy_r` |
| oos_exp_r | 0.1179 | `validated_setups.oos_exp_r` |
| OOS/IS ratio | 1.1229 → "112%" | computed from above |
| Sharpe | 0.0871 | `validated_setups.sharpe_ratio` |
| Sample size | 1472 | `validated_setups.sample_size` |
| Scratch count IS | 69 | canonical orb_outcomes |
| Scratch NULL pnl_r | 0 | canonical orb_outcomes |
| Mean scratch R | +0.1827 | canonical orb_outcomes (realized-eod) |
| IS pooled t | 3.600 | result MD line 25 |
| OOS power | 0.088 (STATISTICALLY_USELESS) | result MD line 28 |
| OOS Long ExpR | +0.218 (N=35, t=1.059) | result MD per-direction breakdown |
| OOS Short ExpR | -0.0451 (N=42, t=-0.244) | result MD per-direction breakdown |
| IS Long t / Short t | 2.591 / 2.498 (both fail t≥3.00 unilaterally) | result MD line 35 |

## Order of operations

1. Pre-reg yaml `scratch_policy:` declaration (DONE — verified canonical realized-eod)
2. sr_review_registry.py OOS/IS figure correction + directional recheck (CURRENT)
3. Test update (one assertion: "112%" / "0.1179" / "0.105" appears in summary)
4. Chordia result MD pooled_finding front-matter + heterogeneity_ack note
5. Candidate-proposal MD bootstrap-path disclosure
6. Criterion 12 doctrine amendment
7. Pulse test line bound bump
8. Local pytest + check_drift + audit_behavioral + audit_integrity
9. Commit per the plan's commit-by-commit ordering; push; ready PR #257; merge

## Verification gates

- `pytest tests/test_trading_app/test_sr_review_registry.py tests/test_research/test_mnq_profile_candidate_proposal.py tests/test_trading_app/test_allocation_promotion.py tests/test_tools/test_pulse_integration.py -q`
- `python pipeline/check_drift.py`
- `python scripts/tools/audit_behavioral.py`
- `python scripts/tools/audit_integrity.py`
- Recompute canonical figures via duckdb once more and grep the registry entry to confirm they match.

## Done criteria

- All 7 files in scope_lock edited per the plan; no other file touched.
- All four verification gates green locally.
- CI green on PR #257.
- Lane allocation post-merge on main shows 3 lanes (2 DEPLOY, 1 PROVISIONAL NYSE_OPEN RR1.5).
- Memory golden-nuggets already saved (3 feedback files indexed in MEMORY.md).
