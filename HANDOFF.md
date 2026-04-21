# HANDOFF.md — Cross-Tool Session Baton

## Update (2026-04-21 autonomous — ML clean-room reset path established in isolated worktree)

Work stayed isolated in `/tmp/canompx3-ml-sizing-v1` on branch `research/ml-sizing-v1`.
No changes were made to the dirty shared checkout on
`research/pr48-sizer-rule-oos-backtest`.

### What was established

- The meta-label scaffold now targets a dense positive family from canonical data:
  - `MNQ TOKYO_OPEN E2 RR1.5 CB1 O5 LONG-ONLY NO_FILTER`
- Dataset stats from canonical `daily_features` + `orb_outcomes`:
  - pre-2026 rows: `859`
  - wins: `421`
  - losses: `438`
  - pre-2026 ExpR: `+0.071158R`
  - forward `2026+` rows: `36`
- Scratches are excluded from the binary target.
- `2026+` is explicitly marked forward-monitor only and too thin for ML sign-off.

### New artifacts in the isolated worktree

- `trading_app/meta_labeling/feature_contract.py`
- `trading_app/meta_labeling/dataset.py`
- `docs/audit/hypotheses/2026-04-21-meta-label-sizing-v1.yaml`
- `docs/plans/2026-04-21-ml-clean-room-reset-plan.md`

### Key research decision

Do **not** treat this as "retrain the old ML."

The repo evidence now supports a clean-room reset:

- prior ML lineage already failed pooled meta-labeling and exposed negative-baseline / selection / execution risks
- current family is a valid substrate for testing, but not proof that ML helps
- highest-EV next step is a strictly pre-2026 bakeoff:
  - static baseline
  - simple monotonic allocator
  - RF allocator
  - HGB allocator

If simple ties or beats ML, ML should be declared unnecessary for this family.
