# Decision Ledger

Use this file for durable accepted decisions that should survive handoff churn.

## Current

- `runtime-shell-unification` — Startup orientation should consume one derived `system_brief` instead of parallel ad hoc summaries.
- `capsule-is-task-owner` — Active scoped work should carry a single work capsule with route, scope, and verification obligations.
- `history-split` — `HANDOFF.md` is current baton only; durable decisions and debt belong in dedicated ledgers.
- `pr48-conditional-edge-recovered` — The unfinished PR48 conditional-edge / multi-confluence line has been recovered from `origin/wt-codex-conditional-edge-framework`, replayed cleanly onto published `origin/main`, and re-run against canonical layers. Current truth is `CONTINUE as conditional-role / allocator shortlist`, not fresh discovery and not live-ready promotion. The next honest move is the bounded translation stage in `docs/runtime/stages/pr48-conditional-role-validation-translation.md`, not another generic confluence scan.
- `orb-cap-key-shape` — ORB-cap registry identity is `(orb_label, instrument)`, not session-only. Multi-instrument sessions must preserve per-instrument caps; callers must not collapse them back to a session key.
- `pr48-sizer-split` — PR48 frozen rel-vol sizing is no longer a pooled MES/MGC story. `MGC` is the only research-level deploy candidate from the 2026-04-23 frozen-rule replay; `MES` improved but remains `SIZER_ALIVE_NOT_READY` and must not be promoted alongside MGC.
- `mgc-payoff-compression-scope` — The 5-minute `GC -> MGC` translation gap is actioned at the diagnostic stage: `PAYOFF_COMPRESSION_REAL` and `LOW_RR_RESCUE_PLAUSIBLE` both hold, but the rescue signal is broader `MGC` target-shape behavior, not a narrow proxy-only revival. Future follow-up must be a narrow MGC exit-shape prereg, not broad GC proxy reopening.
- `l1-europe-flow-pre-break-kill` — The restored L1 EUROPE_FLOW pre-break-context prereg was executed honestly on 2026-04-23 and closed `KILL`. `pre_velocity_HIGH_Q3` never cleared raw significance, and `rel_vol_HIGH_Q3` showed IS lift but failed `K=2` BH-FDR and flipped OOS sign. Do not reopen banned break-bar or ATR-normalized replacement variants from this path.
- `nyse-close-followup-orbg8-next` — The 2026-04-23 MNQ NYSE_CLOSE RR1.0 follow-up closed `CONTINUE`, not `PARK` or `KILL`. Repo truth says the family is alive-but-underexecuted: broad RR1.0 stays positive on O5/O15/O30, only three narrow O5 RR1.0 filters were actually tried, and the exact native `ORB_G8` RR1.0 path was locked twice but never executed. The direct next move is the frozen `MNQ NYSE_CLOSE ORB_G8 RR1.0` prereg, not a raw portfolio unblock and not another broad sweep.
