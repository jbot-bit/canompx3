# Qwen Overlay-Claims Re-Audit — Ground-Truth Verdict Table

**Date:** 2026-06-11
**Type:** read-only re-audit (no capital, no schema, no candidate validation)
**Trigger:** two externally-generated ("Qwen, zero-hallucination canonical")
overlay pastes were submitted for evaluation. The pastes claimed to prevent
hallucination but fabricated ~40% of their specific factual claims (the exact
class they claimed to prevent) and propagated a STALE "awaiting OOS" framing on a
candidate that is in fact already deployed and OOS-consumed.

**Purpose:** record the TRUE current status of every overlay claim Qwen named, so
the next session (or the next external paste) cannot re-chase a deployed / dead /
consumed / fabricated item. Companion corrected prompt:
`docs/governance/ai_session_canonical_prompt.md`.

**Method:** each claim cross-checked against (a) the dated `docs/audit/results/`
verdict doc, (b) `trading_app/prop_profiles.py` (is it a live lane?), (c) the
live `strategy-lab` lane allocator, NOT against the `docs/STRATEGY_BLUEPRINT.md`
summary line (which goes stale — see `STRATEGY_BLUEPRINT.md:441`). Every number
below is either re-derived live or cited to a result doc; **no Qwen number is
trusted.**

---

## Scope / question

**Question:** which of the externally-generated (Qwen) overlay claims are honest,
verified, and worth acting on — and which are fabricated, stale, or already
resolved? **Scope:** the specific factual claims in the two pastes (VWAP lanes,
NR7, gap filter, MGC US_DATA_830, prior-day-loss CME_REOPEN, DB engine, column
names, the no-CVD argument, the process discipline). Out of scope: re-running any
scan (the only named live candidate is OOS-consumed and locked) and any code change.

## Reproduction / outputs

Every status below was produced read-only from:
- `grep`/`Read` over `docs/STRATEGY_BLUEPRINT.md`, `trading_app/prop_profiles.py`,
  `trading_app/config.py`, `pipeline/init_db.py`, `trading_app/holdout_policy.py`,
  `research/vwap_comprehensive_family_scan.py`.
- Live `strategy-lab` MCP `get_lane_allocation_summary(topstep_50k_mnq_auto)` —
  returned `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` `status: DEPLOY`
  (rebalance 2026-05-30, staleness OK) on 2026-06-11.
- `gold-db` MCP `get_strategy_fitness` was ATTEMPTED but FAILED (`Table
  validated_setups does not exist`) — see Caveats; deployment truth taken from
  `prop_profiles.py` + the allocator instead.
- File-existence checks confirming every cited result-doc / pre-reg path resolves
  (and catching one that does not — the blueprint:301 citation, row 2).

## Verdict table

| # | Qwen claim | Qwen's number | Reality | Proof (file:line / result-doc) | Verdict |
|---|---|---|---|---|---|
| 1 | VWAP MNQ US_DATA_1000 overlay "awaiting OOS / candidate to validate" | "pending OOS" | **Already a LIVE deployed lane AND its OOS holdout is already consumed (one-shot, locked 2026-04-18).** No OOS gate remains; no pre-reg to write. | Lane `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` `active=True` ExpR +0.210 N=701 — `prop_profiles.py:704`, `:804`. Allocator `status: DEPLOY` (rebalance 2026-05-30, staleness OK) — live `strategy-lab` query 2026-06-11. OOS window `[2026-01-01, 2026-04-18)` consumed — `docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md` (run 2026-04-18T08:50Z, 333s). One-shot lock — `research/vwap_comprehensive_family_scan.py:90-97`. | **ALREADY-LIVE + OOS-CONSUMED** |
| 2 | VWAP "MNQ CME_PRECLOSE O5" cited as live/usable | "live" | **DEAD since 2026-04-18** — OOS direction reversal under Mode A. Was PENDING (28 OOS days), never deployed. | `STRATEGY_BLUEPRINT.md:301` (DEAD row: IS +0.167/+0.168/+0.162 t=5.35/3.93/3.04, OOS −0.156/−0.354/−0.496, dir_match FAIL). `config.py:3389` ("CME_PRECLOSE O5 = 28 OOS days (PENDING)"). Pre-reg `docs/audit/hypotheses/2026-04-13-mnq-vwap-cme-preclose.yaml` (present). Independently confirmed in the consumed scan: `2026-04-18-vwap-comprehensive-family-scan.md:102,167` (CME_PRECLOSE O5 RR1.5 |t|=2.66, fails boot_p + BH-FDR K_family). ⚠️ NOTE: the result-doc path the blueprint:301 row cites (`…2026-04-18-vwap-bp-cmepreclose-c8-recheck.md`) **does NOT exist on disk** — a stale citation in the blueprint itself (the same stale-path trap this audit documents). | **STALE (DEAD-since) + blueprint cites a dead path** |
| 3 | NR7 "fire rate 33%" (session-specific overlay) | 33% | Standard NR7 fires 33%, but **session-specific NR7 fires 15.6%** — and BOTH are DEAD (0/85 and 0/96 BH-FDR survivors). | `STRATEGY_BLUEPRINT.md:289` ("Session-specific NR7 … fire rate 15.6% … 0/96 BH FDR survivors"). | **FABRICATED (wrong number) + DEAD** |
| 4 | Gap filter "0/101 BH survivors" | 0/101 | The **0/101** count is REAL — but it is the **gap-fill-during-ORB** filter, DEAD. (Qwen's count was right; the framing as a usable/novel finding is wrong — it's a closed NO-GO.) | `STRATEGY_BLUEPRINT.md:295` ("101 tests … 0/101 BH FDR survivors … DEAD"). | **VERIFIED-number / STALE-framing (closed NO-GO)** |
| 5 | "MGC US_DATA_830 size gate → 800-day trigger" | invented strategy | **No such strategy/lane exists.** No `MGC US_DATA_830` lane in `prop_profiles.py`; no "800-day trigger" anywhere in repo. | Absent from `prop_profiles.py` (MGC lanes are TOKYO_OPEN only). No grep hit for "800-day" / "US_DATA_830 size gate". | **FABRICATED (invented strategy)** |
| 6 | "Prior-day LOSS, MGC CME_REOPEN CB1 G4+, t=3.34 p=0.0016" | t=3.34, p=0.0016 | **Invented t-stat and p-value.** No result doc or validated row carries this cell with these statistics. | No matching `docs/audit/results/` doc; no `validated_setups` row (prior-day-direction × CME_REOPEN G4 at t=3.34). Per Volatile Data Rule, a memory-cited t/p with no result-doc anchor is fabricated. | **FABRICATED (invented statistics)** |
| 7 | Queries use SQLite `sqlite_master` / `PRAGMA table_info` | n/a | **Code-breaking error.** `gold.db` is DuckDB — those queries FAIL. Real idiom is `information_schema.tables` / `.columns`. | `pipeline/init_db.py:20` (`import duckdb`), `:640`, `:649`; `pipeline/check_drift.py:252`, `:266`. | **FABRICATED (wrong engine)** |
| 8 | Columns `session_vwap`, `prev_outcome` | n/a | **Neither column exists.** Real columns are per-ORB `orb_{label}_vwap` and `prev_day_direction` / `prev_day_close`. | `pipeline/init_db.py:226`, `:577` (`orb_{label}_vwap`); `:314` (`prev_day_direction`); `:312` (`prev_day_close`). | **FABRICATED (invented columns)** |
| 9 | "No CVD / L2 → fakeout-fading structurally impossible" | (argument) | **TRUE.** `bars_1m` is OHLCV-only; no order-flow/depth data exists. Order-flow-premised strategies must be rejected at intake. | `pipeline/init_db.py:34` (`bars_1m` DDL — OHLCV columns only). | **VERIFIED** |
| 10 | Process: pre-reg first, Mode A holdout 2026-01-01, multi-RR×aperture, BH-FDR, MinBTL | (process) | **VERIFIED** — matches repo discipline. | `holdout_policy.py:90` (HOLDOUT_SACRED_FROM); `.claude/rules/research-truth-protocol.md` (pre-reg + BH-FDR + MinBTL); `.claude/rules/backtesting-methodology.md` (multi-RR×aperture). | **VERIFIED** |

---

## What this closes

- **The "awaiting OOS" stale loop is closed for good (row 1).** The VWAP
  US_DATA_1000 candidate is not a candidate — it is a live lane whose one-shot
  OOS window is spent. Re-running the scan is blocked by design
  (`research/vwap_comprehensive_family_scan.py:90-97`) and would require deleting
  the audit trail — a Mode A violation. **No action remains on this candidate
  except recording its true state (this doc).**
- **CME_PRECLOSE is permanently logged DEAD (row 2)**, not "live" — so it is not
  re-chased as a deployment target.
- **The fabricated items (rows 3, 5, 6, 7, 8) are named explicitly** so they are
  not silently absorbed into future work.

## Caveats / disconfirming / limitations

- **This audit did NOT re-run any backtest.** All "DEAD" / "FABRICATED" verdicts
  rest on existing result docs + the verified absence of supporting artifacts, not
  on fresh computation. A fabricated stat is judged fabricated because no result
  doc carries it — absence of evidence is treated as disconfirming ONLY for claims
  that, if true, would necessarily have left a committed artifact (a pre-reg, a
  result doc, a validated_setups row).
- **The live `gold-db` MCP query failed** (`validated_setups` missing) because this
  worktree's `gold.db` is an empty 12 KB husk — so the ALREADY-LIVE verdict (row 1)
  leans on `prop_profiles.py` + the file-backed `strategy-lab` allocator, NOT on a
  DB fitness query. If those two committed surfaces were themselves stale, row 1
  could be wrong; they were cross-checked against each other and agree.
- **Numbers re-derived from result docs inherit those docs' Mode (A vs B).** The
  consumed VWAP scan is Mode A; the deployed lane's +0.210 ExpR in `prop_profiles.py`
  is a Mode-B-grandfathered figure (N=701) and is cited as a deployment label, not
  as a re-validated Mode A statistic.
- **Verdicts are point-in-time (2026-06-11).** A future DB rebuild or scan can move
  any of them — re-verify against live surfaces before acting, per the Volatile
  Data Rule.

## Lesson logged

A status word ("awaiting OOS", "live", "PROMISING") is the stalest part of any
artifact. Falsify it against the cheapest ground truth — *does the dated result
doc exist? is the lane already in `prop_profiles.py` / the allocator?* — BEFORE
building on it. The first draft of the governing plan propagated the stale
"awaiting OOS" line; an adversarial second-pass caught that the lane was already
deployed and OOS-consumed. This is the same stale-title trap recorded in
`memory/feedback_blocker_framed_batons_outlive_their_fix_verify_before_reauditing_2026_06_06.md`.

## Related

- `docs/governance/ai_session_canonical_prompt.md` — corrected anti-hallucination
  prompt distilled from this audit.
- `docs/audit/results/2026-04-18-vwap-comprehensive-family-scan.md` — the consumed
  OOS result (proof the holdout is spent).
- `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO / §9 — verdict registry (summary lines
  STALE; verify against result docs + `prop_profiles.py`).
