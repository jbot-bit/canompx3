# London `overnight_range_pct≥80` Conditioner — Reopen-Decision Memo (2026-06-02)

**Type:** reopen-decision memo (NOT a discovery scan, NOT a capital pre-reg).
**Author:** claude
**Scope:** decide whether to reopen the only surviving London candidate
(MES LONDON_METALS O30 RR1.5 long `overnight_range_pct≥80`) for a capital pre-reg
now, or let its locked signal-only shadow run. No discovery, no new cells tested,
no DB mutation.
**Decision requested by operator:** "draftr the london thing."
**Canonical sources queried live this session:** gold-db MCP (`outcomes_stats`),
research-catalog MCP (`estimate_k_budget`, `search_research_catalog`), and the two
result/hypothesis files cited inline below.
**Capital at risk:** none. No DB mutation. No new pre-reg filed. No sacred-window read.

---

## Why this memo exists

A prior `/clear` session handed off a **CONTINUE (one path)** verdict for the only
surviving London candidate:

> overnight_range_pct≥80 directional conditioner on `MES_LONDON_METALS_E2_RR1.5` long.

The baton's premise was that the graveyard reopen rule ("new dataset") was now met by
accrued holdout data, and that the candidate "isn't measured-dead — it died on sample
power, not edge validity."

This memo grounds that premise against current truth surfaces. **Two facts the baton did
not account for invert its conclusion.**

---

## Finding 1 — the candidate was ALREADY powered-OOS re-judged, and it returned UNVERIFIED

`docs/audit/results/2026-05-29-powered-oos-graveyard-resweep-and-mgc-wide-scan.md:41`
re-judged this exact cell with accrued holdout data, using **trade-fraction OOS**
(the powered method the baton was hoping would rescue it) and the **canonical
triple-join** (`orb_minutes` pinned, RULE 9):

| Cell | N | FULL t | IS t | Frac-OOS N | Frac-OOS ExpR | Frac-OOS t | Power | Tier | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| MES LONDON_METALS O30 RR1.5 long ovn≥80 | 201 | 2.64 | 2.87 | 61 | +0.073 | 0.48 | **0.46** | STATISTICALLY_USELESS | **UNVERIFIED** |

Two load-bearing points:

- **The optimistic figure that likely inspired the baton's optimism is RETRACTED.**
  Line 30 of the same doc: an earlier exploratory "H1 t=4.59" was a **3×-inflated
  artefact** from a defective join missing `d.orb_minutes = o.orb_minutes` (inflates N
  by 3×, t by √3≈1.73×). The honest full-period t is **2.64**, not 4.59.
- **OOS power 0.46 < 0.50 → the OOS slice is statistically useless** (RULE 3.3
  `.claude/rules/backtesting-methodology.md`; `research/oos_power.py::POWER_TIERS`). An
  underpowered OOS cannot distinguish "signal alive", "signal dead", or "signal
  reversed." The correct verdict is therefore **UNVERIFIED — not CONTINUE-to-capital,
  and not DEAD.**

The baton framed the bar as "unforgiving t≥3.79 it still faces." It does not *face* that
bar in the future tense — on the most recent powered read it **already missed it**:
full-period t=2.64 is under both the t≥3.79 without-theory bar and the t≥3.00
with-theory bar (and the with-theory carve-out does not even apply — see Finding 3).

## Finding 2 — the 2026 OOS window is ALREADY claimed by a live one-shot shadow

`docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml` is a
**locked, currently-running** signal-only shadow on this exact cell
(MES LONDON_METALS **O30** RR1.5 long E2 CB1, `overnight_range_pct≥80`). Its terms:

- `this_is_the_one_shot_read: true` — it has **already consumed** the Mode-A
  single-shot read of the `2026-01-01 → 2026-12-15` window.
- `review_date: 2026-12-15` — gates are evaluated **once**, then.
- `re_tuning_after_observation: forbidden`; `no_rescue_if_shadow_fails: true`;
  `no_window_extension_after_first_result: true`.
- `do_not_re_open_unless:` a brand-new V2 pre-reg that (a) does **not** reuse the
  2026-01-01→2026-12-15 window, and (b) cites a **new mechanism**, not a statistical
  rescue argument.

**Consequence:** filing any new O30 + ovn≥80 capital pre-reg over the 2026 window would
**double-consume the sacred holdout** and void the existing shadow — the exact
selection-bias move banned by `pre_registered_criteria.md` Amendment 2.7 and Criterion 5
selection-bias clause. Drift check #57 (MinBTL) does **not** catch this class; only this
decision gate does.

## Finding 3 — mechanism grant and the live base rate

- **Base rate confirms the conditioner framing is reasonable** (not the edge — the
  framing): MES LONDON_METALS E2 RR1.5 CB1 **unconditional** is N=5324, ExpR **−0.084**,
  Sharpe −0.077 (gold-db `outcomes_stats`, live 2026-06-02). A negative unconditional
  base is the correct setup for a *directional conditioner* — but a conditioner has to
  clear the bar on its own conditioned sample, which it does not (Finding 1).
- **No theory grant.** The shadow's tier-1 block records
  `fitschen_2013_path_of_least_resistance.md` as CORE ORB premise but
  `with_theory_claim: false` — overnight-range-percentile as a volatility-regime filter
  is "one step adjacent." So the **without-theory t≥3.79 bar applies** at any capital
  decision (Criterion 4). t=2.64 is nowhere near it.
- **K-budget is not the constraint.** `estimate_k_budget(MES, n_trials=1)` = PASS
  (6.65yr horizon, 6.65yr headroom). The wall is **power and effect size**, not MinBTL.

---

## VERDICT

**UNVERIFIED — shadow-pending. No new pre-reg is filed. No capital action.**

The single correct action is to **let the existing O30 signal-only shadow run to its
locked review date, 2026-12-15**, and judge it once, then, on its pre-committed gates.
Opening a parallel contract now would either (a) double-consume the sacred window, or
(b) launder a power-0.46 candidate toward capital — both banned.

This is **not** a kill. The candidate is genuinely UNVERIFIED (data-power-limited), per
the 2026-05-29 doctrine: *park, do not "wait" as a deployment gate; do not rescue on a
statistical argument.*

## If forward motion is wanted before 2026-12-15

Only two literature-clean options exist, and **neither is the baton's "continue":**

1. **A different, untested aperture (O5 or O15).** The baton's "MES_LONDON_METALS_E2_RR1.5"
   omitted the aperture; every artefact on file is O30. O5/O15 + ovn≥80 is a *fresh,
   never-tested cell* — it would need its own Pathway-B K=1 pre-reg, IS-only first, and
   does not touch the O30 shadow's window. Risk: no IS edge has been measured there yet;
   it may die at the IS gate.
2. **CPCV** (AFML Ch 12) — the stronger tool for short-OOS data the 2026-05-29 doc names
   as the right instrument. **BLOCKED — DOCTRINE-ONLY / NOT-YET-IMPLEMENTED** per
   `pre_registered_criteria.md` Criterion 8 and `backtesting-methodology.md` RULE 3.3. A
   pre-reg may not cite CPCV as a gate until it is built. Not available today.

---

## Limitations

- **The candidate is data-power-limited, not measured-dead.** OOS power 0.46 means
  the OOS slice cannot distinguish "alive", "dead", or "reversed" — this memo's
  UNVERIFIED verdict is a statement about *insufficient evidence*, not a kill.
  Do not cite it as a NO-GO.
- **No theory grant.** Overnight-range-percentile as a volatility-regime filter is
  one step adjacent to the Fitschen ORB premise; the without-theory t≥3.79 bar
  applies, and full-period t=2.64 is far below it.
- **Window-locked.** The 2026 sacred OOS read is already consumed by the live O30
  signal-only shadow; this memo cannot be used to justify a second read of that
  window before the 2026-12-15 review.
- **Single-aperture evidence.** Every artefact on file is O30; O5/O15 + ovn≥80 is
  untested, so "London is dead" would overclaim — only the O30 cell is characterised.
- This is a desk-judgment reconciliation memo, not an independent re-run; the
  numbers are quoted from the cited 2026-05-29 powered re-judge and live MCP
  queries, not recomputed from `orb_outcomes` in this session.

## Reproduction / provenance (all live this session, nothing from memory)

| Claim | Source |
|---|---|
| Powered re-judge UNVERIFIED, t=2.64, power 0.46 | `docs/audit/results/2026-05-29-powered-oos-graveyard-resweep-and-mgc-wide-scan.md:41` |
| "t=4.59" retracted as 3×-join artefact | same doc, line 30 |
| O30 shadow locked, one-shot 2026-12-15, no-rescue | `docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml` |
| Base rate ExpR −0.084, N=5324 | gold-db `outcomes_stats` MES/LONDON_METALS/E2/RR1.5/CB1 (live) |
| K-budget PASS for K=1 | research-catalog `estimate_k_budget(MES, 1)` (live) |
| t≥3.79 without-theory bar, OOS power floor | `pre_registered_criteria.md` Criterion 4; `backtesting-methodology.md` RULE 3.3 |
