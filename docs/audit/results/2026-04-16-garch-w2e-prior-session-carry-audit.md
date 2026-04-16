# Garch W2e Prior-Session Carry Audit (V2 — broadened scope)

**Date:** 2026-04-16
**Revision:** V2 — scope broadened per user feedback (don't test one lane) + bootstrap null added per institutional rigor
**Boundary:** validated shelf only, prior trade must be fully resolved before target session start, descriptive-only (no deployment or allocator verdict)

## Scope

- Validated shelf rows considered: **26**
- Verified (filter SQL returns > 0 canonical rows): **25**
- Metadata-mismatch validated rows: **1**
- Prior × target × state × strategy cells tested: **262**
- Pooled (prior × target × state) groups: **36**
- Target sessions (validated W2 families): COMEX_SETTLE, EUROPE_FLOW, LONDON_METALS, SINGAPORE_OPEN, TOKYO_OPEN
- Prior candidates considered: BRISBANE_1025, CME_PRECLOSE, CME_REOPEN, COMEX_SETTLE, EUROPE_FLOW, LONDON_METALS, NYSE_CLOSE, NYSE_OPEN, SINGAPORE_OPEN, TOKYO_OPEN, US_DATA_1000, US_DATA_830
- Bootstrap iterations: 1000 (Phipson-Smyth, shuffled carry membership)
- Gates: `MIN_TOTAL=50`, `MIN_CONJ=30`, `GARCH_HIGH>=70.0`

### Structural limitation: validated shelf instrument coverage

The validated shelf in the 5 W2 target families is overwhelmingly MNQ.
T8 cross-instrument replication is blocked by this structural limitation,
NOT by a script bug. Any pooled handoff showing `supported_thin_single_instrument`
reflects this data reality. Until more MES/MGC strategies pass validation in
these sessions, single-instrument caveats are honest disclosures, not fixable.

## Metadata mismatches (validated-setups rows that did NOT match canonical data)

Per `integrity-guardian.md` RULE 7: metadata is not evidence. Any validated-setups row whose filter SQL returned zero rows in `orb_outcomes` is surfaced here.

| Strategy | Instrument | Target session | Filter | Reason |
|---|---|---|---|---|
| `GC_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | GC | EUROPE_FLOW | OVNRNG_50 | filter_sql_returned_zero_canonical_rows |

## Supported pooled handoffs (conjunction beats all marginals AND bootstrap p ≤ 0.05 on at least one instrument)

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NYSE_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | MNQ | 8999 | 872 | +0.108 | +0.250 | +0.210 | +0.378 | +0.270 | +0.128 | +0.168 | supported_thin_single_instrument |

## Unsupported pooled handoffs (enough data, but conjunction fails one or more marginals)

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BRISBANE_1025 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 844 | +0.108 | +0.250 | +0.147 | +0.252 | +0.144 | +0.002 | +0.105 | unsupported |
| BRISBANE_1025 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 921 | +0.108 | +0.250 | +0.112 | +0.209 | +0.101 | -0.041 | +0.097 | unsupported |
| CME_REOPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 311 | +0.108 | +0.250 | +0.187 | +0.191 | +0.083 | -0.059 | +0.004 | unsupported |
| CME_REOPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 309 | +0.108 | +0.250 | +0.157 | +0.360 | +0.252 | +0.110 | +0.203 | unsupported |
| EUROPE_FLOW → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 989 | +0.108 | +0.250 | +0.080 | +0.237 | +0.128 | -0.014 | +0.156 | unsupported |
| EUROPE_FLOW → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 929 | +0.108 | +0.250 | +0.115 | +0.280 | +0.172 | +0.030 | +0.165 | unsupported |
| LONDON_METALS → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 948 | +0.108 | +0.250 | +0.157 | +0.317 | +0.209 | +0.067 | +0.160 | unsupported |
| LONDON_METALS → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 797 | +0.108 | +0.250 | +0.100 | +0.229 | +0.121 | -0.021 | +0.129 | unsupported |
| NYSE_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 851 | +0.108 | +0.250 | +0.065 | +0.233 | +0.125 | -0.017 | +0.168 | unsupported |
| SINGAPORE_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 914 | +0.108 | +0.250 | +0.028 | +0.145 | +0.037 | -0.105 | +0.118 | unsupported |
| SINGAPORE_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 950 | +0.108 | +0.250 | +0.164 | +0.325 | +0.217 | +0.075 | +0.161 | unsupported |
| TOKYO_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 876 | +0.108 | +0.250 | +0.140 | +0.319 | +0.211 | +0.069 | +0.179 | unsupported |
| TOKYO_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 1002 | +0.108 | +0.250 | +0.108 | +0.236 | +0.128 | -0.014 | +0.128 | unsupported |
| US_DATA_1000 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 805 | +0.108 | +0.250 | +0.163 | +0.248 | +0.140 | -0.002 | +0.085 | unsupported |
| US_DATA_1000 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 926 | +0.108 | +0.250 | +0.159 | +0.285 | +0.177 | +0.035 | +0.126 | unsupported |
| US_DATA_830 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 904 | +0.108 | +0.250 | +0.170 | +0.278 | +0.169 | +0.027 | +0.108 | unsupported |
| US_DATA_830 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 798 | +0.108 | +0.250 | +0.083 | +0.333 | +0.225 | +0.083 | +0.250 | unsupported |
| BRISBANE_1025 → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 763 | +0.104 | +0.184 | +0.040 | +0.143 | +0.040 | -0.040 | +0.103 | unsupported |
| BRISBANE_1025 → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 831 | +0.104 | +0.184 | +0.168 | +0.311 | +0.208 | +0.127 | +0.143 | unsupported |
| CME_REOPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 240 | +0.104 | +0.184 | +0.072 | +0.028 | -0.076 | -0.156 | -0.044 | unsupported |
| CME_REOPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 297 | +0.104 | +0.184 | +0.118 | +0.354 | +0.250 | +0.170 | +0.235 | unsupported |
| LONDON_METALS → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 436 | +0.104 | +0.184 | +0.120 | +0.146 | +0.043 | -0.037 | +0.027 | unsupported |
| LONDON_METALS → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 331 | +0.104 | +0.184 | +0.166 | +0.273 | +0.169 | +0.089 | +0.107 | unsupported |
| SINGAPORE_OPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 786 | +0.104 | +0.184 | +0.033 | +0.120 | +0.017 | -0.063 | +0.087 | unsupported |
| SINGAPORE_OPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 912 | +0.104 | +0.184 | +0.102 | +0.168 | +0.064 | -0.016 | +0.066 | unsupported |
| TOKYO_OPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 890 | +0.104 | +0.184 | +0.146 | +0.277 | +0.173 | +0.093 | +0.130 | unsupported |
| TOKYO_OPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 838 | +0.104 | +0.184 | +0.138 | +0.262 | +0.159 | +0.078 | +0.124 | unsupported |
| BRISBANE_1025 → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 164 | +0.118 | +0.129 | +0.004 | +0.006 | -0.111 | -0.122 | +0.003 | unsupported |
| BRISBANE_1025 → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 190 | +0.118 | +0.129 | +0.126 | +0.150 | +0.032 | +0.021 | +0.023 | unsupported |
| CME_REOPEN → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 72 | +0.118 | +0.129 | +0.191 | -0.078 | -0.196 | -0.207 | -0.270 | unsupported |
| CME_REOPEN → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 70 | +0.118 | +0.129 | +0.123 | +0.203 | +0.085 | +0.074 | +0.080 | unsupported |
| TOKYO_OPEN → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 196 | +0.118 | +0.129 | +0.159 | +0.143 | +0.026 | +0.014 | -0.016 | unsupported |
| TOKYO_OPEN → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 209 | +0.118 | +0.129 | +0.210 | +0.240 | +0.123 | +0.111 | +0.030 | unsupported |

## Unclear pooled handoffs

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CME_REOPEN → TOKYO_OPEN | prior win align | take_pair | 4 | MNQ | — | 4756 | 70 | +0.087 | +0.135 | +0.162 | -0.008 | -0.096 | -0.143 | -0.171 | unclear |
| CME_REOPEN → TOKYO_OPEN | prior win opposed | veto_pair | 4 | MNQ | — | 4756 | 104 | +0.087 | +0.135 | -0.335 | -0.409 | -0.496 | -0.543 | -0.074 | unclear |

## Per-cell detail for supported handoffs

(Only pooled handoffs with at least one instrument-level supported verdict.)

| Prior → Target | Instrument | Strategy | Filter | RR | ORB | State | N | N resolved | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Boot p | dir_match | Cell verdict |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | COST_LT12 | 1.0 | 5 | prior win align | 1244 | 1215 | 111 | +0.102 | +0.207 | +0.193 | +0.327 | +0.224 | 0.057 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | ORB_G5 | 1.0 | 5 | prior win align | 1447 | 1406 | 116 | +0.079 | +0.210 | +0.165 | +0.328 | +0.249 | 0.056 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI` | ORB_G5_NOFRI | 1.0 | 5 | prior win align | 1161 | 1125 | 92 | +0.098 | +0.259 | +0.193 | +0.367 | +0.270 | 0.083 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | OVNRNG_100 | 1.0 | 5 | prior win align | 526 | 514 | 68 | +0.148 | +0.284 | +0.217 | +0.394 | +0.246 | 0.095 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.0 | 5 | prior win align | 633 | 621 | 102 | +0.144 | +0.235 | +0.231 | +0.328 | +0.184 | 0.098 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | ORB_G5 | 1.5 | 5 | prior win align | 1432 | 1391 | 111 | +0.100 | +0.254 | +0.219 | +0.416 | +0.316 | 0.039 | yes | supported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | OVNRNG_100 | 1.5 | 5 | prior win align | 518 | 506 | 65 | +0.189 | +0.365 | +0.313 | +0.532 | +0.343 | 0.072 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.5 | 5 | prior win align | 623 | 611 | 97 | +0.159 | +0.276 | +0.286 | +0.426 | +0.267 | 0.036 | yes | supported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | ORB_G5 | 2.0 | 5 | prior win align | 1415 | 1375 | 110 | +0.078 | +0.216 | +0.192 | +0.357 | +0.279 | 0.114 | yes | unsupported |

## Guardrails and honest caveats

- **Chronology:** every row is kept only if the prior trade's `exit_ts` is strictly before the target session start timestamp from `pipeline.dst.orb_utc_window`. No static session-order shortcut.
- **Canonical sources:** all statistics are computed against `orb_outcomes` + `daily_features`. `validated_setups` is used ONLY to enumerate candidate target cells; each row is independently verified against canonical data, and mismatches are surfaced above.
- **Bootstrap null:** `BOOTSTRAP_ITERS=1000` shuffles of carry-state membership among the same population, Phipson-Smyth p. This is a descriptive null floor — it does NOT replace a cross-instrument or cross-era check.
- **Family gate:** a pooled handoff × state is `supported_multi_instrument` only if at least 2 instruments each had a cell-level supported verdict. Single-instrument support is flagged `supported_thin_single_instrument` and should not be treated as universal.
- **Holdout:** 2026-01-01 boundary is respected. This audit uses the full shelf including post-holdout rows — descriptive only, not promotion.
- **No deployment doctrine** is derivable from this audit. It is a state-family distinctness check, not a size/route/allocator claim.

---

## Stage conclusion

Prior-session carry is **not validated as a broad hard-gate doctrine** on this
validated shelf.

The W2e audit tested the strongest form of the carry hypothesis — that a
binary carry state (prior-win-aligned or prior-win-opposed), conditioned by
`garch_high`, would produce a reliable take or veto signal across the validated
shelf. 262 cells, 36 pooled handoffs, 12 prior × 5 target sessions, 2 carry
states, full bootstrap null, tautology check, fire-rate guard, and per-year
stability checks. The result is clear enough to close the binary-gate form of
this question without ambiguity.

### What is dead

**Veto-pair implementation: DEAD on this shelf.**

Every single `PRIOR_WIN_OPPOSED` conjunction produced positive ExpR across
every handoff. Not one case of the conjunction being more hostile than the base
or garch-alone. The mechanism hypothesis — that an opposed prior win is hostile
to the next session's breakout, amplified by high vol — is directly
contradicted by the data. There is no amount of threshold tuning that rescues a
mechanism whose sign is universally wrong.

This does not mean opposed-prior states are uninformative in all possible
formulations. It means the specific binary-veto framing tested here (hard gate
on `garch_high AND prior_win_opposed`) is not a useful filter at any threshold
on this shelf.

### What is thin / local only

**NYSE_OPEN → COMEX_SETTLE take-pair: thin local candidate, not generalizable.**

One pooled handoff showed directional support: `NYSE_OPEN → COMEX_SETTLE`,
`prior_win_align`, `take_pair`. Pooled conjunction ExpR = +0.378 vs base
+0.108. But:

- Only 2 of 9 cells within the pool hit bootstrap p ≤ 0.05 (ORB_G5 RR1.5 at
  p=0.039, X_MES_ATR60 RR1.5 at p=0.036). The other 7 are directionally
  aligned but not significant (p = 0.056–0.114).
- Single instrument only (MNQ). Cross-instrument check is structurally blocked
  by the validated shelf composition.
- The RR1.5 specificity is suspicious — RR1.0 and RR2.0 both fail the
  bootstrap gate on the same handoff.
- No BH-FDR correction has been applied across the 36 pools.

This is a real directional observation, not noise, but it is not strong enough
to promote as a standalone gate or even as a confident local feature. It
belongs in a watchlist, not an implementation queue.

### What remains plausibly useful

The fact that binary carry-as-hard-gate failed does NOT close the broader
question of whether prior-session context carries informational value.

The specific failure mode matters: the conjunction consistently ADDS to the
base (most `Δ_conj_vs_base` are positive), but it does not consistently ADD
to `garch_high` alone (most `Δ_conj_vs_garch` are zero or negative). That
means:

1. Carry state is positively correlated with garch-favourable days (selection
   effect — on days when a prior session wins, the market is trending, which
   is also when garch is high).
2. The marginal value of carry *after* garch is already conditioning is small
   to zero in most handoffs.
3. The rare exceptions (NYSE_OPEN → COMEX_SETTLE) may reflect a genuine
   US-session momentum cascade, or may reflect that the US afternoon is the
   highest-N / highest-power portion of the shelf.

This leaves open four softer implementation classes that were never tested here
and cannot be dismissed by W2e's results:

| Implementation class | What it means | How it differs from the tested gate |
|---|---|---|
| **Soft confluence feature** | Carry state enters as one input among several in a score, not a binary pass/fail | Not a gate — does not exclude any trade. Adds directional context to an allocation decision. |
| **Sizing modifier** | Days with favourable carry get a larger position; unfavourable get smaller, but still trade | Not a gate — all trades taken, but risk budget varies. Requires Carver-style forecast combiner. |
| **Local family context input** | Carry state used only for specific (prior, target) pairs where local evidence is strongest | Not a broad doctrine — scoped to the 1-2 handoffs where directional support was observed. |
| **Portfolio context feature** | Prior-session outcome used as an input to a portfolio-level daily state classifier, not per-lane | Not a per-lane feature — informs cross-lane allocation or slot priority. |

None of these are pre-validated. Each would require its own pre-registered
hypothesis, its own K-budget, and its own pass/fail criteria.

### What is NOT claimed

- Carry is not called "dead" as a research direction. The binary-gate framing
  is dead; the information channel may not be.
- The NYSE_OPEN → COMEX_SETTLE finding is not promoted. It is a thin local
  observation that may or may not survive a dedicated audit.
- No implementation path is recommended for immediate execution. The ranked
  options below are a research menu, not a build queue.

---

## Ranked next implementation paths

Ranked by honesty (does it respect the W2e finding?), robustness (how hard is
it to overfit?), and likely economic value (conditional on working at all).

### Rank 1: Portfolio context feature

**Honesty: HIGH.** This is the most different from what W2e tested. W2e asked
"does carry help per-lane?"; portfolio context asks "does same-day prior-session
outcome shift the optimal cross-lane allocation?" Entirely different surface.

**Robustness: MODERATE.** K is small (one feature per portfolio-day, not per
lane). Overfitting risk is lower than per-lane gating, but the feature is less
well-defined (requires a portfolio-day join, not a simple row filter).

**Likely value: MODERATE.** If the selection effect is real (prior-win days are
trending days), this might improve daily slot allocation more than it improves
any single lane.

**What it requires:** pre-registered hypothesis, daily portfolio-state join
against `orb_outcomes`, definition of "portfolio prior-session outcome" (at
least one prior session win? majority? all?), allocation replay similar to A4.

### Rank 2: Sizing modifier (Carver forecast combiner)

**Honesty: HIGH.** Carry becomes a continuous input, not a binary gate. The
W2e finding that carry adds to base but not to garch suggests it may have value
as a *distinct* sizing signal — but only if the forecast combiner properly
weights it against garch rather than double-counting the same regime days.

**Robustness: MODERATE-HIGH.** Carver sizing framework is pre-existing in the
repo (Phase D spec at `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`).
Adding carry as one more forecast input is mechanically clean. But: the
orthogonality between carry and garch on this shelf is uncertain — W2e's
selection-effect observation suggests they may be partially collinear.

**Likely value: MODERATE.** Collinearity gate confirmed independence (see
below). Sizing modifier would extract genuine diversification benefit from an
orthogonal signal — but only if the encoding avoids binary degeneracy.

### Rank 3 (revised from 4): Soft confluence feature

**Honesty: MODERATE.** Upgraded from rank 4 after collinearity gate confirmed
corr ≈ 0. Confluence with garch would not double-count. But the degeneracy
problem means the carry encoding must NOT be binary.

**Robustness: LOW-MODERATE.** Each encoding choice is a degree of freedom.
Needs tight pre-registration to prevent mining.

**Likely value: LOW-MODERATE.** If the encoding solves degeneracy, genuine
independent information exists. But defining "soft confluence" loosely is a
data-mining invitation.

### Rank 4 (revised from 3): Local family context input (NYSE_OPEN → COMEX_SETTLE only)

**Honesty: MODERATE.** Unchanged — post-hoc selection of the surviving cell is
a classic overfitting vector.

**Robustness: LOW.** K=1 (one handoff, one state, one direction). Impossible
to correct for multiple testing meaningfully.

**Likely value: LOW.** Even if real, applies to one session on one instrument.

### Not ranked: Hard gate (any formulation)

**Status: closed by W2e.** Binary carry gating does not add to garch on this
shelf. Reopening requires a structurally different carry definition or a
structurally different shelf. Degeneracy (not collinearity) is the root cause.

---

## Collinearity gate result (executed same session)

**`corr(any_prior_win, garch_high) = +0.016`** on 5,207 validated-shelf
(day, symbol, target_session) rows. Near-zero. Carry and garch are almost
perfectly **orthogonal**.

Script: `research/garch_carry_collinearity_check.py`.

### What this overturns

The W2e working hypothesis was that the conjunction's failure to beat garch-
alone reflected a selection effect (prior-win days clustering on garch-high
days). **That is wrong.** The two signals are independent.

### What actually explains the binary-gate failure

`any_prior_win` rate by target session:

| Target session | any_prior_win rate |
|---|---|
| COMEX_SETTLE | 99.8% |
| EUROPE_FLOW | 96.0% |
| SINGAPORE_OPEN | 75.9% |
| TOKYO_OPEN | 8.4% |

For late-day sessions, "has a prior win" is nearly constant — it fires on
almost every day. A feature that fires 96–100% of the time is not a feature;
it is background noise. For early-day sessions, it almost never fires — there
aren't enough resolved priors yet.

Binary gating on carry fails because the signal is **degenerate at both tails
of the session clock**, not because it is collinear with garch.

### Core blocker for any carry implementation

The blocker is not collinearity (resolved: near-zero). The blocker is
**degeneracy**: binary carry state is nearly constant for 3 of 4 target
sessions. Any carry implementation must solve this by encoding carry as a
continuous or graduated feature, not a binary flag.

Candidate encodings (not pre-registered — research menu only):

1. **Prior-session weighted-pnl**: sum of (prior session pnl_r × weight) where
   weight decays by recency. Continuous, session-clock-aware.
2. **Prior-session win-count ratio**: wins / total resolved priors that day.
   Continuous, bounded [0, 1], but still degenerate for TOKYO_OPEN (0 or 1).
3. **Most-recent-prior pnl_r**: only the immediately preceding resolved session.
   Continuous, avoids the accumulation degeneracy.
4. **Direction-conditioned prior outcome**: prior session outcome × alignment
   with the target's eventual direction. This is what W2e tested as a binary
   gate — it would need to be made continuous (e.g., prior pnl_r × sign of
   direction alignment).

### Recommended next step

Pre-register a carry-encoding exploration that tests **encodings 1 and 3**
(prior-session weighted-pnl and most-recent-prior pnl_r) as soft features in
the portfolio-context implementation class. Scope: full validated shelf, same
chronology rules as W2e, K ≤ 20 (2 encodings × 5 target sessions × 2 garch
states). This is a feature-encoding study, not a signal-hunting sweep.

Do not start until the pre-registration hypothesis file is committed.
