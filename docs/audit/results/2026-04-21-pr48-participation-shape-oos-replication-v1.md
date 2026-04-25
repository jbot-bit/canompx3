# PR #48 participation-shape — OOS β₁ replication v1

**Replication of:** `docs/audit/results/2026-04-20-participation-shape-cross-instrument-v1.md` — same axes, OOS window only.

**Pre-commit:** Pathway B K=1 per instrument, K_family=3 independent hypotheses.

**Pass criterion:** sign(β₁_OOS) == sign(β₁_IS) AND t_OOS >= +2.0.

**Confirmatory audit** (no new pre-reg per research-truth-protocol.md § 10).

## Headline

| Instrument | N_OOS | β₁_OOS | t_OOS | one-tailed p | β₁_IS (PR #48) | Sign match? | t>=+2.0? | Verdict |
|---|---:|---:|---:|---:|---:|:---:|:---:|---|
| MNQ | 771 | +0.14433 | +0.964 | 0.1676 | +0.27775 | ✔ | ✘ | **OOS_WEAK_BUT_RIGHT_SIGN** |
| MES | 702 | +0.36543 | +2.492 | 0.0065 | +0.33025 | ✔ | ✔ | **OOS_CONFIRMED** |
| MGC | 601 | +0.42276 | +2.519 | 0.0060 | +0.29975 | ✔ | ✔ | **OOS_CONFIRMED** |

## Per-year OOS (where N>=50)

| Instrument | Year | N | β₁ | t |
|---|---:|---:|---:|---:|
| MNQ | 2026 | 771 | +0.14433 | +0.964 |
| MES | 2026 | 702 | +0.36543 | +2.492 |
| MGC | 2026 | 601 | +0.42276 | +2.519 |

## Summary + interpretation

- OOS_CONFIRMED (sign-match AND t>=+2.0): **2 of 3** — MES, MGC
- Right-sign-but-weak (sign-match, t<+2.0): MNQ
- Sign-flipped: none

**Verdict:** PR #48 participation-shape is **OOS-CONFIRMED on MES, MGC**. The 1 non-confirming instrument(s) must be treated as UNVERIFIED (right-sign or weak) or DEAD (sign-flipped). Cross-instrument universality is weakened; deploy-as-sizer restricted to confirmed instruments only.

## Methodology caveats (IMPORTANT)

**Rank normalisation:** this test ranks `rel_vol` **within the OOS sample only**. That establishes whether the monotonic relationship HOLDS in OOS data, but is NOT yet a test of a specific deployable rule. A production sizer rule would freeze rank thresholds on IS data and apply them to fresh OOS `rel_vol`. That's a separate, stronger test ("IS-trained-threshold applied to OOS realizations") required before any capital deployment.

**Pooled mean vs slope interpretation:** earlier the skeptical re-audit flagged that MES OOS pooled mean is −0.09R. That is correct — the average MES trade in OOS loses 0.09R. But it is NOT evidence against the participation shape, because the sizer rule is about DIFFERENTIAL performance across participation buckets, not pooled expectancy. Top-quintile vs bottom-quintile can be meaningfully asymmetric even when pooled mean is negative. β₁_OOS = +0.365 says exactly that.

**MNQ UNVERIFIED:** sign matches, t = +0.96 below the pre-committed +2.0 gate. Could be a real effect with insufficient OOS power, or a real weakening of MNQ-specific participation edge. Needs another 6-12 months of OOS data or a Pathway-B-style rule-backtest to discriminate.

## Reverses the re-audit's "MISCLASSIFIED as deploy candidate"

The skeptical re-audit (`2026-04-21-recent-claims-skeptical-reaudit-v1.md`) labelled PR #48 as "MISCLASSIFIED as deploy candidate — VALID as IS descriptive only." That classification was based on (a) no OOS run, and (b) the misleading pooled-OOS-mean negative for MES. This OOS replication invalidates point (a) for MES/MGC and clarifies point (b).

**Updated classification for PR #48:**
- MES: OOS-CONFIRMED pattern; ALIVE as a sizer-overlay deploy candidate pending IS-trained-rule backtest.
- MGC: OOS-CONFIRMED pattern; ALIVE as a sizer-overlay deploy candidate pending IS-trained-rule backtest.
- MNQ: UNVERIFIED (sign-right, power-low); watch.

## Not done by this result

- No deployment or capital action.
- No writes to validated_setups / lane_allocation / edge_families / live_config.
- Does NOT derive or backtest a concrete IS-trained sizer rule — the next bounded step if either MES or MGC is to move toward shadow.
- Does NOT re-test filtered universes (PR #48 is unfiltered-only).
- Does NOT apply Bailey DSR — DSR is a SR-based test, not applicable to OLS slope parameters. The sign+power gate (K=3 Pathway B) is the OLS-appropriate Phase 0 complement.