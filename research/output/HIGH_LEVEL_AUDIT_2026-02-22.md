# High-Level Audit (2026-02-22)

Prompted by concern: "p-value exactly 0 is suspicious" and request for no-BS audit.

## 1) Was `p = 0` suspicious?
Yes, if reported literally. In prior quick check it happened because:
- finite permutations (300)
- naive estimator `count >= obs / N` can print `0.0000` when none exceed

### Fix applied
Re-ran permutation sanity check with:
- 1000 permutations
- corrected estimator: `p = (ge + 1) / (N + 1)`

So p is never exactly zero.

## 2) Updated permutation sanity results (corrected)
- A1: p = 0.0010
- A2: p = 0.0010
- A3: p = 0.0060
- A0: p = 0.0150
- B1: p = 0.0390
- B2: p = 0.1239 (weak)

Interpretation:
- A-tier candidates (A0/A1/A2/A3) look unlikely to be pure random alignment in this test.
- B2 is weak under this noise check.

## 3) Script/process issues found and fixed
- Pair-tag validation issues in lead-lag scanner (could allow malformed tags) — fixed.
- Frequency/year normalization bug (could distort signals/year) — fixed.
- Non-standard leader tag handling (`B2`) in overlay script caused bad column bind — fixed.

## 4) Remaining methodological risks (still real)
1. Data-snooping risk from iterative reuse of same dataset.
2. Some scripts compute quantile thresholds on full sample (should be train-only for strict deployment).
3. Repeated viewing of 2025 test can leak selection bias over many cycles.
4. Class imbalance in certain candidates (very high ON-rate) can exaggerate uplift interpretation.

## 5) Practical confidence tier (current)
- Higher confidence: A1, A2, A3, A0
- Medium/weak: B1
- Weak/noise-prone: B2

## 6) Mandatory next controls before promotion
- Freeze hypotheses and thresholds.
- Train/dev/test split locked and one-way.
- BH-FDR within each family scan.
- Forward-only shadow validation window.
- No parameter retune after test reveal.

## Bottom line
Not pure noise, but also not immunity from data-snooping.
A-tier set survives this audit; weaker candidates should not be promoted without stricter forward checks.
