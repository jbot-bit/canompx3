---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/<companion-or-self-with-anchor>.md
flip_rate_pct: 0.0
# heterogeneity_ack: true   # uncomment and set when flip_rate_pct >= 25
---

# <Short title of the pooled claim>

Date: YYYY-MM-DD
Author: <your-session-id or initials>
Scope: <instruments / sessions / apertures / RRs covered by the pooled test>

## Headline pooled claim

One paragraph stating the pooled p-value, ExpR, and BH framing (K value). Name
the exact cells in the universe.

## Per-cell breakdown (mandatory)

Table of every cell with:

| instrument | session | orb_minutes | rr | entry | confirm | filter | direction | N | ExpR | sign | p |
|---|---|---|---|---|---|---|---|---|---|---|---|

`sign` column: `+` if per-cell ExpR agrees with pooled direction, `-` if it
opposes, `0` if |per-cell ExpR| < 0.02 (noise floor).

Flip rate: `<count of '-' cells> / <total cells> = <flip_rate_pct>%`. Record the
exact number in the `flip_rate_pct` front-matter field above.

## Heterogeneity verdict

- If `flip_rate_pct < 25`: pooled framing is safe to quote in memory and
  doctrine. Record below why the dissenting cells are not a structural
  artefact.
- If `flip_rate_pct >= 25`: set `heterogeneity_ack: true` in front-matter and
  record below why the pooled claim is misleading as a standalone quote. Any
  trading decision MUST be made on the per-lane cells, not the pooled number.

## Implications

- What this changes in memory / doctrine / allocator.
- What it does NOT change (explicit).

## Provenance

- Research script path
- Exact SQL / pre-reg hash
- Holdout window applied (Mode A `trading_day < 2026-01-01`)
- K framing(s) reported
