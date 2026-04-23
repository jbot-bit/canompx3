---
slug: mgc-5m-payoff-compression-audit
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-22
updated: 2026-04-22
task: Follow the GC->MGC translation audit with a narrow MGC 5-minute payoff-compression audit on the warm translated families. Test whether the rescue path is exit/payoff handling rather than more proxy discovery.
---

# Stage: MGC 5m payoff-compression audit

## Question

`docs/audit/results/2026-04-19-gc-mgc-translation-audit.md` already answered the first question:

- GC overlap-era strength is real
- trigger parity is mostly fine
- the bridge breaks mainly in **MGC 5-minute payoff translation**

The unresolved next move is:

> on the warm translated families, is the right rescue question lower-RR / exit-shape handling rather than more GC proxy exploration?

## Scope Lock

- Instrument in scope: `MGC`
- Aperture in scope: `O5` only
- Families in scope: `US_DATA_1000`, `NYSE_OPEN`, `EUROPE_FLOW`
- Compare against overlap-era `GC` only as reference, not as proof of deployability
- No broad proxy reopening
- No 15m/30m expansion in this stage

## Blast Radius

- Research-only:
  - one runner under `research/`
  - one result doc under `docs/audit/results/`
- No production code changes
- No shelf writes or live-config writes

## Approach

1. Start from the warm translated families identified in the translation audit.
2. Audit the MGC payoff stack at `O5`:
   - WR
   - avg win
   - avg loss
   - expectancy across low-RR bands
   - any obvious exit-shape compression signature
3. Report whether the evidence supports:
   - `PAYOFF_COMPRESSION_REAL`
   - `LOW_RR_RESCUE_PLAUSIBLE`
   - `NO_RESCUE_SIGNAL`
4. If rescue looks plausible, stop at prereg recommendation for a narrow exit-shape follow-up.

## Suggested Branch / PR

- Branch: `research/mgc-5m-payoff-compression`
- PR title: `research(mgc): audit 5m payoff compression on translated gold families`

## Acceptance Criteria

1. Scope stays on `MGC O5` warm families only.
2. Translation-audit conclusions are treated as upstream truth, not re-proved broadly.
3. Result doc clearly states whether the right next move is an exit-shape prereg or a hard stop.
4. No broad GC proxy re-opening language appears in the result.

## Non-goals

- Not a new GC proxy campaign
- Not a wider-aperture gold audit
- Not a deployment proposal

