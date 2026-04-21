# G4 — Chordia t-Band Certificate

**Candidate:** ________________________
**Observed |t|:** ________________________

---

## Purpose

Per Chordia et al 2018 (`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`):
- No pre-registered theory → require `|t| ≥ 3.79` for any claim of edge
- Pre-registered theory (local extract + page cite) → `|t| ≥ 3.00`

Both thresholds locked in `pre_registered_criteria.md` Amendment 2.2.

## Theory justification required for t ≥ 3.00 band

The candidate must cite ONE or MORE of:

| Literature extract | Page / section | Theory claim | Used to justify |
|---|---|---|---|
| `docs/institutional/literature/________________.md` | | | t ≥ 3.00 |

Claims are verifiable to the extract text. Training-memory references are NOT acceptable per CLAUDE.md Local Academic / Project-Source Grounding Rule.

## Band assignment

- [ ] `|t| ≥ 3.79` — Chordia strict, no theory required. Applies.
- [ ] `|t| ≥ 3.00` — Chordia with-theory band. At least one verified literature extract above. Applies.
- [ ] Below both bands. Candidate does NOT clear G4.

## Smell-test interaction

If `|t| > 7` (PR #48-class), G5 (smell-test clearance) is required in addition to G4. G4 alone is insufficient for high-t findings per edge-finding-playbook.md §3.

## Verdict

- [ ] CLEAR — `|t|` exceeds the applicable band; theory citation (if band is 3.00) is verifiable.
- [ ] FAIL — below threshold, or band-3.00 claimed without verifiable theory.

## Failure disposition

FAIL → candidate cannot advance under G4. Either rewrite pre-reg to target stricter effect size (requires more data) or accept as RESEARCH_SURVIVOR not CANDIDATE_READY.

## Literature citation

- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`
- `docs/institutional/pre_registered_criteria.md` Amendment 2.2

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of this certificate: ________________
- Pinned `pre_registered_criteria.md` commit SHA at eval time: ________________
