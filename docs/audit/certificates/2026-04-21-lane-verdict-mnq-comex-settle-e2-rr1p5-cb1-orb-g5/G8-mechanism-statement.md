# G8 — Mechanism Statement Certificate

**Candidate:** `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`

---

## Purpose

This is a project-canon-grounded mechanism statement for a deployed-lane retrospective verdict.

## Mechanism statement

```text
This is an R1 pre-entry filter that excludes undersized ORBs. The project-canon claim is operational: a materially formed ORB is more likely to represent a real opening range than a noise print, so the gate screens low-information setups rather than inventing a new alpha source.
```

## Literature grounding

### B — PROJECT-CANON-GROUNDED

- [x] Mechanism is supported by `docs/institutional/mechanism_priors.md` §4
- [x] Mechanism is consistent with the canonical filter implementation in `trading_app/config.py`
- [x] This lane is treated as an R1 filter / operational screen, not a new discovery claim

## Mechanism-prior-hierarchy check (`mechanism_priors.md` §4)

- [x] Candidate's proposed role matches the effect shape and implementation role
- [x] No role reframing is required in Phase B

## Verdict

- [x] GROUNDED — project canon, not training-memory only.

## Literature citation

- `docs/institutional/mechanism_priors.md`
- `trading_app/config.py`

## Authored by / committed

- Author: `Codex`
- Commit SHA of this certificate: `dfb1bbab`
