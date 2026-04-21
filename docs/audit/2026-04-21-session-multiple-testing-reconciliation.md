# 2026-04-21 session multiple-testing reconciliation

## Scope

This reconciliation covers only the orthogonal golden-egg hunt session on
`research/orthogonal-golden-egg-hunt-v1`. Session-local trials counted against
the ledger:

- `2026-04-21-mgc-regime-ortho-v1` = `12` family cells
- `2026-04-21-mes-session-boundary-v1` = `0` executed cells
  - blocked before data contact on a timing-validity failure
- `2026-04-21-mgc-microstructure-v1` = `8` family cells

**Total executed trials:** `20`

Authority:

- `docs/institutional/pre_registered_criteria.md` § Multiple Testing
- `docs/institutional/pre_registered_criteria.md` Amendment 2.1
- `docs/institutional/pre_registered_criteria.md` Amendment 2.2
- `outputs/session_trial_ledger.md`

## Phase 3 survivor set before session reconciliation

Current-session family outcomes, generated from canonical data in this session:

- `docs/audit/results/2026-04-21-mgc-regime-ortho-v1.md`
  - family BH-FDR survivors: `0`
  - strongest raw cell: `LDM_PREVWEEK_LONG`
    - `t_IS = 2.602`
    - `raw_p = 0.0049`
    - `q_family = 0.0587`
    - family nulls: destruction-shuffle `p=0.3466`, rng-null `p=0.3586`
- `docs/audit/2026-04-21-mes-session-boundary-timing-block.md`
  - blocked before data contact
  - contributes `0` executed trials and `0` survivors
- `docs/audit/results/2026-04-21-mgc-microstructure-v1.md`
  - family BH-FDR survivors: `0`
  - strongest raw cell: `USD830_RANGECONC_LONG`
    - `t_IS = 0.924`
    - `raw_p = 0.1782`
    - `q_family = 1.0000`
    - family nulls: destruction-shuffle `p=0.4263`, rng-null `p=0.3944`

## Binding multiple-testing gate

The decisive framework for this session remains the repo's BH-FDR / pathway
selection stack, not Harvey-Liu and not DSR.

Session result:

- Families with at least one Phase 3 BH-FDR survivor: `0 / 2 executed`
- Cells surviving family-level BH-FDR and eligible to propagate: `0 / 20`
- Session-level pathway result: **no survivors**

Because no cell survived its own locked family gate, there is no downstream
candidate set for a session-level salvage. The multiple-testing answer is
therefore fail-closed and simple: **this session produced zero empirical
survivors.**

## Supplementary checks

Supplementary checks do not override the binding framework.

- Harvey-Liu:
  - no current-session cell reached even the untheorized Chordia band `t=3.79`
  - strongest raw `t` in the session was `2.602`
  - therefore Harvey-Liu cannot rescue any candidate here
- DSR:
  - cross-check only per Amendment 2.1
  - no candidate is in a position where DSR would affect promotion because the
    empirical survivor set is already empty

## Sequential-monitoring note

This session consumed `20` executed trials from the local session cap of `300`
and produced no empirical survivors. That is informative for future alpha
budgeting even though it does not change the current disposition:

- do not treat these families as live pending-approval candidates
- if the hunt expands, carry forward the fact that top-tier orthogonal families
  `F3` and `F7` already failed honest family-level selection in this window
- the blocked `F6` family should be remembered as a timing-invalid design, not
  as an empirical miss

## Verdict

**Session-local multiple-testing verdict: no GOLD or SILVER candidates
survived.** The orthogonal hunt added posture-clearing evidence via the PCC
artifacts, but the candidate scan itself produced zero empirical survivors once
the repo's binding family-level BH-FDR / pathway framework was applied.
