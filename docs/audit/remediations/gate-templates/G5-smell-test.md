# G5 — Smell-Test Clearance Certificate

**Candidate:** ________________________
**Observed |t|:** ________________________
**Observed Δ_IS (per-trade R):** ________________________

---

## When this certificate is required

Per `docs/institutional/edge-finding-playbook.md` §3:

> If top survivors have `|t| > 10` or `Δ_IS > ±0.6`, assume look-ahead until proven otherwise. Real edges on 20+ years of clean data produce `|t| 3-5`, not blowouts.

Applies to any candidate where `|t| > 7` OR `|Δ_IS| > 0.6R`.

**Known instance (2026-04-21):** PR #48 participation-shape with MNQ t=+9.59, MES t=+11.80, MGC t=+7.54 → triggers this tripwire.

## Required investigations

### Investigation 1 — Look-ahead mechanical check

- [ ] G1 timing-validity certificate attached and CLEAR
- [ ] No variable is computed from break-bar or later (or, if so, candidate has been reframed to post-break role per G1)
- [ ] Window computation uses canonical `pipeline.dst.orb_utc_window`
- [ ] Feature source-bar reviewed line-by-line in the candidate's script

Evidence:
```
$ grep -n "overnight\|break\|orb_\|rel_vol" <script>
<paste output>
```

### Investigation 2 — Scale-artifact check

Per failure-log 2026-04-20 entry: absolute-points filters on instruments that changed price scale over the IS window produce artificially inflated t-stats.

- [ ] Candidate uses NO absolute-points thresholds, OR
- [ ] Absolute thresholds have been passed through scale-stability audit (fire-rate-by-year + lift-by-year)

Evidence (if absolute thresholds present):
```
<per-year fire-rate + lift table>
```

### Investigation 3 — Effective-N check

A very high `|t|` can arise from effectively-correlated trials being counted as independent observations.

- [ ] Candidate's trade-level observations reviewed for temporal dependence (e.g., overlapping positions, intraday multi-leg clustering)
- [ ] If dependence found, effective-N recomputed via block bootstrap (block size matched to dependence horizon)

Evidence:
```
<block bootstrap output or statement of no dependence>
```

### Investigation 4 — Comparison to prior literature

Real edges at `|t| > 7` on 20+ years of clean futures data would be an extraordinary claim requiring extraordinary evidence. Does the candidate's mechanism have a published prior?

- [ ] Local literature extract (`docs/institutional/literature/________.md`) documents a prior effect of this magnitude on similar data
- [ ] OR: candidate is tagged as "extraordinary claim, requires replication on at least one independent market"

## Verdict

- [ ] CLEAR — all 4 investigations pass; high `|t|` is explained by genuine edge magnitude + literature prior
- [ ] SUSPECT — at least one investigation raises concern; candidate demoted to RESEARCH_SURVIVOR pending independent-market replication
- [ ] FAIL — look-ahead confirmed (Investigation 1 fails) OR scale artifact (Investigation 2 fails); candidate KILLED

## Failure disposition

FAIL → candidate cannot advance. Pre-reg must be rewritten with the identified failure mode corrected, and G5 rerun on fresh data.

## Literature citation

- `docs/institutional/edge-finding-playbook.md` §3
- Failure-log entry 2026-04-20 absolute-threshold scale audit
- Failure-log entry 2026-04-21 RULE 3.5 post-hoc rejection (if G5 is being applied as a post-hoc addition, document it explicitly)

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of this certificate: ________________
