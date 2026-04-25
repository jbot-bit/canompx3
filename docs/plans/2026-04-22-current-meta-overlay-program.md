# Current-Meta Overlay Program

**Created:** 2026-04-22  
**Owner:** canompx3  
**Status:** LOCKED — docs-only design package  
**Authority:** `RESEARCH_RULES.md`, `docs/institutional/pre_registered_criteria.md`, `docs/institutional/mechanism_priors.md`, `.claude/rules/research-truth-protocol.md`, `.claude/rules/backtesting-methodology.md`

---

## Executive decision

The project should **not** pivot into generic rolling ML, HMM/GMM regime engines, or black-box meta-labeling on the current ORB feature stack.

The correct move is a **three-track current-meta program**:

1. **MNQ-first structural geometry overlay** on canonical pre-trade fields
2. **Execution-safe participation overlay** on adjacent instruments
3. **Separate microstructure Gate 0** as a future system-upgrade program

This keeps the user's core idea alive:

> Build a machine that adapts to what is working now

But it does it in a way that is still truthful to repo evidence:

- **static-first**
- **mechanism-first**
- **execution-safe**
- **no OOS peeking**
- **no dead-path ML resurrection**

---

## What is grounded already

### 1. Broad pooled ORB ML is still dead

This repo already killed the broad form of the idea:

- `docs/institutional/mechanism_priors.md` § 7 lists:
  - `ML on ORB outcomes — DEAD`
  - `Regime-conditional rolling window discovery — NO-GO`
  - `Vol-regime adaptive parameter switching — DEAD`
- `RESEARCH_RULES.md` and `pre_registered_criteria.md` still bind on MinBTL, OOS discipline, and anti-smoothing logic.

So the path is **not**:

- weekly XGBoost retraining
- black-box take/not-take scoring on pooled ORB outcomes
- “adaptive” ML as a blanket justification for reopening killed lines

### 2. Structural geometry is likely under-encoded

The repo already has repeated evidence that prior-day and session-boundary interaction matters:

- `docs/institutional/mechanism_priors.md` § 2
- `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`
- canonical fields already exist in `daily_features` for:
  - `prev_day_high`
  - `prev_day_low`
  - `prev_day_close`
  - `overnight_high`
  - `overnight_low`

What is missing is a cleaner **distance / co-location / open-air** encoding, not a brand-new mechanism.

### 3. Participation state is real, but the old role was wrong

Committed repo evidence already says `rel_vol` is interesting:

- `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`
- `docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition-is-only-quantile.md`
- `docs/audit/results/2026-04-21-recent-claims-skeptical-reaudit-v1.md`

But the old framing was wrong:

- `trading_app/config.py` explicitly marks `VolumeFilter` as **E2-excluded**
- `rel_vol` resolves at `BREAK_DETECTED`, not before the E2 resting stop order
- `docs/plans/2026-03-02-live-trading-infrastructure.md` still blocks live `E3`

So the participation line must be retested as:

- **E1-only**
- post-break
- static-first

not as an E2 overlay and not as a generic deployable sizer claim.

### 4. True microstructure is a different program

If the desk wants to model “what other algorithms are doing,” that is a **real** path, but it is not the current ORB truth engine.

It would require:

- new market-data schemas
- new truth tables
- event-time features
- a new Gate 0 falsification question

It is **not** an overlay on `bars_1m -> daily_features -> orb_outcomes`.

---

## Program architecture

## Track A — Structural Geometry Overlay (Primary / MNQ-first)

**Objective:** encode whether a breakout is:

- breaking **through** a visible level
- breaking **into** a nearby wall
- operating in **open air**
- forming **inside** prior-day structure

**Canonical inputs already available:**

- `daily_features.prev_day_high`
- `daily_features.prev_day_low`
- `daily_features.prev_day_close`
- `daily_features.overnight_high`
- `daily_features.overnight_low`
- `daily_features.orb_{session}_high`
- `daily_features.orb_{session}_low`
- `daily_features.orb_{session}_break_dir`

**Why this is the primary track:**

- `topstep_50k_mnq_auto` is still the best live-adjacent book in the repo
- committed evidence on `main` already shows multiple strong MNQ level-interaction cells:
  - `MNQ US_DATA_1000 O5 RR1.0 long F5_BELOW_PDL`
  - `MNQ COMEX_SETTLE O5 RR1.0 long F6_INSIDE_PDR`
  - `MNQ NYSE_CLOSE O5 RR1.5 long F3_NEAR_PIVOT_15`
- the likely missing value is cleaner geometry encoding, not another generic overlay family

**Immediate build target:**

- design-only first
- no production feature-build yet
- no prereg yet on a new synthetic feature column

**Why no immediate MNQ prereg yet:** two reasons bind here.

1. The feature family is still being specified. The repo’s prereg gate forbids pretending a future `daily_features` column already exists.
2. Several exact MNQ anchor cells have already been inspected in committed research and in fresh read-only review this session. The honest next step is the **feature contract**, then a fresh prereg on untouched exact scope — not retroactive registration after another look.

**Target roles, in order:**

1. `R1` binary avoid/take
2. `R3` size-down / size-normal / size-up
3. constrained adaptive overlay only if `R1/R3` static value survives

**Explicit non-goals:**

- no hardcoded “always skip if clearance < 1R” production rule
- no direct jump to target modifiers
- no direct jump to rolling ML

## Track B — Execution-Safe Participation Overlay (Secondary / adjacent)

**Objective:** re-test the `rel_vol` participation line in the only execution role that is currently safe:

- `E1`
- `O5`
- `RR1.5`
- `MES` first

This track is immediate and executable, because the feature already exists in canonical truth.

**Immediate artifact in this package:**

- [2026-04-22-mes-e1-rel-vol-family-v1.yaml](../audit/hypotheses/2026-04-22-mes-e1-rel-vol-family-v1.yaml)

**Why this is secondary, not primary:**

- committed repo evidence already shows multiple `MES rel_vol_HIGH_Q3` survivors on O5
- `MES` gives a 20-cell family (`10 sessions × 2 directions`) that stays inside a clean MinBTL envelope
- `MGC` remains alive as a follow-on, but a broad 20-cell MGC family is too loose relative to its native horizon unless we narrow the session scope first
- `MNQ` is not being abandoned; it simply is not the honest first prereg on this specific participation reroute because the strongest repo-grounded opportunity on MNQ is still structural geometry, not post-break participation

**Target roles, in order:**

1. `R1` top-quintile confirmation gate
2. `R3` three-band size map only if the binary gate survives
3. no rolling adaptation until the static family proves incremental value

## Track C — Constrained Current-Meta Overlay

This is the repo-compatible version of the user’s “learn what is working now” idea.

It is **gated** behind Tracks A/B.

It is allowed only if:

- a new structural feature class proves value statically, or
- the execution-safe participation family proves value statically

**Allowed model shapes:**

- monotone logistic
- isotonic scorer
- banded ordinal score
- at most 2–4 frozen features

**Required comparison:**

1. null / uniform
2. static overlay
3. rolling constrained overlay

If rolling does not beat static, the rolling layer dies.

**Banned shapes:**

- random forest
- xgboost
- broad pooled classifier
- weekly unconstrained retraining
- “adaptive” black-box sizing because it sounds institutional

## Track D — Microstructure Gate 0

This is the correct home for the long-horizon system upgrade.

It is **not** an overlay on the current research stack.

**Question:**

Can market-depth / queue / trade-sign imbalance features predict session-open follow-through at a horizon we can still monetize after our costs?

**Required before any model work:**

- data-layer design
- schema design
- event-time truth tables
- one falsifiable Gate 0 spec

**No-go today:**

- no pretending `bars_1m` is enough to model MBO behavior
- no “CVD proxy” language unless the data layer exists
- no IRL/Hawkes/DeepLOB implementation before Gate 0 proves monetizable signal

---

## What this program is trying to maximize

The real objective is not “use ML.”

It is:

- find the highest-quality trades now
- identify when the book is in a better or worse state now
- express that state through safe overlays
- only add adaptation if it beats a static benchmark

That means the sequencing is:

1. **discover real state variables**
2. **prove static utility**
3. **then test whether adaptation improves on static**

not the other way round.

---

## Immediate queue

### 1. Ship this package

- plan doc
- immediate `MES E1 rel_vol` prereg

### 2. Write the MNQ boundary-geometry feature contract

This is the highest-EV primary-book artifact after this package lands.

It should define:

- distance-to-overhead in `R`
- distance-to-support in `R`
- co-location state
- inside-range / outside-range geometry
- session-safety matrix for overnight fields

### 3. Run the `MES E1 rel_vol` family exactly as pre-registered

- no MNQ
- no E2
- no E3
- no MGC broad family in the same run

### 4. Open Microstructure Gate 0 only after the above

No schema work before the current canonical-overlay paths are clarified.

---

## Kill / continue table

| Track | Continue if | Kill if |
|---|---|---|
| Structural geometry | feature contract yields a small exact preregable test surface | effect cannot be stated on existing canonical columns without post-hoc lane-picking |
| MES E1 participation | one or more family cells survive with positive on-signal mean and OOS direction | zero survivors and family median delta <= 0 |
| Constrained current-meta overlay | rolling constrained overlay beats static on a surviving feature class | rolling adds no incremental value over static |
| Microstructure Gate 0 | data upgrade proves monetizable signal at our horizon/cost | signal exists only at horizons we cannot trade profitably |

---

## Autonomous recommendation

Do **not** build a generic rolling “brain.”

Build the smallest truthful machine that can later earn the right to become adaptive:

1. **MNQ-first static structural geometry**
2. **adjacent static execution-safe participation**
3. **constrained rolling overlay only if 1 or 2 survives**
4. **true microstructure as a separate upgrade program**

That is the best non-pigeonholed path for this repo.
