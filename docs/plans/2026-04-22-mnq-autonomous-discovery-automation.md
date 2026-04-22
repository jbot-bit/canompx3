# MNQ Autonomous Discovery Automation

Date: 2026-04-22
Branch: `wt-codex-mnq-hiroi-scan`

## Purpose

Convert the recovered board-driven discovery workflow into a durable Codex
automation path that can widen across alive MNQ mechanism classes without
reopening broad dead search space or collapsing back into exact-cell tunnel
vision.

This automation is for the alive MNQ discovery program that was active on
2026-04-22 and should remain broad-but-bounded:

1. refresh broad and narrow read-only boards
2. pick the next bounded candidate inside those boards
3. run the cheap gate
4. advance only if discovery write + validator are justified
5. park or kill weak branches explicitly

## Design Principles

- Deterministic first:
  - repo scripts refresh evidence
  - the model does not recreate the board logic in prose
- Worktree isolated:
  - one dedicated discovery branch and worktree
  - one coherent Codex thread resumed in place
- Small durable prompts:
  - method belongs in repo docs and a skill
  - the runtime prompt only carries turn-local context
- Machine-checkable loop output:
  - Codex returns JSON only
  - the runner validates the JSON before acting on it
- No blind automation:
  - commits and PRs are optional outputs, never implied
  - discovery writes still require the cheap gate and repo-native verification

## Canonical Loop

### 1. Refresh the tiered board stack

Run the boards in tiers:

Tier 0: broad alive mechanism maps

- `research/mnq_unfiltered_baseline_cross_family_v1.py`
- `research/mnq_live_context_overlays_v1.py`

Tier 1: bounded candidate surface

- `research/mnq_layered_candidate_board_v1.py`

Tier 2: mechanism-family board

- `research/mnq_prior_day_family_board_v1.py`

Tier 3: transfer board

- `research/mnq_geometry_transfer_board_v1.py`

These are deterministic read-only evidence builders. Tier 0 prevents tunnel
vision. Tiers 1-3 convert that breadth into bounded next moves.

### 2. Reconstruct the queue

Primary queue docs:

- `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
- `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`
- `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md`
- `docs/plans/2026-04-22-mnq-usdata1000-geometry-family-register.md`
- the board outputs above

This is where the model regrounds the current alive mechanisms and excludes
already-solved or already-paused branches.

### 3. Choose one bounded next move

Allowed next moves:

- write a new narrow prereg
- run `research/phase4_candidate_precheck.py` on one exact hypothesis
- run a narrow discovery write and validator pass when the cheap gate is clean
- write a result note that parks or kills a path
- harden the workflow docs or runner if a real process gap is found

Disallowed next moves:

- reopen dead ML or dead pair-stacking stories
- broad random family rescans
- threshold shopping after seeing OOS
- multiple simultaneous bridge writes from one loop iteration
- treating Tier 0 route maps as direct promotion authority

### 4. Verify before durable action

If files changed:

- run the smallest repo-native verification that matches the blast radius
- summarize verification in the loop JSON
- only then recommend commit/PR

### 5. Continue or stop explicitly

Every iteration must say one of:

- continue running
- no honest move
- parked
- killed

Silently drifting the queue is not allowed.

## Role Split

### Deterministic layer

- `scripts/tools/run_mnq_discovery_board_stack.sh`
- board scripts in `research/`
- `research/phase4_candidate_precheck.py`
- repo-native tests and validators

### Agentic layer

Codex:

- interprets the latest bounded evidence
- chooses the next honest narrow move
- writes preregs, result docs, or narrow runner changes
- decides whether to continue, park, or kill

## Runtime Contract

The headless runner must:

- use the dedicated `wt-codex-mnq-hiroi-scan` worktree
- acquire a local lock so only one discovery loop runs on that worktree
- refresh the board stack before each iteration
- resume the same Codex thread when available
- require JSON-only output
- validate the JSON locally before reading fields
- persist loop state on disk for crash recovery and repetition detection
- reuse the shared repo `.venv-wsl` if the worktree itself does not have one

## Why This Is Better

- It preserves the broad-but-bounded lens that produced real outcomes tonight.
- It automates the stable parts first, per current Codex best practice.
- It avoids giant prompts and replaces them with repo truth.
- It does not confuse a board refresh with a promotion claim.
- It adds file-backed loop state, lock discipline, and fail-closed repetition
  handling instead of trusting chat context.
