# Codex Hook Parity

Claude hooks are canonical source material. Codex should not edit `.claude/hooks/`;
when behavior is needed in Codex, copy or adapt it into `.codex/hooks/` or
`scripts/infra/codex_*` and test that Codex-owned surface.

## Codex-Owned Hooks

- `.codex/hooks/session_start.py` - compact startup hints for Codex sessions
- `.codex/hooks/user_prompt_submit_grounding.py` - prompt-triggered routing and grounding hints

## Claude Hook Behaviors To Consider For Codex

- `.claude/hooks/_branch_state.py`
- `.claude/hooks/_context_state.py`
- `.claude/hooks/_crg_usage_log.py`
- `.claude/hooks/_memory_capture.py`
- `.claude/hooks/autopilot-tier-guard.py`
- `.claude/hooks/bias-grounding-guard.py`
- `.claude/hooks/branch-context.py`
- `.claude/hooks/branch-flip-guard.py`
- `.claude/hooks/completion-notify.py`
- `.claude/hooks/context-gauge.py`
- `.claude/hooks/data-first-guard.py`
- `.claude/hooks/discovery-loop-guard.py`
- `.claude/hooks/head-flip-guard.py`
- `.claude/hooks/intent-router.py`
- `.claude/hooks/judgment-review-nudge.py`
- `.claude/hooks/judgment-review-soft-block.py`
- `.claude/hooks/mcp-git-guard.py`
- `.claude/hooks/memory-capture-advisory.py`
- `.claude/hooks/memory-capture-sessionstart.py`
- `.claude/hooks/new-skill-eval-nudge.py`
- `.claude/hooks/plugin-router.py`
- `.claude/hooks/post-compact-reinject.py`
- `.claude/hooks/post-edit-pipeline.py`
- `.claude/hooks/post-edit-schema.py`
- `.claude/hooks/pre-edit-discovery-marker.py`
- `.claude/hooks/pre-edit-guard.py`
- `.claude/hooks/risk-tier-guard.py`
- `.claude/hooks/session-heartbeat.py`
- `.claude/hooks/session-start.py`
- `.claude/hooks/shared-state-commit-guard.py`
- `.claude/hooks/shell-canon-guard.py`
- `.claude/hooks/stage-awareness.py`
- `.claude/hooks/stage-closed-code-review-nudge.py`
- `.claude/hooks/stage-gate-guard.py`
- `.claude/hooks/subagent-budget-guard.py`
- `.claude/hooks/targeted-grounding-router.py`
- `.claude/hooks/worktree-destroy-guard.py`
- `.claude/hooks/worktree_guard.py`

## Adaptation Rule

Prefer executable Codex checks over copied prose. Good Codex destinations:

- `.codex/hooks/` for Codex runtime prompt/session hooks
- `scripts/infra/codex_local_env.py` for environment doctor/setup checks
- `scripts/infra/codex-project*.sh` for launcher-time enforcement
- `scripts/infra/codex-worktree.sh` for isolated mutating work
- `scripts/infra/codex_parity.py` for Claude-vs-Codex capability drift

If a Claude hook blocks live/capital/schema risk, Codex should either expose an
equivalent block in a Codex-owned surface or explicitly document why the launcher
or repo preflight already covers it.
