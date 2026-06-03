---
status: active
owner: joshdlees
last_reviewed: 2026-05-23
superseded_by: ""
---

# Slack Control Room Design

**Date:** 2026-05-23
**Scope class:** operator workflow / cross-tool coordination
**Verdict:** adopt, but only as an observer and coordination layer. Slack must not become a source of truth for canompx3 runtime state, trading decisions, research verdicts, or repo history.

## Decision

This fits canompx3 because the project already has multiple operator surfaces:

- Claude Code terminal / VS Code for repo edits and hook-enforced workflow.
- Codex for implementation, review, verification, stale-state detection, and WSL workflow checks.
- GitHub Actions for CI.
- `HANDOFF.md`, `docs/plans/`, stage files, runtime ledgers, and generated reports for durable state.

The gap is not a missing editor. The gap is situational awareness: runs finish, CI changes, decisions accumulate, and the operator has to reconstruct status from several places. A Slack control room can reduce that overhead if it stays narrow:

- announce what happened,
- link to the canonical artifact,
- summarize discussion into action items,
- never replace the canonical artifact.

## Audit Pass Findings

The first design was directionally correct but under-specified in five places:

1. It said Slack was non-authoritative, but did not define what to do when Slack and repo artifacts disagree.
2. It treated notification as mostly positive, but did not define silence rules. Without silence rules, a control room becomes another noisy inbox.
3. It named ChatGPT summarization, but did not guard against summary bias, recency bias, or over-weighting the loudest Slack thread.
4. It did not simulate bad cases: Slack outage, duplicate CI messages, a hook posting too much, or a Slack-only decision.
5. It did not define uninstall/rollback criteria if the integration adds noise or leakage risk.

This revision fixes those gaps by adding conflict handling, silence rules, bias controls, simulation checks, and rollback thresholds.

## Non-negotiable Boundary

Slack is not authoritative.

| Topic | Authoritative surface | Slack role |
|---|---|---|
| Project decisions | `docs/plans/`, `HANDOFF.md`, `docs/runtime/decision-ledger.md` | Announce and link |
| Claude/Codex edits | git commits, diffs, stage files, hook outputs | Notify status |
| CI result | GitHub Actions | Mirror notification |
| Trading/runtime truth | code, `gold.db`, runtime read models, `docs/runtime/*` generated artifacts | Link only, no copied live values unless stamped |
| Research verdicts | prereg/result docs, `chordia_audit_log.yaml`, validated DB surfaces | Announce verdict artifact |
| Secrets | `.env`, OS/user secret store, Slack app config | Never post |

Conflict rule:

- If Slack disagrees with repo/GitHub/runtime truth, the repo/GitHub/runtime artifact wins.
- If Slack contains a decision with no canonical artifact, the decision is incomplete.
- If ChatGPT summarizes a thread into an action item, the action item remains a draft until it is written into a repo artifact or GitHub issue/PR.

## Official Documentation Constraints

This design intentionally follows the narrowest interpretation of the official docs.

### OpenAI ChatGPT Slack app

Official OpenAI Help says the ChatGPT Slack app can search Slack messages/files the user already has access to, draft posts/replies, and summarize long conversations into action items. It also states:

- Slack installation is workspace-level and may already be installed by an admin or authorized user.
- Slack workspace installation must be allowed.
- The user needs a paid Slack plan and a Plus, Pro, Business, or Enterprise/Edu ChatGPT account.
- Enterprise/Edu admins must enable the Slack app and any required RBAC.
- Semantic search is not universal; it is available only for Slack customers with AI enabled on Business+ or Enterprise+.
- Enabling the app may broaden access pathways across connected ChatGPT apps.
- Requests/data may be transmitted to Slack and handled under Slack policies, so workspace policies matter.

canompx3 implication: do not design around semantic search as mandatory. Treat ChatGPT Slack results as access-scoped, policy-scoped, and incomplete unless checked against repo/GitHub truth.

### Claude Code hooks

Official Claude Code docs describe hooks as lifecycle handlers that receive JSON event context. Current Claude Code docs include command hooks and newer HTTP/MCP/prompt/agent handlers. The existing canompx3 repo hook stack is command-hook based.

canompx3 implication: Phase 1 uses a command hook wrapper (`slack-notify.py`) that posts to Slack. Do not start with direct HTTP hooks because env handling, redaction, local logging, and tests are clearer in the existing command-hook pattern.

### Slack incoming webhooks

Slack docs state incoming webhooks post JSON payloads to a unique URL tied to a single channel/user installation. Slack also says webhook URLs contain secrets, leaked URLs may be revoked, and incoming webhooks do not let the app delete a message after posting.

canompx3 implication: webhook URLs never enter git, logs, Slack messages, or screenshots. If a message is wrong, post a correction linking the canonical artifact; do not rely on deletion.

### GitHub Slack app

GitHub docs say default repository Slack notifications include events such as issues, pulls, default-branch commits, releases, and deployments. Reviews, workflows, comments, branches, discussions, and all-branch commits are opt-in. Workflow notifications use `/github subscribe owner/repo workflows` and may require extra permissions.

canompx3 implication: start with pulls and workflows only if the default subscription is too noisy. Use workflow filters before building custom CI webhooks.

## Channel Model

### `#canompx3-control`

Purpose: decision channel.

Allowed posts:

- explicit operator decisions,
- plan/spec approvals,
- deploy/no-deploy decisions,
- capital-class stop/go decisions,
- short action-item summaries produced from Slack threads.

Required shape:

- decision title,
- owner,
- canonical link/path,
- status: proposed, accepted, blocked, superseded,
- next action.

Forbidden:

- raw logs,
- long tool output,
- unstamped live metrics,
- secrets or broker/account details,
- discussion that claims to supersede repo docs without a matching repo artifact.

Silence rule:

- Do not post routine "no change" messages.
- Do not post repeated reminders for the same blocked decision more than once per day.
- Do not let ChatGPT auto-post decisions. Summaries should be requested by the operator or posted as explicitly draft action lists.

### `#canompx3-claude-runs`

Purpose: Claude Code run status.

Allowed posts:

- session start/stop,
- needs-input notifications,
- hook blocks,
- drift/test completion summaries,
- stage opened/closed notifications,
- commit created notifications.

Transport:

- Prefer a small Claude hook notifier that posts compact JSON-derived messages to Slack.
- Initial rollout should use `Notification` and `Stop` events only.
- Do not post every `PreToolUse` / `PostToolUse` event. That would create noise and leak local paths/tool chatter.

Required redaction:

- no `.env` values,
- no Slack webhook URL,
- no full command output unless explicitly whitelisted,
- no raw exception trace if it may include env or account context.

Silence rule:

- Post only state transitions: started, needs input, blocked, stopped with outcome, verification summary, or commit summary.
- Suppress repeated hook noise from the same stage within a cooldown window.
- Do not post every test target, file edit, or tool call.

### `#canompx3-prs`

Purpose: GitHub and CI updates.

Recommended setup:

- Use the GitHub Slack app for PR and workflow notifications.
- Subscribe only to this repo.
- Keep this channel mechanical: PR opened, review requested, CI failed, CI fixed, PR merged.
- Prefer GitHub app subscriptions over custom hooks for workflow runs.

Do not duplicate GitHub notifications through Claude hooks unless the GitHub app cannot cover the case.

Silence rule:

- Disable default-branch commit spam if it duplicates the operator's commit notifications.
- Subscribe to workflow notifications only after checking whether default PR notifications already include enough status.
- Keep reviews/comments enabled only if they produce useful decision context.

## ChatGPT In Slack

Adopt the ChatGPT Slack app for summarization and retrieval, not for direct repo mutation.

Use cases:

- summarize long `#canompx3-control` threads,
- extract decisions and action items,
- find prior Slack discussions by topic,
- produce a draft checklist for the operator to convert into a repo plan or handoff update.

Boundary:

- ChatGPT's Slack access is limited to messages, threads, channels, and files the installing user/app can access.
- A Slack summary is only a draft. Any durable decision must be written to `docs/plans/`, `HANDOFF.md`, or the relevant runtime decision ledger.
- Do not connect the Slack app to repo secrets, broker credentials, or local database artifacts.

Bias controls:

- Ask for evidence paths, not just conclusions.
- Ask the summary to separate decisions, claims, open questions, and guesses.
- Require "not found in Slack" instead of filling gaps from memory.
- Treat recent Slack activity as recency-biased. Before acting, check the matching repo artifact.
- Treat high-volume discussion as availability-biased. A noisy thread is not more authoritative than a quiet committed plan.
- Treat a polished ChatGPT summary as automation-biased. It still needs repo verification.

## Claude Code Integration

Claude Code remains the repo-editing surface. Slack receives status only.

Initial hook design:

- New script: `.claude/hooks/slack-notify.py`.
- Reads Claude hook event JSON from stdin.
- Exits 0 if no webhook env var is present.
- Posts only compact, event-specific messages.
- Uses per-channel webhook env vars or one webhook with a configured target channel.
- Fails open for notification errors. Slack outage must not block edits, tests, drift, commits, or shutdown.

Recommended env vars:

- `CANOMPX3_SLACK_CLAUDE_RUNS_WEBHOOK`
- `CANOMPX3_SLACK_CONTROL_WEBHOOK` only if later needed for explicit decision announcements
- `CANOMPX3_SLACK_NOTIFY_ENABLED=1`

Registration:

- Start in `.claude/settings.local.json` or user-level Claude settings, not committed repo settings, until noise and privacy are proven.
- If stabilized, move the hook script into the repo and register a no-op-by-default hook in `.claude/settings.json`.
- Never commit webhook URLs.

Logical constraint:

- A Slack notifier may fail open for notification errors, but it may not mask a failed test, drift violation, hook block, or commit failure as success.
- Message text must preserve uncertainty. If verification did not run, the post must say "verification not run" rather than implying green.
- Message text should link to the canonical artifact instead of copying volatile values.

## Approaches Considered

### A. Slack as full control plane

Slack channels become the main operator UI. All decisions, CI, Claude status, and run controls are routed through Slack.

Rejected. This creates a second source of truth, increases leakage risk, and encourages decisions outside the repo's evidence chain.

### B. Slack as observer/control room

Slack receives status, GitHub/CI notifications, and human-readable summaries. Durable truth remains in repo/GitHub/runtime artifacts.

Chosen. It reduces operator overhead without weakening the existing fail-closed repo workflow.

### C. No Slack integration

Keep all status in terminals, GitHub, and repo files.

Rejected for now. It preserves rigor but leaves the current coordination overhead untouched.

## Rollout Plan

### Phase 0 - Workspace setup, no repo mutation

1. Create the channels:
   - `#canompx3-control`
   - `#canompx3-claude-runs`
   - `#canompx3-prs`
2. Install/connect:
   - ChatGPT Slack app for search and summarization.
   - GitHub Slack app for repo PR/CI updates.
3. Add a pinned channel note in `#canompx3-control`:
   - Slack is not authoritative.
   - Decisions must link to a repo artifact.
   - Secrets and broker/account details are forbidden.

Acceptance:

- GitHub notifications appear in `#canompx3-prs`.
- ChatGPT can summarize a test thread into decisions/actions.
- The pinned `#canompx3-control` note states Slack is non-authoritative.
- The ChatGPT Slack app install path and account/plan prerequisites are confirmed.
- Semantic search is marked optional unless the Slack workspace plan supports it.
- GitHub workflow notifications are explicitly subscribed or explicitly deferred.
- No repo hook/config files changed.

### Phase 1 - Claude status notifier, local only

1. Create `.claude/hooks/slack-notify.py` with redaction and fail-open behavior.
2. Register it locally for `Notification` and `Stop` events.
3. Configure `CANOMPX3_SLACK_CLAUDE_RUNS_WEBHOOK` outside the repo.
4. Run two manual hook-event fixtures:
   - stop event with no edits,
   - notification event requiring input.

Acceptance:

- Missing env var exits 0 with no post.
- Valid webhook posts one compact message to `#canompx3-claude-runs`.
- Invalid/revoked/archived-channel webhook exits 0 and records only local stderr/debug output.
- No secret or raw env value appears in any message.
- A failed test fixture posts "failed" or "blocked", never "done".
- A no-change stop event stays silent unless Claude is explicitly waiting for input.
- A malformed hook JSON fixture exits 0 and posts nothing.
- A Windows path / WSL path fixture redacts or shortens local machine paths before posting.

### Phase 2 - Repo-backed notifier, still no authority change

1. Add tests for redaction, message shape, and fail-open behavior.
2. Add docs for local setup.
3. Register a no-op-by-default hook in `.claude/settings.json` only after Phase 1 noise is acceptable.

Acceptance:

- `python -m pytest .claude/hooks/tests/test_slack_notify.py -q` passes.
- `python pipeline/check_drift.py` passes.
- Hook remains disabled unless `CANOMPX3_SLACK_NOTIFY_ENABLED=1`.
- Redaction tests cover webhook URL, `.env`-style keys, bearer tokens, and broker/account-looking strings.
- Noise tests prove repeated identical stop events are suppressed by cooldown.
- Unit tests cover Slack webhook error classes: invalid token, no service, channel archived, malformed payload, and transient non-200 response.
- Unit tests prove the notifier never returns exit code 2.

### Phase 3 - Decision summaries

1. Use ChatGPT in Slack to summarize `#canompx3-control` threads into:
   - decision,
   - canonical repo artifact,
   - open questions,
   - next action.
2. Manually paste or commit the resulting durable item into `HANDOFF.md`, `docs/plans/`, or `docs/runtime/decision-ledger.md`.

Acceptance:

- Every Slack-derived decision has a canonical repo path.
- No decision exists only in Slack after the session ends.
- A disagreement between Slack summary and repo artifact is resolved in favor of the repo artifact.

### Phase 4 - Review and rollback gate

After one week of use, review the integration.

Keep it only if:

- useful posts outnumber noise,
- no secrets or sensitive account details were posted,
- at least one Slack summary became a cleaner repo artifact or action item,
- no repo decision lived only in Slack,
- Claude/Codex workflow stayed faster or clearer.

Rollback if:

- Slack creates extra review burden,
- notifications are routinely ignored,
- any sensitive content leaks,
- operators start treating Slack summaries as source-of-truth,
- hook failures become confusing during real repo work.

## Implementation Notes

Message shape for Claude run posts:

```text
canompx3 Claude Code: <event>
status: <done | needs-input | blocked | failed>
branch/worktree: <short safe label>
canonical: <repo path or GitHub URL if available>
next: <one line>
```

For failures, include only the failure class and canonical local artifact path, not raw logs by default.

Implementation defaults:

- Use `urllib.request` or the standard library first; no new dependency is needed for a single webhook POST.
- Default timeout: 2 seconds.
- Default max message size: 3,000 characters, with hard truncation and canonical link/path retained.
- Default cooldown key: event type + stage/branch + outcome.
- Default local debug path: `.claude/hooks/state/slack-notify-debug.jsonl`, ignored by git through the existing hook-state ignore rule.
- Exit code: always 0 for notifier failures. The notifier is observability, not a gate.

Not allowed in implementation:

- posting raw transcript paths,
- posting full local absolute paths unless shortened to a repo-relative path,
- posting raw command output by default,
- posting webhook URLs or token-like strings,
- adding a committed `.claude/settings.local.json`,
- touching `pipeline/`, `trading_app/`, `gold.db`, broker config, allocator/profile files, or live runtime state for this integration.

## Blast Radius

| Phase | Files/systems touched | Explicitly not touched | Risk |
|---|---|---|---|
| Current design pass | `docs/plans/active/2026-05/2026-05-23-slack-control-room-design.md`, `HANDOFF.md` | code, hooks, settings, Slack/GitHub app config | documentation drift only |
| Phase 0 | Slack channels/apps, GitHub Slack subscription | repo files, secrets, trading runtime | external account/app permission scope |
| Phase 1 | local env vars, local/user Claude settings, optional local hook script | committed `.claude/settings.json`, repo code, CI | local notification noise/leakage |
| Phase 2 | `.claude/hooks/slack-notify.py`, hook tests, setup doc, optional no-op hook registration | `pipeline/`, `trading_app/`, `gold.db`, `docs/runtime/lane_allocation.json`, broker/live files | repo hook behavior if misregistered |
| Phase 3 | Slack summaries plus repo handoff/plan/decision-ledger updates | direct deploy or allocator mutation | source-of-truth confusion |
| Phase 4 | disable hooks/subscriptions if review fails | canonical artifacts | rollback/admin cleanup |

Capital-class blast radius: none if this plan is followed. Any future request to let Slack trigger deploys, lane changes, broker actions, allocator writes, or live trading controls is a new capital-class design and must not be treated as an extension of this plan.

## Simulation Matrix

| Scenario | Expected Slack behavior | Canonical follow-up |
|---|---|---|
| Claude finishes a doc-only plan update | One stop summary in `#canompx3-claude-runs` if enabled | Link to plan path and mention verification run |
| Claude is blocked by a hook | One blocked message with blocker class, no raw dump | Operator reads local terminal/hook output |
| CI fails after PR opened | GitHub app posts to `#canompx3-prs` | Fix from GitHub Actions logs and repo tests |
| Slack webhook is missing | No post, hook exits 0 | Local workflow continues |
| Slack webhook returns error | No block, local debug only | Rotate or repair webhook later |
| ChatGPT summarizes a control thread | Draft decisions/actions only | Human writes durable repo artifact |
| Slack thread says deploy, repo plan says blocked | Repo plan wins | Update Slack with canonical blocked path |
| ChatGPT cannot find a prior decision | It says not found | Search repo docs/GitHub before deciding |
| Duplicate CI and Claude messages appear | Disable one source | Keep GitHub app as CI authority |
| A message would include raw logs or account data | Suppress or redact | Link local artifact instead |
| ChatGPT app is unavailable because of Slack/ChatGPT plan limits | Mark Phase 0 blocked/degraded | Continue with GitHub + Claude notifications only |
| Semantic Slack search is unavailable | Use keyword search only | Do not promise semantic retrieval |
| ChatGPT lacks access to a private channel | Summary says access-limited/not found | Check the repo/GitHub artifact directly |
| Slack retention has removed old messages | Summary says not found | Search committed docs and git history |
| GitHub workflow subscription asks for extra permissions | Pause and review scope | Prefer filtered workflow subscription over custom CI hook |
| Slack webhook URL leaks | Revoke/rotate webhook immediately | Audit repo/logs/Slack for exposure |
| Wrong message is posted | Post correction with canonical link | Do not rely on incoming webhook deletion |
| Claude hook environment lacks webhook env var | Stay silent | Fix local settings/env, not repo code |
| Stop hook fires after an interrupted session | Post nothing unless event proves a completed state | Operator reads terminal state |
| Detached worktree posts a status | Include safe worktree label, not a full machine path | Link canonical branch/commit if available |
| Two agents post about same stage | Cooldown/dedupe suppresses repeats | HANDOFF remains the shared baton |

## Risks

- Noise: too many lifecycle events can make Slack useless. Mitigation: start with `Notification` and `Stop` only, add cooldowns, and review after one week.
- Secret leakage: webhook URLs and env values must never be committed or printed. Mitigation: env-only config, redaction tests, `.env` remains ignored and read-denied.
- Source-of-truth drift: Slack threads can look like decisions. Mitigation: pinned rule plus required canonical link/path.
- External dependency: Slack, OpenAI, or GitHub app outage must not block repo work. Mitigation: all notification failures fail open.
- Access scope: ChatGPT summaries only see Slack content accessible to the connected user/app. Mitigation: treat summaries as convenience, not proof.
- Summary bias: ChatGPT may over-summarize, flatten dissent, or miss quiet blockers. Mitigation: require decisions/claims/open questions/guesses separation and repo verification.
- Automation bias: a clean Slack summary can look more complete than the underlying evidence. Mitigation: require canonical artifact links before action.
- Recency bias: latest Slack thread may override older committed constraints in the operator's head. Mitigation: repo artifacts win on conflict.
- Security scope creep: installing apps may expand Slack/GitHub/OpenAI access paths. Mitigation: admin review, MFA, minimal channels, and one-week rollback gate.
- Message permanence: incoming webhook messages cannot be deleted by the webhook. Mitigation: post corrections, keep payloads minimal, and never include sensitive details.
- Permission surprises: GitHub workflows and ChatGPT Slack may ask for more access than expected. Mitigation: review scopes before enabling, and defer features that require broad write/admin access.
- Plan-limit surprise: some desired ChatGPT Slack features may not be available on the current Slack/ChatGPT plan. Mitigation: design degrades to keyword search and manual summaries.

## Official Source Grounding

- OpenAI Help: [ChatGPT app in Slack](https://help.openai.com/en/articles/12462158-chatgpt-app-in-slack) can search accessible Slack messages/files and summarize conversations into action items, with account/plan/admin/RBAC and policy caveats.
- Claude Code Docs: [hooks reference](https://code.claude.com/docs/en/hooks) documents lifecycle events, command/HTTP/MCP/prompt/agent handlers, JSON input/output, and hook locations. The repo default remains command-hook based.
- Anthropic Docs: [Claude Code settings](https://docs.anthropic.com/en/docs/claude-code/settings) documents shared project settings vs local project settings and deny rules for sensitive files.
- Slack Docs: [incoming webhooks](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/) post JSON payloads to channel-specific secret URLs, cannot delete posted messages through the webhook, and return explicit error classes.
- GitHub Docs: [custom notifications for GitHub in Slack](https://docs.github.com/en/enterprise-server%403.17/integrations/how-tos/slack/customize-notifications) document default events, opt-in `workflows`, filters, and extra permissions for workflow notifications.

## Next Concrete Step

Do Phase 0 first. After the channels and apps are connected, implement Phase 1 as a local-only Claude hook and test it against fixture hook events before registering it in committed settings.
