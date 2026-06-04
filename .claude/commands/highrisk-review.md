---
description: Adversarial Codex review of shipped work vs origin/main — hunts over-claims, fake DONE/PASS, silent failures, relaxed capital gates. Runs in an isolated worktree (never touches main).
argument-hint: '[--wait|--background] [--base <ref>] [extra focus ...]'
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Bash(node:*), Bash(git:*), AskUserQuestion
---

Run an **adversarial, anti-over-claim** Codex review of everything shipped on this
branch versus a base ref, through the shared Codex plugin runtime.

Purpose (load-bearing): this is the antidote to "we patted ourselves on the back
too much." Codex is a genuinely independent model — a fresh brain that did not
write our claims — so it will not rubber-stamp this session's own work. We point
it at the **committed changeset** (not the working tree) so it reviews what
actually landed.

Raw slash-command arguments:
`$ARGUMENTS`

## Core constraints

- **Review-only.** Do not fix issues, apply patches, edit files, or imply you are
  about to. Your only job is to run the review and return Codex's output verbatim.
- **Never touch `main`.** The review MUST run from an isolated worktree checked out
  off the base ref (default `origin/main`). If invoked from the main checkout,
  create/reuse a dedicated review worktree first (see Worktree setup).
- **Default scope is the shipped changeset**, i.e. `--scope branch --base <ref>`
  (default `<ref>` = `origin/main`). This is the whole point — do NOT fall back to
  working-tree scope, which would only review uncommitted noise.

## Review log — incremental, no re-litigating (the memory layer)

State file: `docs/runtime/highrisk_review_log.yaml` (create on first run). This is
what makes repeat runs cheap and stops Codex re-flagging settled findings.

Schema:
```yaml
last_reviewed:
  base_ref: origin/main          # what we diffed against last time
  base_sha: <40-hex>             # resolved SHA of base_ref at that run
  head_sha: <40-hex>             # HEAD that was reviewed
  ran_at: "YYYY-MM-DDTHH:MMZ"    # stamp passed in by the caller (no Date.now in-script)
  verdict: needs-attention|approve
findings:                        # disposition memory, append-only
  - id: <file>:<line_start>      # stable key (file + first line)
    gist: "<one-line what>"
    status: open|fixed|accepted-risk|wontfix
    noted_at: "YYYY-MM-DDTHH:MMZ"
    note: "<why, if accepted/wontfix>"
```

Run protocol:
1. **Read the log first.** If `last_reviewed.head_sha` is an ancestor of current
   HEAD (`git merge-base --is-ancestor <head_sha> HEAD`), default the review base
   to that SHA — review ONLY commits landed since (incremental; biggest token
   saver). If the user passed an explicit `--base`, that wins. If the log is
   missing or the watermark is unreachable (rebased away), fall back to
   `origin/main` and say so.
2. **Suppress settled findings.** Collect log entries with status
   `fixed`/`accepted-risk`/`wontfix` and append to the focus text:
   "Already-triaged (do NOT re-raise unless materially changed): <id — gist>; …".
   This is the anti-rehash guard.
3. **After the review returns**, update the log: set `last_reviewed` to the just-
   reviewed base/head/verdict + timestamp; for each NEW Codex finding not already
   present, append a `findings` entry with `status: open`. Do not auto-close
   anything — the operator decides disposition. Keep edits minimal; never delete
   history.
4. Stamp timestamps from a value passed in (e.g. `date -u +%Y-%m-%dT%H:%MZ` run in
   Bash), never invented — `Date.now()` is unavailable to scripts here anyway.

Note: this file is generated review state. It is fine to commit it (durable across
`/clear` and worktrees) — keep entries one-line and honest, same hygiene as other
`docs/runtime/` ledgers. If a run finds nothing new since the watermark, say
"nothing new shipped since <head_sha> — skipping" and do not call Codex at all
(zero-token fast path).

## Base ref resolution

- Use `--base <ref>` if the user supplied it; else use the incremental watermark
  from the review log (step 1 above); else default to `origin/main`.
- Run `git fetch origin --quiet` once so the base is current before diffing.
- If `git rev-parse --verify <ref>` fails, STOP and tell the user the base ref is
  unknown — do not silently widen scope.

## Worktree setup (isolation — efficiency-aware)

1. Determine the canonical root: `git -C C:/Users/joshd/canompx3 rev-parse --show-toplevel`.
2. Reuse an existing review worktree if one already exists (cheap — avoids a
   full 6k-file checkout):
   `git worktree list --porcelain` → look for a path matching
   `*canompx3-codex-highrisk-review` or a branch `review/codex-highrisk*`.
3. Only if none exists, create one off the base ref:
   `git -C <canonical-root> worktree add -b review/codex-highrisk-<YYYY-MM-DD> ../canompx3-codex-highrisk-review <base-ref>`
   (the date suffix avoids branch-name collisions across days; if the branch
   already exists, reuse it with `git worktree add <path> review/codex-highrisk-<date>`).
4. Run Codex from inside that worktree path (pass it as the companion's `cwd` by
   running `node` with that directory as the working dir).
5. **Sync the worktree to the SHA being reviewed.** A reused worktree may sit on a
   stale HEAD, which would make `<base>...HEAD` diff the wrong range. Before
   diffing, fast-forward it to the canonical HEAD under review:
   `git -C <worktree> fetch origin --quiet` then
   `git -C <worktree> reset --hard <head-sha-to-review>` (the worktree is a
   throwaway review checkout off a `review/*` branch — a hard reset here is safe
   and never touches main; do NOT do this if the worktree path resolves to the
   main checkout — abort instead).

## Edge cases (cover every one — fail toward a correct review, never a silent wrong one)

| Case | Required behavior |
|---|---|
| **First run, no log** | No watermark → base = `origin/main`. Normal full review. Create the log after. |
| **Watermark unreachable** (squash/rebase dropped `head_sha`) | `merge-base --is-ancestor` fails → fall back to `origin/main`, STATE this in the reply ("watermark <sha> rebased away; reviewing full range"). Never silently review nothing. |
| **base_ref advanced** | `origin/main` moved past last review → that's expected; incremental base is the prior `head_sha`, not the old base_sha. Diff `<prior_head>...HEAD`. |
| **Empty range** (`<base>...HEAD` empty) | Nothing shipped since base. Say "nothing to review since <base>" and STOP — do NOT call Codex (zero-token). For the incremental path this is the "nothing new" fast path. |
| **Detached HEAD** | SHAs still work; merge-base/diff are SHA-based. Drop branch-name language from the reply; proceed on SHAs. |
| **Stale review worktree** | Sync per Worktree-setup step 5 before diffing. |
| **Concurrent runs / log race** | The log lives under `docs/runtime/`, already watched by `shared-state-commit-guard.py` (3-check protocol on commit). Do NOT add a new lock. Before writing the log, re-read it fresh; append only your NEW findings; if a peer advanced it mid-run, merge by union of `findings` (keep both), never overwrite. On commit, honor the shared-state guard. |
| **Malformed / unreadable log YAML** | Fail OPEN: treat as no-log (full review vs `origin/main`), warn once, and write a fresh well-formed log after (do not clobber silently — rename the bad file to `*.corrupt-<head_sha>` so history isn't lost). |
| **Malformed / empty Codex output** | Do NOT write any `findings` to the log from unparseable output. Record the run's `last_reviewed` watermark + `verdict: error`, surface the raw Codex stderr/stdout to the user, and stop. Never invent findings to fill the log. |
| **Offline / `git fetch` fails** | Warn "could not refresh origin; using local `origin/main`" and proceed with the local ref. Do not abort — local refs are still a valid base. |
| **Worktree create fails** (disk, dirty, branch exists) | Surface the exact git error and STOP. NEVER fall back to running the review from the main checkout — isolation is non-negotiable. If the branch exists, reuse it (attach a worktree to the existing `review/*` branch). |
| **Huge / bulk-path diff** | Before invoking Codex, check `git -C <worktree> diff --shortstat <base>...HEAD`. If it spans known-bulk paths (`artifacts/`, generated data, vendored dirs) or > ~2000 changed lines, exclude bulk paths via pathspec (`-- . ':(exclude)artifacts/'`) OR warn the user and recommend a tighter `--base`. Token blowout on generated files is never worth it. |

## Token & rigor framing (the reusable value)

Always append this anti-over-claim focus to whatever extra focus the user passed.
It steers Codex toward high-cost surfaces and away from style/naming (the
adversarial template already bars cosmetic findings at its finding bar), so tokens
go to what matters:

> Focus on whether the SHIPPED work is actually as done/safe as its own commit
> messages, stage files, and HANDOFF claim. Specifically hunt: (1) "DONE"/"PASS"/
> "verified"/"clears the gate" claims not supported by the diff or by runnable
> evidence; (2) drift-check or gate verdicts whose result depends on uncacheable
> inputs (DB reads, git history, path existence) but are cached/treated as static
> — a stale PASS on a blocking gate; (3) silent failures, swallowed exceptions,
> band-aids over canonical sources (pipeline/dst.py, cost_model.py,
> asset_configs.py, paths.py, holdout_policy.py) instead of fixing upstream;
> (4) capital-path gates (account survival, prop caps, telemetry/live preflight,
> bracket/fill parity) that were relaxed, waived, or model-diverged from the live
> engine; (5) prop contract caps leaking into self_funded sizing. Prefer one
> defensible no-ship finding over many weak ones. If the work is genuinely sound,
> say so plainly — do not manufacture findings.

## Literature grounding (ALWAYS-ON for capital / statistical findings)

High-risk findings about sizing, drawdown, multiplicity, overfitting, Sharpe
inflation, or holdout discipline must be checkable against this repo's curated
literature — not Codex's training memory. Per `.claude/rules/targeted-grounding.md`
(`/resource` / `/lit` route) and the Local Academic Grounding Rule.

Grounding protocol (the orchestrator does this — Codex cannot read these files):

1. **Verify tooling first (honesty gate).** Run
   `python scripts/tools/check_pdf_tooling.py` and
   `python scripts/tools/check_literature_coverage.py`. If PDF text extraction is
   unavailable, say so — do NOT imply any raw-PDF was read.
2. **Map finding → extract.** For each capital/statistical finding Codex returns,
   look up the relevant curated extract in `docs/institutional/literature/` via
   `resources/INDEX.md`. Canonical anchors:
   - sizing / vol-targeting / Kelly → `carver_2015_volatility_targeting_position_sizing.md`, `carver_2015_ch12_speed_and_size.md`
   - Sharpe inflation / deflated Sharpe / sample selection → `bailey_lopez_de_prado_2014_deflated_sharpe.md`, `bailey_lopezdeprado_2014_dsr_sample_selection.md`
   - backtest overfitting / pseudo-math → `bailey_et_al_2013_pseudo_mathematics.md`
   - multiplicity / many-trials → `benjamini_hochberg_1995_fdr.md` (and Harvey-Liu when present)
   - data snooping → `aronson_2007_ebta_data_snooping.md`
   - regime → `chan_2008_ch7_regime_switching.md`
   (Enumerate live: `ls docs/institutional/literature/*.md` — the set grows.)
3. **Annotate each grounded finding** with the extract it's supported by
   (`[grounded: <extract>.md]`). The relevant extract text MAY be appended to the
   Codex focus payload so Codex weighs it — but keep it to the 1-2 extracts that
   actually bear on the shipped diff (token economy; do not dump the corpus).
4. **Honesty rule (hard).** If a capital/statistical finding has NO covering
   extract, label it `UNSUPPORTED: no local literature extract` — do NOT
   manufacture a citation and do NOT cite from memory as if read. If only the
   curated extract was read (not the raw PDF), say "per curated extract", never
   "per <paper> (read in full)".

Non-capital, non-statistical findings (hooks, git, worktree, CLI) need no
literature grounding — skip it for those to save tokens.

## Execution mode

Decide foreground vs background by the SHIPPED-changeset size, not the working tree:

- If args include `--wait` → foreground, no question.
- If args include `--background` → background, no question.
- Otherwise size it: `git -C <worktree> diff --shortstat <base-ref>...HEAD`.
  - Tiny (≈1-2 files, no directory-sized change) → recommend Wait.
  - Anything larger or unclear → recommend Background (Codex's branch review of a
    real shipped range is rarely tiny).
  - Then call `AskUserQuestion` exactly once, recommended option first with
    `(Recommended)` suffix: `Wait for results` / `Run in background`.

## Foreground flow

Run from the review worktree directory:
```bash
node "${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs" adversarial-review --wait --scope branch --base <base-ref> "<anti-over-claim focus + user focus>"
```
Return the command stdout verbatim, exactly as-is. Do not paraphrase, summarize,
add commentary, or fix anything mentioned.

## Background flow

```typescript
Bash({
  command: `node "${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs" adversarial-review --background --scope branch --base <base-ref> "<anti-over-claim focus + user focus>"`,
  description: "Codex high-risk adversarial review",
  run_in_background: true
})
```
Do not call `BashOutput` or wait this turn. After launching, tell the user:
"High-risk Codex review started in the background. Check `/codex:status` for
progress." When it completes, return Codex's output verbatim.

## Notes

- The companion already implements `--scope branch` and `--base <ref>`
  (codex-companion.mjs; commands/adversarial-review.md). This command is a thin,
  durable preset — no script logic of its own.
- Argument handling: preserve user args; never weaken the adversarial framing;
  never strip `--wait`/`--background`.
