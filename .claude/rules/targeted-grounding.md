# Targeted Grounding And Second-Pass Check

The user should not need to remember exact commands, skills, plugins, or review
rituals.

When the prompt asks to check, improve, implement, fix, build, review, plan,
audit, verify, or similar, Claude should automatically do a compact grounding
pass before acting.

`2P` means "second pass" and should trigger this same route. Semantic
equivalents should also trigger it, including second pass, double check, fresh
eyes, another pass, sanity check, red-team, critique, stress-test, take a look,
look over, is this good, what am I missing, poke holes, blind spots, hold up,
sense check, smell test, thoughts on, QA, will this work, spot flaws, gotchas,
and risks.

1. Identify the smallest relevant truth surfaces.
2. Read only the likely touched files/docs/runtime state.
3. Treat prior summaries and first-draft plans as claims.
4. Do a second-pass critique for gaps, silence, bias, error, and simpler
   improvements before presenting or editing.
5. Keep the cue low-token. Do not broaden into a full repo audit unless risk
   or ambiguity requires it.

This rule is prompted by `.claude/hooks/targeted-grounding-router.py`.

## Every Plan Is 2-Pass — MANDATORY

Any plan/design intent (`plan`, `design`, `brainstorm`, `approach`, `prereg`,
`hypothesis`, `spec`, `proposal`, `memo`, `architecture`, `how should/do we`,
`4t`) auto-fires the PLAN route — the operator must NOT have to ask for it. The
first draft is always wrong, so:

1. **Pass 1** — draft the plan.
2. **Pass 2** — self-critique for gaps, silences, bias, error, and a simpler
   path. Never present a single-pass plan.
3. **Fold in a rigor section** covering: no-bias/no-look-ahead, honesty
   (verified vs claimed), literature grounding, edge cases
   (NULL/empty/sparse/failure), future-proofing/hardening.

When the operator DOES say "improve/check/gaps/silences" on a plan, the same
route fires — do it properly, not a token gesture. The hook's `PLAN_INTENT`
regex is the source of truth for triggering; keep it in sync with this list.

### Enforcement (v2) — the cue is not advisory

The prompt cue above is only a reminder; these layers ENFORCE it so a thin plan
cannot reach the operator:

- **`.claude/hooks/_plan_rigor.py`** — single source of truth: the five
  `RIGOR_PILLARS`, the 2nd-pass marker, and the performative-honesty tripwire.
  The cue, the gate, and the backstop all import it; they cannot drift.
- **`.claude/hooks/plan-rigor-gate.py`** (`PreToolUse:ExitPlanMode`) — reads the
  plan text, audits it, and **soft-blocks (exit 2)** if a pillar is missing, the
  2nd pass is not shown, or rigor is claimed without evidence. Returns the plan to
  Claude to fix; never reaches the operator thin. Fail-open on any parse error.
  This is the strong layer — it audits structured plan text, never guesses.
- **`.claude/hooks/plan-stop-backstop.py`** (`Stop`) — covers plans the operator
  explicitly asked for but that were written in chat (bypassing ExitPlanMode). It
  keys on KNOWN intent: the router drops a per-turn breadcrumb
  (`state/plan-intent.json`, session-keyed) when the PLAN route fires; the
  backstop fires the advisory only if that breadcrumb is pending AND the reply
  lacks rigor, then clears it. It does NOT classify prose as plan-vs-report
  (that was tried, false-fired on a completion report, and was removed —
  2026-05-31). Advisory injection only.

Parity between the pillar list here and `_plan_rigor.RIGOR_PILLARS` is enforced
by `check_plan_rigor_pillar_parity` in `pipeline/check_drift.py`.

## Resource And Fetch Triggers

`/resource` and `/lit` mean: read the local grounding truth before planning or
acting.

Important: the full `resources/` PDF corpus is a local-PC asset, not guaranteed
remote/CI state. In worktrees or remote agents, the tracked `resources/INDEX.md`
and `docs/institutional/literature/` extracts may exist while the raw PDFs do
not. If the raw resource file is missing, say that directly and do not imply
raw-PDF verification. Use tracked curated extracts only, or rerun on the PC
that has `C:\Users\joshd\canompx3\resources`.

Required route:

1. Run `python scripts/tools/check_pdf_tooling.py` before making PDF-backed
   claims. If the task needs OCR or the text yield is low, run with
   `--require-ocr` or explicitly verify `ocrmypdf --version`.
2. Run `python scripts/tools/check_literature_coverage.py` or inspect
   `resources/INDEX.md` to determine whether the relevant resource has a
   curated extract and whether the raw resource file is present locally.
3. Open `resources/INDEX.md`.
4. Prefer the mapped curated extract in `docs/institutional/literature/` when
   one exists and the extract covers the claim.
5. Use the raw `resources/` PDFs only when the curated extract is missing or the
   prompt specifically needs raw-page verification, and only when the raw file
   is present on this PC.
6. Never answer from memory or feel when the user invoked `/resource`, `/lit`,
   "canonical truth", "trading bible", or "local literature".
7. No cheating or skim-claims: report what was actually opened/extracted. Do
   not imply full-PDF reading unless the full PDF was extracted or the curated
   extract says it covers the relevant scope. Before dismissing a PDF, extract
   the TOC plus at least three relevant/mid-document pages.

For `research`, `fetch`, `look up`, `search`, `docs`, `changelog`, `release
notes`, `upgrade`, `fixes`, `user comments`, `GitHub issues`, forums, Reddit,
or StackOverflow:

- Separate official/primary sources from unofficial user reports.
- Use official docs, release notes, changelogs, source code, specs, and vendor
  pages as the primary layer.
- Use user comments/issues/forums as useful signals only when labeled
  unofficial and treated with caution.
- It is acceptable to act on unofficial information when it is the best live
  signal, but say that explicitly and prefer corroboration.
