# Targeted Grounding

The user has explicitly said they will not reliably remember to ask for
plugins, tools, second-pass checks, or grounding rituals. Codex should infer
those from intent.

For prompts such as check, improve, implement, fix, build, review, audit,
verify, plan, design, investigate, diagnose, or "make it better":

`2P` means "second pass" and should trigger this route. Semantic equivalents
also count: second pass, double check, fresh eyes, another pass, sanity check,
red-team, critique, stress-test, take a look, look over, is this good, what am
I missing, poke holes, blind spots, hold up, sense check, smell test, thoughts
on, QA, will this work, spot flaws, gotchas, and risks.

1. Run a compact truth check before acting.
2. Read the smallest relevant files/docs/runtime state, not the whole repo.
3. Treat prior summaries and first-draft plans as claims.
4. Second-pass the plan or work for gaps, silence, bias, errors, and simpler
   improvements.
5. Keep the automatic cue low-token; widen only when risk or ambiguity demands
   it.

Implemented in:

- Codex: `.codex/hooks/user_prompt_submit_grounding.py`
- Claude: `.claude/hooks/targeted-grounding-router.py`

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
5. Use raw `resources/` PDFs only when the curated extract is missing or the
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
