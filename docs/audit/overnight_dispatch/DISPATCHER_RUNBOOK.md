# Dispatcher Runbook (read this every fire)

You are the lean dispatcher for an overnight capital code-review loop. Your job
each 15m fire: dispatch ONE review, persist to disk, stay lean. Do NOT review code
yourself — that's the subagent's job, in ITS context, not yours.

## Each fire, do exactly this:

1. Read `WORKLIST.md`. Find the first row with status `PENDING` (lowest id).
   - If none PENDING and none INPROGRESS → all done. Write `SUMMARY.md`
     (aggregate the `results/*.md` verdicts, list any CONFIRMED capital bugs),
     send a PushNotification one-liner, and STOP the loop (omit ScheduleWakeup).
   - If a row is INPROGRESS with no matching `results/<id>.md`, a prior subagent
     died — reset it to PENDING and proceed.

2. Flip that row to `INPROGRESS` in WORKLIST.md (one Edit).

3. Spawn ONE subagent with a NARROW prompt. Agent type by target:
   - capital/live/broker/sizing/gate paths → `live-risk-auditor` (read-only).
   - generic correctness/dead-code → `evidence-auditor` or `general-purpose`.
   Pick `live-risk-auditor` for every current worklist row (all capital paths).
   The prompt must contain:
   - the exact target path(s) + focus from the row
   - "Adversarially review for capital-wreck bugs ONLY. Confirm/refute the named
     finding. Ground in actual code — quote line numbers. If you propose a fix,
     describe it; do NOT edit files."
   - "Return ≤300 words: VERDICT (CONFIRMED/REFUTED/PARTIAL), evidence (quoted
     lines), severity, proposed fix sketch."
   - "repo_root = C:/Users/joshd/canompx3"

4. Write the subagent's full return verbatim to `results/<id>.md` (with a header:
   id, target, verdict, date). Flip the WORKLIST row to `DONE`.

5. Report ≤2 lines to the user: which id ran, its verdict. Then ScheduleWakeup
   (15m cadence is the cron — for dynamic continuation use ~900s fallback).

## Hard rules
- ONE subagent per fire. Never two. (subagent-budget.md)
- You do NOT edit production code. This loop AUDITS; fixes are a separate
  operator-gated session (capital = Tier B).
- All state on disk. Your own context must not accumulate review bodies — write
  them to results/ and only keep the 1-line verdict in your reply.
- If past ~80K tokens in THIS session, note it and rely on disk state; a fresh
  fire after summarization re-reads this runbook + WORKLIST and continues cleanly.
