# Stage-Gate Protocol

Before any non-trivial work, check `docs/runtime/STAGE_STATE.md`:
- Exists with active stage? → verify mode matches intent, continue or reclassify
- Missing? → run /stage-gate to classify first
- Stale? → check drift first (git log on scope files), age second (>4h fallback)

Trivial work (≤2 non-core files, mechanical, obvious acceptance) → /stage-gate writes TRIVIAL state.
Core files (pipeline logic, config, schema, validation, session, DB-write) can NEVER be TRIVIAL.

Do NOT edit production code (pipeline/, trading_app/, protected scripts/) without an active STAGE_STATE.
Do NOT commit STAGE_STATE.md alone — bundle with stage code changes at checkpoints only.
