# Codex Memory Policy

Codex uses the shared workspace memory files.

## Write Targets

- Daily log: `memory/YYYY-MM-DD.md`
- Long-term memory: `MEMORY.md` in the main session

## What To Capture

- Durable setup decisions
- Project-specific workflow lessons
- User preferences that materially affect execution
- Important environment facts that are likely to matter again

## What Not To Dump

- Large command transcripts
- Temporary exploration notes with no future value
- Secrets unless explicitly asked to retain them

## This Repo

When Codex changes its own adapter layer, record:

- what changed
- why it changed
- anything the next session should trust about the setup

Also record user-level Codex integration changes that materially affect this repo, such as MCP registration.
