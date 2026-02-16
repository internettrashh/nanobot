---
name: memory
description: Two-layer memory system with automatic cloud recall when available.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/HISTORY.md` — Append-only event log.

## Recall

If cloud memory is active (you'll see a "Recalled Context" section in your context), recall is **fully automatic** — relevant memories are injected every turn. Do NOT manually read MEMORY.md or grep HISTORY.md; that data is already available to you.

If cloud memory is not active, search past events manually:
```bash
grep -i "keyword" memory/HISTORY.md
```

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically summarized and appended to HISTORY.md when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
