---
name: gsd
description: Get Shit Done (GSD) spec-driven development framework for Antigravity. Deconstructs complex features into phased specs, creates atomic task checklists, prevents context rot, enforces verification gates, and produces clean atomic commits.
---

# Get Shit Done (GSD) Protocol

The GSD methodology structures development to prevent **context rot**, eliminate silent hallucinations, and deliver verifiable, production-grade features.

## Core Phases

### Phase 1: Requirements & Anti-Hallucination Q&A
Before writing any code:
1. Analyze existing codebase patterns, interfaces, and dependencies.
2. Identify any ambiguities, hidden assumptions, or architectural trade-offs.
3. Clarify gray areas upfront with the user rather than guessing product decisions.

### Phase 2: Spec & Task Roadmap (`tasks.md`)
Create or update `tasks.md` in the workspace root with a clear checklist:
```markdown
# Feature: <Feature Name>

## Architecture & Interfaces
- [ ] Interface specification
- [ ] Data models and state schemas

## Implementation Steps
- [ ] Task 1: Component / Module setup
- [ ] Task 2: Core logic & data flow
- [ ] Task 3: Edge cases & error handling

## Verification Gate
- [ ] Unit tests passing
- [ ] End-to-end integration test
- [ ] Performance & regression check
```

### Phase 3: Atomic Execution
- Execute **exactly one task** at a time.
- Keep file edits contiguous and minimal.
- Do not introduce unrequested abstractions or premature refactors.

### Phase 4: Verification Gate
- Run tests (`pytest`, `npm test`, or custom test harnesses).
- Verify zero syntax errors, type errors, or broken imports.
- Check edge cases (empty inputs, network timeouts, invalid arguments).

### Phase 5: Atomic Git Commit & Memory Persistence
- Commit completed tasks with clean, semantic commit messages:
  `git commit -m "feat(<scope>): implement <task description>"`
- Distill key architectural decisions into `progress.txt` or `.gemini/` rules so context persists across sessions.
