---
name: ralph-loop
description: Ralph Loop autonomous development engine for Antigravity. Executes long-running tasks across clean iterations using externalized memory (PRD.md + progress.txt) until all goals are achieved, recovering from interruptions or context limits.
---

# Ralph Loop Autonomous Execution Engine

The Ralph Loop is designed to run complex, long-running agentic coding tasks without suffering from context window degradation or getting stuck in loops.

## How the Loop Works

```
                     ┌──────────────────────────────┐
                     ▼                              │
┌──────────────┐   ┌──────────────┐   ┌───────────┐ │
│ Read PRD.md  │──>│ Execute Next │──>│ Verify &  │─┴── [All Tasks Done?] ──> 🎉 Done!
│ & progress   │   │ Single Task  │   │ Commit    │
└──────────────┘   └──────────────┘   └───────────┘
```

### Core Components

1. **`PRD.md` (Product Requirements Document)**:
   - Read-only task master plan created at the beginning.
   - Defines the target outcome, architectural constraints, and testable acceptance criteria.

2. **`progress.txt` (Externalized State Tracker)**:
   - Appended after each iteration.
   - Records:
     - What was just implemented.
     - Files modified or created.
     - Test results.
     - Next immediate step.

3. **Fresh Iteration Execution**:
   - Instead of trying to hold 20 steps of history in memory, the agent reads `PRD.md` and `progress.txt` to instantly orient itself.
   - Implements the single next uncompleted task.
   - Runs verification tests.
   - Makes a git commit.
   - Updates `progress.txt` and continues until all tasks in `PRD.md` are marked complete.

## Triggering the Ralph Loop
- Use the **`/goal`** slash command in Antigravity for autonomous overnight or long-running execution.
- Pair with `Always Proceed` execution policies for non-blocking terminal runs.
