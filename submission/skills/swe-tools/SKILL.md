---
name: swe-tools
description: Fast helpers for fixing an issue in /workspace. locate.py ranks the source files to edit, find_tests.py finds existing tests for a file or symbol, check_patch.py reviews the diff before submit_patch.
---

# swe-tools

Run each script with `run_skill_script(skill_name="swe-tools", file_path="scripts/<name>.py", args={...})`.
They read the repository in /workspace and print a short plain-text report.

## scripts/locate.py
Ranks the non-test source files most likely to need the fix.
- `args={"query": "<issue title + key names, error text, quoted strings>"}`
- optional `"top": 8`
Prints each file with its score and the matching `def`/`class` lines. Read the top one or two files next.

## scripts/find_tests.py
Finds existing tests that exercise a file or symbol and prints a ready pytest command.
- `args={"target": "rich/table.py"}` or `args={"target": "SymbolName"}`

## scripts/check_patch.py
Run before every submit_patch. Shows the diff and flags: syntax errors, edits to test or
config files (the grader discards them), stray untracked files, and an empty patch.
- `args={"clean": "true"}` also deletes scratch files (repro.py, debug_*.py, ...) left in the repo.
Prints `OK: ready for submit_patch` when nothing is wrong.
