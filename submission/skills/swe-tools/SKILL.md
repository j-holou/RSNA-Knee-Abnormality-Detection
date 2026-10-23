---
name: swe-tools
description: Fast helpers for fixing an issue in /workspace. locate.py ranks the source files to edit, repro.py runs and reruns a reproduction of the bug, lookup.py shows definitions and uses of several names in one call, replace_lines.py edits by line numbers when edit_file fails, find_tests.py finds existing tests, check_patch.py reviews the diff before submit_patch.
---

# swe-tools

Run each script with `run_skill_script(skill_name="swe-tools", file_path="scripts/<name>.py", args={...})`.
They read the repository in /workspace and print a short plain-text report.

## scripts/locate.py
Ranks the non-test source files most likely to need the fix.
- `args={"query": "<issue title + key names, error text, quoted strings>"}`
- optional `"top": 8`
Prints each file with its score and the matching `def`/`class` lines. Read the top one or two files next.

## scripts/repro.py
Saves a short reproduction script to /tmp and runs it from /workspace (60 s limit).
- `args={"code": "<python code that asserts the expected behavior, ending with print(\"REPRO OK\")>"}`
- `args={"rerun": "true"}` runs the saved script again after your edit.
Prints the output and `RESULT: PASS` or `RESULT: FAIL`. Before the fix it should fail; after the fix it should pass.

## scripts/lookup.py
One call instead of many greps. For each comma-separated name prints its definitions with
numbered code, and where it is used.
- `args={"names": "Console.print, soft_wrap, 'exact error text'"}`
- `Class.method` finds the method inside that class; a quoted entry is searched as literal text.
- optional `"file": "rich/console.py"` limits the search to one file or directory.

## scripts/replace_lines.py
Use when edit_file fails to match. Replaces lines start..end (numbers from lookup.py or
read_file) with the new code, written exactly as given including indentation.
- `args={"file": "rich/console.py", "start": 120, "end": 124, "text": "<new lines>"}`
Refuses changes that break Python syntax and prints the region before and after.

## scripts/find_tests.py
Finds existing tests that exercise a file or symbol and prints a ready pytest command.
- `args={"target": "rich/table.py"}` or `args={"target": "SymbolName"}`

## scripts/check_patch.py
Run before every submit_patch. Shows the diff and flags: syntax errors, edits to test or
config files (the grader discards them), stray untracked files, and an empty patch.
- `args={"clean": "true"}` also deletes scratch files (repro.py, debug_*.py, ...) left in the repo.
Prints `OK: ready for submit_patch` when nothing is wrong.
