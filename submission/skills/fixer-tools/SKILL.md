---
name: fixer-tools
description: Edit helpers for the fixer. replace_lines.py edits by line numbers when edit_file fails; check_patch.py reviews the diff before submit_patch.
---

# fixer-tools

Run each script with `run_skill_script(skill_name="fixer-tools", file_path="scripts/<name>.py", args={...})`.

## scripts/replace_lines.py
Use when edit_file fails to match. Replaces lines start..end (numbers from the brief) with the new code,
written exactly as given including indentation.
- `args={"file": "rich/console.py", "start": 120, "end": 124, "text": "<new lines>"}`
Refuses changes that break Python syntax and prints the region before and after.

## scripts/check_patch.py
Run before submit_patch. Shows the diff and flags syntax errors, edits to test or config files,
stray untracked files, and an empty patch.
- `args={"clean": "true"}` also deletes scratch files left in the repo.
