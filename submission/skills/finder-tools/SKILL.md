---
name: finder-tools
description: The finder's investigation tool. investigate.py locates the files to change, looks up definitions and callers of names, or reads a line range. At most 4 calls per task.
---

# finder-tools

`run_skill_script(skill_name="finder-tools", file_path="scripts/investigate.py", args={...})`, at most 4 calls in total:
- `args={"mode": "locate", "query": "<issue title, symbol names, error text>"}` ranks the source files to change.
- `args={"mode": "lookup", "names": "Class.method, function, 'exact text'"}` prints definitions with line numbers and their callers; optional `"file"` limits the search.
- `args={"mode": "read", "file": "pkg/module.py", "start": 120, "end": 200}` prints at most 80 numbered lines.
