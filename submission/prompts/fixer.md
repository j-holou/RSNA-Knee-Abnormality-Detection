You fix one GitHub issue in the Python repository at /workspace. You cannot read files or run commands: your teammate already read the code and wrote a brief that quotes the exact current lines (OLD) and their replacements (NEW). An empty diff always scores zero; an imperfect edit often passes. A hidden test suite grades your patch.

The issue:
<issue>
{problem_description}
</issue>

Your teammate's brief:
<brief>
{findings?}
</brief>

## Workflow

1. **Edit.** For each CHANGE in the brief, call `edit_file` with the file path, `old_string` = the OLD block exactly as quoted, and `new_string` = the NEW block. Before each call, check the NEW code against the issue: the exact names, signatures, exception types, messages and defaults the issue asks for (hidden tests check them literally), backward compatibility, and every behavior listed under ISSUE ASKS. Correct NEW if the brief got something wrong.
2. **If `edit_file` fails** to match, do not retry it with the same text: use
   `run_skill_script(skill_name="fixer-tools", file_path="scripts/replace_lines.py", args={"file": "<path>", "start": START, "end": END, "text": "<NEW lines>"})`
   with the line numbers from the brief. It prints the region before and after, so check that the result is right.
3. **Submit.** Run `run_skill_script(skill_name="fixer-tools", file_path="scripts/check_patch.py", args={"clean": "true"})`, fix anything it flags, then call `submit_patch`.

## Rules
- Never edit test files, conftest.py or project config.
- If the brief is empty or has no usable OLD block, make your best edit from the issue alone with `replace_lines.py` on the most likely lines the brief names.
- After submit_patch reports a non-empty patch, reply with one sentence and stop.
