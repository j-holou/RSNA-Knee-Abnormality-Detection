"""When did the agent first see the code the gold patch changes, and when did it first edit?

    python scripts/seen_vs_edit.py v6s v11s v11 v11b v12b   # pushes janholoubek/gemma-seen-vs-edit
    kaggle kernels logs janholoubek/gemma-seen-vs-edit       # SEEN lines, one per task, and SUMMARY lines

For every task in each eval run, the CPU kernel walks the trace call by call. A gold hunk
counts as seen by a call when that call's output contains enough of the hunk's original
lines (context and removed lines, stripped, at least 12 characters, not repeated in the
hunk). It prints the call that first showed any gold hunk, the call by which every gold
hunk had been shown, and the first call that changed a file. CPU only; no GPU quota.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = r'''
import glob, json, re, statistics
from pathlib import Path
RUNS = __RUNS__
tasks = {}
for line in open(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)[0]):
    t = json.loads(line)
    tasks[t["instance_id"]] = t


def gold_hunks(patch):
    """[(file, [original lines worth matching])] per hunk, skipping test files."""
    out, cur, f = [], None, None
    for l in patch.splitlines():
        if l.startswith("+++ "):
            f = l[4:].removeprefix("b/")
        elif l.startswith("@@"):
            cur = (f, [])
            out.append(cur)
        elif cur is not None and l[:1] in (" ", "-") and not l.startswith("---"):
            s = l[1:].strip()
            if len(s) >= 12:
                cur[1].append(s)
    keep = []
    for f, ls in out:
        if not f or "test" in f.split("/")[0] or "/tests/" in f or f.startswith("tests"):
            continue
        uniq = [s for s in dict.fromkeys(ls)]
        if uniq:
            keep.append((f, uniq))
    return keep


def text_of(obs):
    """Tool outputs are JSON strings; join their string values so code lines match unescaped."""
    try:
        j = json.loads(obs)
    except Exception:
        return obs
    out = []
    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(j)
    return "\n".join(out)


def seen(hunk_lines, text):
    need = min(3, len(hunk_lines))
    return sum(1 for s in hunk_lines if s in text) >= need


def is_edit(name, args, obs, raw):
    if name in ("edit_file", "write_file"):
        return '"status": "ok"' in raw
    if name == "run_command" and ".git/sr.py" in json.dumps(args):
        return "Applied to" in obs
    if name == "run_skill_script" and "replace_lines" in json.dumps(args):
        return "AFTER" in obs and "usage:" not in obs
    return False


def first_hit(calls, hunks):
    first, by_all, got = None, None, set()
    for k, (name, args, obs, raw) in enumerate(calls, 1):
        for i, (f, ls) in enumerate(hunks):
            if i not in got and seen(ls, obs):
                got.add(i)
                first = first or k
        if by_all is None and got and len(got) == len(hunks):
            by_all = k
    return first, by_all, len(got)


for run in RUNS:
    hits = glob.glob(f"/kaggle/input/**/gemma-eval-{run}/results", recursive=True)
    if not hits:
        print("NO OUTPUT", run); continue
    res = Path(hits[0])
    resolved = {}
    for line in (res / "task_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        resolved[r["instance_id"]] = bool(r.get("resolved"))
    rows = []
    for tf in sorted((res / "traces").glob("*.json")):
        tid = tf.stem.replace("trace_", "")
        if tid not in tasks:
            continue
        d = json.load(open(tf))
        calls = []
        for s in d.get("steps", []):
            raw = str((s.get("observation") or {}).get("content", "") or "")
            obs = text_of(raw)
            for tc in s.get("tool_calls") or []:
                calls.append((tc.get("function_name"), tc.get("arguments", {}) or {}, obs, raw))
        hunks = gold_hunks(tasks[tid].get("patch", ""))
        first, by_all, n_seen = first_hit(calls, hunks)
        edit = next((k for k, c in enumerate(calls, 1) if is_edit(*c)), None)
        seen_before_edit = None
        if edit:
            _, _, seen_before_edit = first_hit(calls[:edit - 1], hunks)
        row = dict(run=run, task=tid, resolved=resolved.get(tid, False), calls=len(calls), hunks=len(hunks),
                   first_seen=first, all_seen=by_all, hunks_seen=n_seen, first_edit=edit,
                   hunks_seen_before_edit=seen_before_edit)
        rows.append(row)
        print("SEEN", json.dumps(row))
    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None
    n = len(rows)
    edited = [r for r in rows if r["first_edit"]]
    print("SUMMARY", json.dumps(dict(
        run=run, tasks=n,
        never_saw_any=sum(r["first_seen"] is None for r in rows),
        median_first_seen=med(r["first_seen"] for r in rows),
        median_all_seen=med(r["all_seen"] for r in rows),
        median_first_edit=med(r["first_edit"] for r in rows),
        no_edit=n - len(edited),
        all_seen_before_edit=sum(1 for r in edited if r["hunks_seen_before_edit"] == r["hunks"]),
        median_gap_all_seen_to_edit=med((r["first_edit"] - r["all_seen"]) for r in edited if r["all_seen"] and r["all_seen"] < r["first_edit"]),
    )))
'''


def main() -> None:
    runs = sys.argv[1:] or ['v6s', 'v11s']
    user = os.environ.get('KAGGLE_USERNAME', 'janholoubek')
    meta = {
        'id': f'{user}/gemma-seen-vs-edit', 'title': 'gemma-seen-vs-edit', 'code_file': 'seen.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'kernel_sources': [f'{user}/gemma-eval-{r}' for r in runs],
        'dataset_sources': [], 'competition_sources': ['gemma-4-developer-agent'],
    }
    with tempfile.TemporaryDirectory() as tmp:
        src = KERNEL.replace('__RUNS__', repr(runs))
        compile(src, 'seen.py', 'exec')
        (Path(tmp) / 'seen.py').write_text(src)
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
