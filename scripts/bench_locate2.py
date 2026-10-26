"""Benchmark a locate.py variant: does it rank the gold file, and list the gold function?

    python scripts/bench_locate2.py --data data [--locate path/to/locate.py] [--top 8] [--tasks eval/dev48.txt]

For every task with a snapshot, extracts the repository without .git (cached), runs locate.py
on the problem statement, and reports per task:
  file_rank  rank of the first gold source file in locate's list (None = not listed)
  defs       gold hunks whose enclosing def/class line locate printed under that file
Gold hunks and their enclosing definitions come from the reference patch and the snapshot.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF_RE = re.compile(r'^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)')


def is_test(f):
    return bool(re.search(r'(^|/)(tests?|testing)/|(^|/)test_|conftest', f))


def gold_hunks(patch):
    """[(file, old_start_line)] for non-test files."""
    out, f = [], None
    for l in patch.splitlines():
        if l.startswith('+++ '):
            f = l[4:].removeprefix('b/')
        elif l.startswith('@@') and f and not is_test(f):
            m = re.match(r'@@ -(\d+)', l)
            out.append((f, int(m.group(1))))
    return out


def enclosing_def(lines, start, size=7):
    """Line number (1-based) of the def/class enclosing the hunk's changed region."""
    probe = min(len(lines), start + 3)  # hunks start with ~3 context lines
    for i in range(probe - 1, -1, -1):
        if DEF_RE.match(lines[i]):
            return i + 1
    return None


def extract(snap, dest):
    if (dest / '.done').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snap) as tf:
        members = [m for m in tf if not m.name.lstrip('./').startswith('.git') and m.isfile()
                   and m.size < 2_000_000]
        tf.extractall(dest, members=members, filter='data')
    (dest / '.git').mkdir(exist_ok=True)
    (dest / '.done').write_text('')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data')
    ap.add_argument('--locate', type=Path, default=ROOT / 'submission/skills/swe-tools/scripts/locate.py')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--tasks', type=Path)
    ap.add_argument('--cache', type=Path, default=Path('/tmp/claude-0/srccache'))
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    a.locate = a.locate.resolve()
    keep = set(a.tasks.read_text().split()) if a.tasks else None
    tasks = [json.loads(l) for l in (a.data / 'tasks.jsonl').open()]
    rows = []
    for t in tasks:
        tid = t['instance_id']
        snap = a.data / 'snapshots' / f'{tid}.tgz'
        if not snap.exists() or (keep and tid not in keep):
            continue
        ws = a.cache / tid
        extract(snap, ws)
        hunks = gold_hunks(t['patch'])
        gold_files = list(dict.fromkeys(f for f, _ in hunks))
        want = set()
        for f, start in hunks:
            p = ws / f
            if p.exists():
                d = enclosing_def(p.read_text(errors='replace').splitlines(), start)
                if d:
                    want.add((f, d))
        for f in ('reads', 'read_cap'):  # the skill's reading budget must not interfere
            (ws / '.git' / f).unlink(missing_ok=True)
        env = {k: v for k, v in os.environ.items() if k != 'PWD'}
        env['PWD'] = str(ws)
        out = subprocess.run([sys.executable, '-I', str(a.locate), '--query', t['problem_statement'],
                              '--top', str(a.top)], cwd=ws, env=env, capture_output=True, text=True,
                             timeout=300).stdout
        ranked, shown, cur = [], set(), None
        for l in out.splitlines():
            m = re.match(r'(\S+)  \(score', l)
            if m:
                cur = m.group(1)
                ranked.append(cur)
                continue
            m = re.match(r'\s+(\d+): ', l)
            if m and cur:
                shown.add((cur, int(m.group(1))))
        rank = next((i + 1 for i, f in enumerate(ranked) if f in gold_files), None)
        row = dict(task=tid, file_rank=rank, defs_hit=len(want & shown), defs=len(want), gold_files=gold_files)
        rows.append(row)
        if not a.json:
            print(f"{tid:16} file_rank={rank}  defs {len(want & shown)}/{len(want)}  gold={gold_files[:2]}", flush=True)
    n = len(rows)
    top1 = sum(r['file_rank'] == 1 for r in rows)
    top3 = sum(bool(r['file_rank']) and r['file_rank'] <= 3 for r in rows)
    anyr = sum(bool(r['file_rank']) for r in rows)
    dh = sum(r['defs_hit'] for r in rows)
    dn = sum(r['defs'] for r in rows)
    summary = dict(tasks=n, file_top1=top1, file_top3=top3, file_listed=anyr, defs_hit=dh, defs=dn)
    print('SUMMARY', json.dumps(summary))
    if a.json:
        print(json.dumps(rows))


if __name__ == '__main__':
    main()
