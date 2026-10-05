"""Measure locate.py's hit rate against reference patches.

    python scripts/bench_locate.py --data data [--top 5]

For every task whose snapshot is present, extracts it, runs the skill's
locate.py on the problem statement, and checks whether a file edited by the
reference patch (tests excluded) is among the top results.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCATE = ROOT / 'submission/skills/swe-tools/scripts/locate.py'


def patch_files(patch: str) -> list[str]:
    files = re.findall(r'^\+\+\+ b/(\S+)', patch, re.M)
    return [f for f in files if not re.search(r'(^|/)(tests?|testing)/|(^|/)test_', f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data')
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in (a.data / 'tasks.jsonl').open()]
    rows = []
    for t in tasks:
        snap = a.data / 'snapshots' / f'{t["instance_id"]}.tgz'
        if not snap.exists():
            continue
        gold = patch_files(t['patch'])
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(snap) as tf:
                tf.extractall(td, filter='data')
            ws = next((p.parent for p in Path(td).rglob('.git') if p.is_dir()), Path(td))
            out = subprocess.run([sys.executable, str(LOCATE), '--query', t['problem_statement'],
                                  '--top', str(a.top)], cwd=ws, env={**os.environ, 'PWD': str(ws)},
                                 capture_output=True, text=True, timeout=120).stdout
        ranked = [l.split('  (score')[0] for l in out.splitlines() if '  (score' in l]
        rank = next((i + 1 for i, f in enumerate(ranked) if f in gold), None)
        rows.append((t['instance_id'], rank, gold[:3]))
        print(f'{t["instance_id"]:16} rank={rank}  gold={gold[:3]}', flush=True)
    hit = sum(1 for _, r, _ in rows if r)
    top1 = sum(1 for _, r, _ in rows if r == 1)
    print(f'\n{len(rows)} tasks: top-1 {top1}, top-{a.top} {hit}')


if __name__ == '__main__':
    main()
