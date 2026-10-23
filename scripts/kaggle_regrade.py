"""Kaggle CPU kernel: regrade agent patches in a scorer-like sandbox.

Pushed by scripts/push_regrade.py, which fills in RUNS and TASK_IDS. Our eval kernels
grade with swegemma's notebook subprocess sandbox, which fails many correct patches:
the sandbox sees Kaggle's own packages, installs nothing from wheels/, and lacks
test-only packages. This kernel rebuilds the published harness environment (clean
venv, newest competition wheel of every package, pytest 8.4.2, the imp/telnetlib
shims) and regrades the gold patch plus every run's saved patch.

The sandbox setup and per-task calibration follow zbigniewkirchner's public notebook
"Gemma 4 Agent: do gold patches pass locally" (127/129 gold patches pass), and use
its datasets gemma-4-agent-local-eval-wheels and gemma-4-agent-local-eval-pythons.
Prints one line per grading: REGRADE <run> <instance_id> <resolved> <exit_code> <error>.
"""
import asyncio
import glob
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUNS: list[str] = []  # filled in by push_regrade.py; 'gold' grades the reference patches
TASK_IDS: list[str] = []  # filled in by push_regrade.py

COMP = Path(glob.glob('/kaggle/input/**/tasks.jsonl', recursive=True)[0]).parent
WHEELHOUSE = Path(glob.glob('/kaggle/input/**/adk_submission-*.whl', recursive=True)[0]).parent
print(COMP, WHEELHOUSE, sys.version, sep='\n', flush=True)


def fits(w, tag='cp313'):
    py, abi, plat = w.name[:-4].split('-')[-3:]
    if plat != 'any' and 'linux' not in plat:
        return False
    if abi == 'abi3':
        return int(py.split('.')[0][2:]) <= int(tag[2:])
    return any(t.startswith('py3') for t in py.split('.')) or tag in py.split('.')


tmp = Path('/tmp/wheelhouse')
tmp.mkdir(exist_ok=True)
for w in WHEELHOUSE.glob('*.whl'):  # Kaggle strips '+cu128' from wheel names
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    if not (tmp / name).exists():
        os.symlink(w, tmp / name)
small = [str(w) for w in tmp.glob('*.whl')
         if w.stat().st_size < 50 * 1024**2 and 'cutlass' not in w.name.lower() and fits(w)]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *small])
importlib.invalidate_caches()

from adk_eval_core.tracing import SessionTrace  # noqa: E402
from swegemma.config import EvalConfig  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
tasks = {t.instance_id: t for t in load_tasks(COMP / 'tasks.jsonl')}

# __SCORER_ENV__

# ---------------------------------------------------------------- regrade
PATCH = {'text': ''}


class PatchEvaluator(Evaluator):
    async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kwargs):
        return PATCH['text'], None, SessionTrace()


cfg = EvalConfig(tasks_path=COMP / 'tasks.jsonl', snapshots_dir=COMP / 'snapshots',
                 results_dir=Path('/kaggle/working/regrade'), submission_dir=Path('/kaggle/working'),
                 models=None, sandbox='subprocess', wheels_dir=COMP / 'wheels', graph_dir=str(COMP / 'graphs'),
                 embeddings_dir=str(COMP / 'embeddings'), timeout_seconds=300, verification_timeout_seconds=300,
                 display_mode='quiet')
ev = PatchEvaluator(cfg)
ev.sandbox = ev.docker = SubprocessManager(timeout_seconds=300, system_site_packages=False)
pool = ThreadPoolExecutor(max_workers=1)


def patches_for(run):
    if run == 'gold':
        return {tid: tasks[tid].patch for tid in TASK_IDS}
    hits = glob.glob(f'/kaggle/input/**/gemma-eval-{run}/results/patches', recursive=True)
    if not hits:
        print(f'NO OUTPUT for {run}', flush=True)
        return {}
    return {p.stem: p.read_text() for p in Path(hits[0]).glob('*.patch')}


t0 = time.time()
for run in RUNS:
    patches = patches_for(run)
    ids = [t for t in TASK_IDS if t in patches] if run == 'gold' else sorted(patches)
    n = 0
    for tid in ids:
        if not patches[tid].strip():
            print(f'REGRADE {run} {tid} False - empty', flush=True)
            continue
        CURRENT['id'], PATCH['text'] = tid, patches[tid]
        try:
            res = pool.submit(asyncio.run, ev.evaluate_task(task=tasks[tid])).result()
            ok, code, err = bool(res.resolved), res.test_exit_code, (res.error or '')[:120]
        except Exception as e:
            ok, code, err = False, -1, repr(e)[:120]
        n += ok
        print(f'REGRADE {run} {tid} {ok} {code} {json.dumps(err)}', flush=True)
    print(f'SUMMARY {run} resolved={n} graded={len(ids)} ({time.time() - t0:.0f}s)', flush=True)
