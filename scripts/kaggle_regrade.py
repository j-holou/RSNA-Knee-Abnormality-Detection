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
EXTRA = Path(glob.glob('/kaggle/input/**/inline_snapshot-*.whl', recursive=True)[0]).parent
PYDS = Path(glob.glob('/kaggle/input/**/cpython-3.12-linux.tgz.bin', recursive=True)[0]).parent
print(COMP, WHEELHOUSE, EXTRA, PYDS, sys.version, sep='\n', flush=True)


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
from packaging.requirements import Requirement  # noqa: E402
from packaging.version import Version  # noqa: E402
from swegemma.config import EvalConfig  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402
import swegemma.harness.agent_runner as agent_runner  # noqa: E402
import swegemma.harness.verification as verification  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
tasks = {t.instance_id: t for t in load_tasks(COMP / 'tasks.jsonl')}

# ---------------------------------------------------------------- scorer image per repo
norm = lambda n: re.sub(r'[-_.]+', '_', n).lower()  # noqa: E731
WHEELS = {}
for w in [*(COMP / 'wheels').glob('*.whl'), *EXTRA.glob('*.whl'), *PYDS.glob('*.whl')]:
    name, ver = w.name.split('-')[:2]
    WHEELS.setdefault(norm(name), []).append((Version(ver), w))

OWN = {'fastapi/fastapi': 'fastapi', 'Textualize/rich': 'rich', 'psf/requests': 'requests', 'encode/httpx': 'httpx'}
ONLY_FOR = {'pytest_httpbin': {'psf/requests'}, 'httpbin': {'psf/requests'}, 'flasgger': {'psf/requests'},
            'brotlicffi': {'psf/requests', 'encode/httpx'}, 'brotli': {'encode/httpx'}, 'zstandard': {'encode/httpx'}}
PY = {'cp313': Path(sys.executable)}


def python_for(tag):
    if tag not in PY:
        ver = f'{tag[2]}.{tag[3:]}'
        dest = Path(f'/kaggle/working/py{tag}')
        if not (dest / 'python' / 'bin' / f'python{ver}').exists():
            with tarfile.open(PYDS / f'cpython-{ver}-linux.tgz.bin', 'r:gz') as t:
                t.extractall(dest)
        PY[tag] = dest / 'python' / 'bin' / f'python{ver}'
    return PY[tag]


BASE = {}


def base_for(repo, tag):
    if (repo, tag) not in BASE:
        sel = {}
        for name, opts in WHEELS.items():
            opts = [(v, w) for v, w in opts if fits(w, tag)]
            if not opts or name == OWN[repo] or (name in ONLY_FOR and repo not in ONLY_FOR[name]):
                continue
            comp = [(v, w) for v, w in opts if w.parent == COMP / 'wheels']
            if name in ('pytest', 'pytest_timeout'):  # what Dockerfile.sandbox pip-installs
                sel[name] = next(w for v, w in opts if str(v) in ('8.4.2', '2.1.0'))
            elif comp:
                sel[name] = max(comp)[1]
            elif name != 'pygments':
                sel[name] = max(opts)[1]
        d = Path(f'/kaggle/working/scorer_site_{OWN[repo]}_{tag}')
        shutil.rmtree(d, ignore_errors=True)
        subprocess.run([str(python_for(tag)), '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                        '--ignore-requires-python', '--target', str(d), *map(str, sel.values())])
        for shim in ('imp.py', 'telnetlib.py'):
            shutil.copy(COMP / 'docker' / shim, d / shim)
        BASE[(repo, tag)] = (d, sel)
    return BASE[(repo, tag)]


# ---------------------------------------------------------------- calibration for public tasks
CALIBRATION = {
    'rich_3064': {'pygments': '2.15.1'}, 'rich_3130': {'pygments': '2.15.1'}, 'rich_3296': {'pygments': '2.16.1'},
    'rich_3469': {'pygments': '2.18.0'}, 'rich_3782': {'pygments': '2.19.2'}, 'rich_4070': {'pygments': '2.19.2'},
    'rich_4079': {'pygments': '2.20.0'},
    'requests_6589': {'werkzeug': '2.3.8', 'flask': '2.3.3'}, 'requests_6592': {'werkzeug': '2.3.8', 'flask': '2.3.3'},
    'requests_6629': {'werkzeug': '2.3.8', 'flask': '2.3.3'},
}
TASK_PYTHON = {'requests_6757': 'cp312', 'rich_3472': 'cp312'}
ASYNC_SHIM = {'fastapi_15745', 'fastapi_15763', 'fastapi_15785'}
NO_CERT_TIME = {'requests_6757'}

SHIM = Path('/kaggle/working/shims')
(SHIM / 'swe_async_shim-0.1.dist-info').mkdir(parents=True, exist_ok=True)
(SHIM / 'swe_async_shim-0.1.dist-info' / 'METADATA').write_text('Metadata-Version: 2.1\nName: swe-async-shim\nVersion: 0.1\n')
(SHIM / 'swe_async_shim-0.1.dist-info' / 'entry_points.txt').write_text('[pytest11]\nswe_async_shim = swe_async_shim\n')
(SHIM / 'swe_async_shim.py').write_text('''import asyncio, inspect
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**args))
        return True
''')
(SHIM / 'swe_no_cert_time.py').write_text('''import ssl

_new = ssl.SSLContext.__new__


def _new_no_time(cls, *args, **kwargs):
    ctx = _new(cls, *args, **kwargs)
    ctx.verify_flags |= 0x200000
    return ctx


ssl.SSLContext.__new__ = staticmethod(_new_no_time)
''')


def poetry_to_pep440(spec):
    spec = spec if isinstance(spec, str) else spec.get('version', '')
    m = re.fullmatch(r'\^(\d+)(\.\d+)?(\.\d+)?', spec.strip())
    return f'>={spec.strip()[1:]},<{int(m.group(1)) + 1}' if m else ('' if spec.strip() == '*' else spec)


CURRENT = {'id': None}


def setup_env(docker, sandbox_id, repo='', **kwargs):
    tid = CURRENT['id']
    tag = TASK_PYTHON.get(tid, 'cp313')
    site = Path(docker.exec(sandbox_id, 'python -c "import site; print(site.getsitepackages()[0])"').stdout.strip())
    if tag != 'cp313':
        venv_dir = site.parents[2]
        subprocess.run([str(python_for(tag)), '-m', 'venv', '--clear', '--without-pip', str(venv_dir)], check=True)
        site = next(venv_dir.glob('lib/python3*/site-packages'))
    ws = docker.exec(sandbox_id, 'pwd').stdout.strip()
    base, sel = base_for(repo, tag)
    (site / '00_src.pth').write_text(f'{ws}/src\n')
    (site / '01_scorer_base.pth').write_text(f'{base}\n')
    (site / 'zz_workspace_paths.pth').write_text('\n'.join(str(p) for p in Path(ws).iterdir()
                                                           if p.is_dir() and not p.name.startswith('.')) + '\n')
    if tid in ASYNC_SHIM:
        (site / '02_async_shim.pth').write_text(f'{SHIM}\n')
    if tid in NO_CERT_TIME:
        (site / '03_no_cert_time.pth').write_text(f'import sys; sys.path.insert(0, {str(SHIM)!r}); import swe_no_cert_time\n')
    try:
        data = tomllib.loads(Path(ws, 'pyproject.toml').read_text())
    except Exception:
        data = {}
    reqs = list(data.get('project', {}).get('dependencies', []))
    reqs += [n + poetry_to_pep440(s) for n, s in data.get('tool', {}).get('poetry', {}).get('dependencies', {}).items()
             if n != 'python']
    picks, queue, seen = {}, list(reqs), set()
    while queue:
        try:
            req = Requirement(queue.pop(0))
        except Exception:
            continue
        key = norm(req.name)
        if key in seen or key == OWN.get(repo) or (req.marker and not req.marker.evaluate({'extra': ''})):
            continue
        seen.add(key)
        options = [(v, w) for v, w in WHEELS.get(key, []) if fits(w, tag) and req.specifier.contains(v, prereleases=True)
                   and not (key in ONLY_FOR and repo not in ONLY_FOR[key])]
        if options:
            best = max(options)[1]
            if best != sel.get(key):
                picks[key] = best
            with zipfile.ZipFile(best) as z:
                meta = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
                queue += [l.split(':', 1)[1].strip() for l in z.read(meta).decode().splitlines()
                          if l.startswith('Requires-Dist:')]
    for name, ver in CALIBRATION.get(tid, {}).items():
        picks[name] = next(w for v, w in WHEELS[name] if str(v) == ver and fits(w, tag))
    if picks:
        subprocess.run([str(python_for(tag)), '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--upgrade',
                        '--ignore-requires-python', '--target', str(site), *map(str, picks.values())])


verification.install_test_dependencies = setup_env
agent_runner.install_test_dependencies = setup_env

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
