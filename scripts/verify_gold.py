"""Run Phase 2 verification with each task's reference patch.

Confirms the local harness setup can build, patch and test a task without a
model. A task that fails here cannot be solved by any agent locally.

    python scripts/verify_gold.py --data data --task-ids httpx_3672
"""
import argparse
import asyncio
import json
import os
import shlex
from pathlib import Path

from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.harness import container_setup
from swegemma.harness import verification
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from adk_submission import ModelRegistry
from swegemma.sandbox import SubprocessManager

# Sandbox venvs normally inherit the harness's own site-packages, which ship
# their own httpx/requests and shadow the repo under test. Point them at a
# clean env holding only what the Docker image has (pytest + build backends).
TEST_BASE = os.environ.get('SWEGEMMA_TEST_BASE', '.testbase')


class CleanSubprocessManager(SubprocessManager):
    def __init__(self, **kw):
        super().__init__(system_site_packages=False, **kw)

    def start(self, *a, **kw):
        sid = super().start(*a, **kw)
        venv = self._sandboxes[sid]['venv']
        for sp in venv.glob('lib/python*/site-packages'):
            for base in Path(TEST_BASE).glob('lib/python*/site-packages'):
                (sp / '_test_base.pth').write_text(f'{base}\n', encoding='utf-8')
        return sid


# The official image streams each repo's runtime and test dependencies from a
# private cache; the public wheels only cover part of them. Locally, install
# the repo with its dependencies (competition wheels first, then PyPI) and any
# dev requirements files the repo ships.
_orig_install = container_setup.install_test_dependencies
PIP = 'python3 -m pip install -q --find-links=/wheels'
DEP_CMD = (
    f'{PIP} --no-index -e . || {PIP} -e . ; '
    'for f in requirements-dev.txt requirements/*.txt requirements-tests.txt; do '
    f'[ -f "$f" ] && ({PIP} --no-index -r "$f" || {PIP} -r "$f"); done; '
    # httpbin (requests' test server) breaks on werkzeug >= 2.1
    "python3 -m pip show -q httpbin >/dev/null 2>&1 && " + PIP + " 'werkzeug<2.1' 'flask<2.1'; true"
)


# Tests run offline in the official sandbox, so keep proxy settings out of
# the test environment and hand them only to the dependency install.
PROXY_VARS = {k: os.environ.pop(k) for k in list(os.environ)
              if k.lower() in ('http_proxy', 'https_proxy', 'all_proxy')}


def install_test_dependencies(docker, container_id, repo='', **kw):
    _orig_install(docker, container_id, repo, **kw)
    if isinstance(docker, CleanSubprocessManager):
        exports = ''.join(f'export {k}={shlex.quote(v)}; ' for k, v in PROXY_VARS.items())
        docker.exec(container_id, exports + DEP_CMD)


container_setup.install_test_dependencies = install_test_dependencies
verification.install_test_dependencies = install_test_dependencies


async def main() -> None:
    import logging; logging.basicConfig(level=os.environ.get('LOGLEVEL', 'WARNING'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data'))
    ap.add_argument('--task-ids', nargs='*')
    ap.add_argument('--empty', action='store_true', help='verify with no patch (should fail)')
    a = ap.parse_args()

    cfg = EvalConfig(
        tasks_path=a.data / 'tasks.jsonl',
        snapshots_dir=a.data / 'snapshots',
        results_dir=Path('results/gold'),
        submission_dir=Path('submission'),
        models=ModelRegistry(),
        sandbox='subprocess',
        wheels_dir=a.data / 'wheels',
        graph_dir=str(a.data / 'graphs'),
        embeddings_dir=str(a.data / 'embeddings'),
    )
    sandbox = CleanSubprocessManager(timeout_seconds=300)
    tasks = load_tasks(cfg.tasks_path)
    if a.task_ids:
        tasks = [t for t in tasks if t.instance_id in set(a.task_ids)]
    for t in tasks:
        snap, base, patch = resolve_task_snapshot_paths(cfg.snapshots_dir, t.instance_id, t.repo)
        if not snap.exists():
            print(json.dumps({'id': t.instance_id, 'skipped': 'no snapshot'}))
            continue
        res = await verify_task(sandbox, cfg, t, snap, base_snapshot_path=base, patch_path=patch,
                                agent_patch='' if a.empty else t.patch)
        print(json.dumps({'id': t.instance_id, 'resolved': res.resolved, 'error': res.error,
                          'seconds': round(res.duration_seconds, 1)}))
        print((res.test_output or '')[-3000:])


if __name__ == '__main__':
    asyncio.run(main())
