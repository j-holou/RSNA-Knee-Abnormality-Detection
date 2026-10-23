"""Push a CPU kernel that regrades saved agent patches in a scorer-like sandbox.

    python scripts/push_regrade.py --name a --runs gold v6 v6b v7 --tasks eval/dev48.txt eval/rich44.txt
    kaggle kernels logs <user>/gemma-regrade-a | grep -E 'REGRADE|SUMMARY'

'gold' grades the reference patches of the listed tasks (the ceiling check); every other
run name mounts <user>/gemma-eval-<run> and grades all patches in its results/patches/.
See scripts/kaggle_regrade.py for the sandbox. CPU kernels use no GPU quota.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from push_eval import ROOT, kaggle_username


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--tasks', nargs='+', type=Path, default=[ROOT / 'eval/dev48.txt'])
    a = ap.parse_args()
    task_ids = sorted({t for f in a.tasks for t in f.read_text().split() if t})
    user = kaggle_username()
    src = (ROOT / 'scripts/kaggle_regrade.py').read_text(encoding='utf-8')
    src = src.replace('RUNS: list[str] = []', f'RUNS: list[str] = {a.runs!r}', 1)
    src = src.replace('TASK_IDS: list[str] = []', f'TASK_IDS: list[str] = {task_ids!r}', 1)
    meta = {
        'id': f'{user}/gemma-regrade-{a.name}', 'title': f'gemma-regrade-{a.name}', 'code_file': 'regrade.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['gemma-4-developer-agent'],
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse',
                            'zbigniewkirchner/gemma-4-agent-local-eval-wheels',
                            'zbigniewkirchner/gemma-4-agent-local-eval-pythons'],
        'kernel_sources': [f'{user}/gemma-eval-{r}' for r in a.runs if r != 'gold'],
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'regrade.py').write_text(src)
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        print(meta['id'], len(task_ids), 'tasks', a.runs)
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
