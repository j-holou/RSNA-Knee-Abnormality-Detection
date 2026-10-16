"""Push a CPU Kaggle kernel that runs scripts/oracle_traj.py on the public tasks.

    python scripts/push_oracle.py --tasks eval/sound.txt          # janholoubek/gemma-oracle-traj

The kernel reuses the harness setup from scripts/kaggle_eval.py (no model server), writes
our submission/ and runs every task's gold-patch trajectory. Results go to
/kaggle/working/oracle/trajectories.jsonl, which a training kernel can mount as a
kernel source; the log prints one status line per task. CPU kernels use no GPU quota.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from push_eval import ROOT, TEXT_SUFFIXES, kaggle_username


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, default=ROOT / 'eval/sound.txt')
    ap.add_argument('--submission', type=Path, default=ROOT / 'submission')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    files = {str(p.relative_to(a.submission)): p.read_text(encoding='utf-8')
             for p in sorted(a.submission.rglob('*')) if p.is_file() and p.suffix in TEXT_SUFFIXES}
    task_ids = [t for t in a.tasks.read_text().split() if t]

    setup = (ROOT / 'scripts/kaggle_eval.py').read_text(encoding='utf-8')
    setup = setup[:setup.index('# ---------------------------------------------------------------- model server')]
    setup = setup.replace('SUBMISSION_FILES: dict[str, str] = {}', f'SUBMISSION_FILES: dict[str, str] = {files!r}', 1)
    setup = setup.replace('''MODEL_PATH = find_dir(
    '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
    '/kaggle/input/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
)
log(f'data={DATA_DIR} model={MODEL_PATH} tasks={len(TASK_IDS)}')''', "log(f'data={DATA_DIR} tasks={len(TASK_IDS)}')")
    oracle = (ROOT / 'scripts/oracle_traj.py').read_text(encoding='utf-8')
    runner = f'''
(WORKING_DIR / 'oracle_traj.py').write_text({oracle!r})
(WORKING_DIR / 'tasks.txt').write_text({chr(10).join(task_ids)!r})
env = dict(os.environ, ORACLE_DATA=str(DATA_DIR))
subprocess.run([sys.executable, str(WORKING_DIR / 'oracle_traj.py'), '--tasks', str(WORKING_DIR / 'tasks.txt'),
                '--out', str(WORKING_DIR / 'oracle'), '--submission', str(AGENT_DIR)], env=env, check=True)
import shutil
shutil.rmtree(WORKING_DIR / 'oracle' / 'runs', ignore_errors=True)  # keep only the jsonl outputs
log('done')
'''
    source = setup + runner
    user = kaggle_username()
    meta = {
        'id': f'{user}/gemma-oracle-traj', 'title': 'gemma-oracle-traj', 'code_file': 'oracle_kernel.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:'
                        '37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'docker_image_pinning_type': 'original',
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': [], 'model_sources': [],
    }
    compile(source, 'oracle_kernel.py', 'exec')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'oracle_kernel.py').write_text(source, encoding='utf-8')
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        print(f'{meta["id"]}: {len(files)} submission files, {len(task_ids)} tasks')
        if not a.dry_run:
            subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
