"""Run scripts/bench_locate2.py for several locator versions on all dev tasks, in a CPU kernel.

    python scripts/push_bench_locate.py name=path/to/locate.py [name=path ...]
    kaggle kernels logs <user>/gemma-bench-locate | grep -E '^(VARIANT|SUMMARY)'

The kernel mounts the competition data (tasks.jsonl and every task snapshot). CPU only.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from push_eval import ROOT, kaggle_username

KERNEL = r'''
import glob, os, subprocess, sys
from pathlib import Path
SOURCES = __SOURCES__
DATA = Path(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)[0]).parent
work = Path("/kaggle/working/bench")
work.mkdir(parents=True, exist_ok=True)
for name, text in SOURCES.items():
    (work / name).write_text(text)
for name in SOURCES:
    if name == "bench_locate2.py":
        continue
    print("VARIANT", name, flush=True)
    r = subprocess.run([sys.executable, str(work / "bench_locate2.py"), "--data", str(DATA),
                        "--locate", str(work / name), "--cache", "/tmp/srccache"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:], flush=True)
'''


def main() -> None:
    variants = dict(a.split('=', 1) for a in sys.argv[1:])
    sources = {'bench_locate2.py': (ROOT / 'scripts/bench_locate2.py').read_text()}
    for name, path in variants.items():
        sources[f'{name}.py'] = Path(path).read_text()
    user = kaggle_username()
    meta = {
        'id': f'{user}/gemma-bench-locate', 'title': 'gemma-bench-locate', 'code_file': 'bench.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['gemma-4-developer-agent'], 'dataset_sources': [], 'kernel_sources': [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        src = KERNEL.replace('__SOURCES__', repr(sources))
        compile(src, 'bench.py', 'exec')
        (Path(tmp) / 'bench.py').write_text(src)
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
