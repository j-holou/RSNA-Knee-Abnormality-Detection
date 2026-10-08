"""Package an agent directory as submission.zip for the Gemma 4 Developer Agent competition.

    python scripts/make_submission.py [--submission submission] [--ref <git ref>] [--out submission.zip]

With --ref, packages submission/ as it was at that commit (e.g. the v1 commit)
instead of the working tree. Files sit at the archive root (agent.yaml, prompts/,
skills/, ...), which is the layout the harness expects.
"""
import argparse
import io
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', default='submission')
    ap.add_argument('--ref', help='git ref to package submission/ from')
    ap.add_argument('--out', type=Path, default=ROOT / 'submission.zip')
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if a.ref:
            data = subprocess.run(['git', 'archive', a.ref, a.submission], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            tarfile.open(fileobj=io.BytesIO(data)).extractall(tmp, filter='data')
            src = Path(tmp) / a.submission
        else:
            src = ROOT / a.submission
        files = sorted(p for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        bad = [p for p in files if p.suffix not in ALLOWED]
        if bad:
            raise SystemExit(f'disallowed file types: {bad}')
        with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.relative_to(src))
        print(f'{a.out}: {len(files)} files')
        for p in files:
            print('  ', p.relative_to(src))


if __name__ == '__main__':
    main()
