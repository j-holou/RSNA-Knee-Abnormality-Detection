"""Push a CPU kernel that writes submission.zip, then submit it to the competition.

    python scripts/make_submission.py --ref <commit> --out /tmp/v1.zip
    python scripts/push_submission.py --zip /tmp/v1.zip --name v1          # push kernel
    kaggle kernels status <user>/gemma-submit-v1                            # wait for COMPLETE
    kaggle competitions submit gemma-4-developer-agent -k <user>/gemma-submit-v1 \
        -v <version> -f submission.zip -m "v1"

This is a code competition: Kaggle takes submission.zip from a notebook's
output and runs the scoring itself, so the kernel only has to write the file.
"""
import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

KERNEL = '''import base64
from pathlib import Path
Path("/kaggle/working/submission.zip").write_bytes(base64.b64decode(ZIP_B64))
print("wrote submission.zip", len(base64.b64decode(ZIP_B64)), "bytes")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    user = os.environ.get('KAGGLE_USERNAME', 'janholoubek')
    b64 = base64.b64encode(a.zip.read_bytes()).decode()
    meta = {
        'id': f'{user}/gemma-submit-{a.name}', 'title': f'gemma-submit-{a.name}',
        'code_file': 'submit.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': False, 'enable_internet': False,
        'competition_sources': ['gemma-4-developer-agent'],
        'dataset_sources': [], 'kernel_sources': [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'submit.py').write_text(f'ZIP_B64 = {b64!r}\n' + KERNEL)
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
