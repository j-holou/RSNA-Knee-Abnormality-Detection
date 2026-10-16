"""Push the LoRA pipeline kernels to Kaggle.

    python scripts/push_lora.py --stage prep     # CPU: render trajectories -> janholoubek/gemma-lora-prep

The prep kernel mounts the gemma-oracle-traj output and the bf16 QAT twin of the scored
model (google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2).
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from push_eval import ROOT, kaggle_username

STAGES = {
    'prep': {'code': 'kaggle_lora_prep.py', 'gpu': False,
             'kernel_sources': ['{user}/gemma-oracle-traj']},
}
MODEL = 'google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=sorted(STAGES), required=True)
    a = ap.parse_args()
    st = STAGES[a.stage]
    user = kaggle_username()
    meta = {
        'id': f'{user}/gemma-lora-{a.stage}', 'title': f'gemma-lora-{a.stage}', 'code_file': st['code'],
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': st['gpu'], 'enable_internet': True,
        'dataset_sources': [], 'competition_sources': [],
        'kernel_sources': [s.format(user=user) for s in st['kernel_sources']],
        'model_sources': [MODEL],
    }
    if st['gpu']:
        meta['machine_shape'] = 'NvidiaL4'
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / st['code']).write_text((ROOT / 'scripts' / st['code']).read_text(encoding='utf-8'))
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        print(meta['id'])
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
