#!/usr/bin/env bash
# Sets up a local dev environment for the Gemma 4 Developer Agent competition.
# Needs: kaggle CLI with KAGGLE_API_TOKEN, uv, and competition rules accepted.
#
#   scripts/setup_env.sh            # harness + tasks/graphs/wheels + 1 repo snapshot per project
#   ALL_SNAPSHOTS=1 scripts/setup_env.sh   # all 129 snapshots (~21.5 GB)
set -euo pipefail
cd "$(dirname "$0")/.."

COMP=gemma-4-developer-agent
VENV=${SWEGEMMA_VENV:-.venv}
TEST_BASE=${SWEGEMMA_TEST_BASE:-.testbase}

# Harness (swegemma needs Python >= 3.12). Its wheels live in a Kaggle dataset, not PyPI.
mkdir -p .cache/wheelhouse
for w in adk_eval_core-0.1.0 adk_submission-0.2.12 swegemma-0.2.7; do
  f=.cache/wheelhouse/$w-py3-none-any.whl
  [ -f "$f" ] || kaggle datasets download metric/gemma-4-developer-agent-wheelhouse \
    -f "$w-py3-none-any.whl" -p .cache/wheelhouse -q
done
uv venv -q -p 3.13 "$VENV"
VIRTUAL_ENV="$VENV" uv pip install -q .cache/wheelhouse/*.whl

# Clean interpreter base for task sandboxes: only what the official Docker image has.
uv venv -q -p 3.13 "$TEST_BASE"
VIRTUAL_ENV="$TEST_BASE" uv pip install -q pytest pytest-timeout==2.1.0 typer pdm-backend \
  setuptools wheel poetry-core hatchling flit-core editables

# Competition data. The full zip is 22 GB, almost all of it repo snapshots,
# so pull it once and keep only what is asked for.
if [ ! -f data/tasks.jsonl ]; then
  mkdir -p data
  kaggle competitions download "$COMP" -p data -q
  "$VENV/bin/python" - <<'PY'
import os, zipfile
z = zipfile.ZipFile('data/gemma-4-developer-agent.zip')
keep_all = os.environ.get('ALL_SNAPSHOTS') == '1'
smallest = {}
for i in z.infolist():
    if i.filename.startswith('snapshots/'):
        repo = i.filename.split('/')[1].split('_')[0]
        if repo not in smallest or i.file_size < smallest[repo].file_size:
            smallest[repo] = i
for i in z.infolist():
    if keep_all or not i.filename.startswith('snapshots/') or i in smallest.values():
        z.extract(i, 'data')
PY
  rm -f data/gemma-4-developer-agent.zip
fi
echo "Ready. Try: $VENV/bin/python scripts/verify_gold.py --task-ids httpx_3672"
