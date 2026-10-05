"""Kaggle kernel: run our agent on a set of public tasks with the real model.

Not run directly. scripts/push_eval.py fills in SUBMISSION_FILES and TASK_IDS
and pushes the result as a GPU kernel on Kaggle's 4x L4 machines. Results land
in /kaggle/working/results (summary.json, task_results.jsonl, patches/, logs/,
traces/), which `kaggle kernels output` downloads.

Setup follows the official "Getting Started - Gemma 4 Developer Agent" notebook.
"""
import asyncio
import glob
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUBMISSION_FILES: dict[str, str] = {}  # filled in by push_eval.py
TASK_IDS: list[str] = []  # filled in by push_eval.py
RUN_NAME = ''  # filled in by push_eval.py

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '1200'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

WORKING_DIR = Path('/kaggle/working')
RESULTS_DIR = WORKING_DIR / 'results'
T0 = time.time()


def log(msg: str) -> None:
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


def find_dir(*patterns: str) -> Path:
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f'none of {patterns} exist')


# ---------------------------------------------------------------- install harness
WHEELHOUSE_DIR = find_dir(
    '/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse',
    '/kaggle/input/gemma-4-developer-agent-wheelhouse',
)
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    try:
        os.unlink(pth)
    except OSError:
        pass
# Kaggle dataset uploads strip the '+' from local version tags like '+cu128'.
tmp_whl = Path('/tmp/wheelhouse')
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    if not (tmp_whl / name).exists():
        os.symlink(w, tmp_whl / name)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
     *sorted(str(w) for w in tmp_whl.glob('*.whl'))],
    check=True,
)
importlib.invalidate_caches()
log('harness installed')

# ---------------------------------------------------------------- submission + data
AGENT_DIR = WORKING_DIR / 'submission'
for rel, content in SUBMISSION_FILES.items():
    path = AGENT_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

DATA_DIR = find_dir(
    '/kaggle/input/competitions/gemma-4-developer-agent',
    '/kaggle/input/gemma-4-developer-agent',
)
MODEL_PATH = find_dir(
    '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
    '/kaggle/input/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
)
log(f'data={DATA_DIR} model={MODEL_PATH} tasks={len(TASK_IDS)}')

# ---------------------------------------------------------------- model server
import litellm  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models.discovery import validate_single_declared_model  # noqa: E402

litellm.drop_params = True
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
declared_model = validate_single_declared_model(AGENT_DIR)
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count()
log(f'gpus={gpu_count}')
server = VllmServer(
    VllmConfig(
        model=str(MODEL_PATH),
        port=8000,
        host='127.0.0.1',
        tool_call_parser='gemma4',
        reasoning_parser='gemma4',
        max_model_len=32768,
        dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
        # Match the scorer (0.80), not the starter notebook (0.90).
        gpu_memory_utilization=0.80,
        enable_auto_tool_choice=True,
        enable_lora=True,
        max_loras=8,
        max_lora_rank=128,
        tensor_parallel_size=4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1),
        startup_timeout=60 * 20,
    ),
    adapter_manifest=adapters,
)
server.start()
models = server.create_model_registry(
    aliases=[declared_model, TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY'
)
log('vLLM server up')

# ---------------------------------------------------------------- evaluate
section = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text())['evaluation']
limits, gen_constraints = build_submission_limits()
config = EvalConfig(
    tasks_path=DATA_DIR / 'tasks.jsonl',
    snapshots_dir=DATA_DIR / 'snapshots',
    results_dir=RESULTS_DIR,
    submission_dir=AGENT_DIR,
    models=models,
    sandbox='subprocess',
    timeout_seconds=int(section.get('timeout_seconds', 300)),
    max_time_minutes=float(section.get('max_time_minutes', 60)),
    max_tool_calls=int(section.get('max_tool_calls', 100)),
    max_turns=section.get('max_turns'),
    task_ids=TASK_IDS,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    # Same compaction and caching settings as the scorer's scripts/inference.py.
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5, overlap_size=2, token_threshold=14336, event_retention_size=5
    ),
    graph_dir=str(DATA_DIR / 'graphs'),
    embeddings_dir=str(DATA_DIR / 'embeddings'),
    wheels_dir=DATA_DIR / 'wheels',
    display_mode='quiet',
)
result = asyncio.run(Evaluator(config).run())
log('evaluation done')

summary = {
    'run': RUN_NAME,
    'tasks': len(TASK_IDS),
    'wall_seconds': round(time.time() - T0),
    'resolved': result.resolved,
    'resolution_rate': result.resolution_rate,
}
(WORKING_DIR / 'run_summary.json').write_text(json.dumps(summary, indent=2))
log(json.dumps(summary))
server.stop()
