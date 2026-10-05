"""Run a submission on dev tasks locally, with Gemma served by Google's Gemini API.

    GEMINI_API_KEY=... SWEGEMMA_TEST_BASE=... .venv/bin/python scripts/api_eval.py \
        --name v1 --tasks eval/dev24.txt [--submission submission] [--concurrency 2]
    .venv/bin/python scripts/api_eval.py --check   # list Gemma models the key can use

No GPU and no Kaggle queue, so this is the fast loop for comparing agent
variants. It is a proxy, not the scored setup: the API serves full-precision
Gemma rather than the QAT w4a16 build, and its latency differs from 4x L4, so
wall-clock budgets do not transfer (use --time-minutes to loosen them). Confirm
finalists with scripts/push_eval.py.

Patches are graded with scripts/verify_gold.py --patches results/<name>.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import verify_gold  # noqa: E402,F401  (applies the local sandbox patches)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'


def check() -> None:
    from google import genai
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    for m in client.models.list():
        if 'gemma' in m.name.lower():
            print(m.name, getattr(m, 'input_token_limit', ''), getattr(m, 'supported_actions', ''))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='list Gemma models available to the key')
    ap.add_argument('--name')
    ap.add_argument('--tasks', type=Path, default=ROOT / 'eval/dev24.txt')
    ap.add_argument('--submission', type=Path, default=ROOT / 'submission')
    ap.add_argument('--model', default='gemini/gemma-4-31b-it', help='LiteLLM model id')
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--time-minutes', type=float, help='override max_time_minutes from eval_config.yaml')
    a = ap.parse_args()
    if 'GEMINI_API_KEY' not in os.environ:
        raise SystemExit('GEMINI_API_KEY is not set')
    if a.check:
        check()
        return
    if not a.name:
        raise SystemExit('--name is required')

    import litellm
    from adk_submission import ModelRegistry
    from google.adk.apps._configs import EventsCompactionConfig
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    litellm.drop_params = True
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    models = ModelRegistry()
    # Free-tier rate limits are tight; let LiteLLM back off and retry.
    models.register(TARGET_MODEL_NAME, LiteLlm(model=a.model, num_retries=6))

    section = yaml.safe_load((a.submission / 'eval_config.yaml').read_text())['evaluation']
    limits, gen = build_submission_limits()
    task_ids = [t for t in a.tasks.read_text().split() if t]
    out = ROOT / 'results' / a.name
    config = EvalConfig(
        tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots', results_dir=out,
        submission_dir=a.submission, models=models, sandbox='subprocess',
        timeout_seconds=int(section.get('timeout_seconds', 300)),
        max_time_minutes=a.time_minutes or float(section.get('max_time_minutes', 60)),
        max_tool_calls=int(section.get('max_tool_calls', 100)),
        max_turns=section.get('max_turns'),
        task_ids=task_ids, limits=limits, generation_constraints=gen,
        # Same compaction as the scorer, so context behaves as it will on Kaggle.
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=5, overlap_size=2, token_threshold=14336, event_retention_size=5
        ),
        graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'),
        wheels_dir=DATA / 'wheels', concurrency=a.concurrency, display_mode='quiet',
    )
    t0 = time.time()
    result = asyncio.run(Evaluator(config).run())
    summary = {'run': a.name, 'model': a.model, 'tasks': len(task_ids),
               'wall_seconds': round(time.time() - t0),
               'resolved': result.resolved, 'resolution_rate': result.resolution_rate}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
