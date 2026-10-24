"""Push a CPU kernel that checks the scorer-like sandbox in the agent phase, no GPU needed.

    python scripts/push_envcheck.py --tasks rich_3296 requests_7205 fastapi_12942
    kaggle kernels logs <user>/gemma-envcheck | grep -E 'ENVCHECK|OBS'

Runs kaggle_eval.py's harness setup plus scorer_env.py, then for each task a scripted
agent that (1) runs pytest on an existing test file the task's tests touch, (2) applies
the gold patch with git apply, (3) runs the same pytest again, (4) calls submit_patch.
The pytest outputs show whether the agent's own test runs work as on the scorer, and the
grade shows whether the full agent + grading path resolves a correct patch.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from push_eval import ROOT, TEXT_SUFFIXES, kaggle_username

RUNNER = r'''
# ---------------------------------------------------------------- env check
from typing import AsyncGenerator  # noqa: E402
from adk_submission import ModelRegistry  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402
from swegemma.config import EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402

CALLS, STEP = [], [0]


class ScriptLlm(BaseLlm):
    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        for c in llm_request.contents[-1:]:
            for p in c.parts or []:
                if p.function_response:
                    print('OBS', CURRENT['id'], str(p.function_response.response)[-1500:], flush=True)
        if STEP[0] < len(CALLS):
            name, args = CALLS[STEP[0]]
            STEP[0] += 1
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text='done')
        yield LlmResponse(content=types.Content(role='model', parts=[part]))


class ScorerLikeEvaluator(Evaluator):
    async def evaluate_task(self, task, *args, **kwargs):
        CURRENT['id'] = task.instance_id
        return await super().evaluate_task(task, *args, **kwargs)


tasks = {json.loads(l)['instance_id']: json.loads(l) for l in (DATA_DIR / 'tasks.jsonl').read_text().splitlines()}
for tid in TASK_IDS:
    t = tasks[tid]
    tests = [l[6:] for l in t['test_patch'].splitlines() if l.startswith('--- a/')]
    target = tests[0] if tests else 'tests'
    pytest_cmd = f'cd /workspace && timeout 200 python -m pytest {target} -q -x -p no:cacheprovider 2>&1 | tail -8'
    CALLS[:] = [
        ('run_command', {'command': 'cd /workspace && which python && python --version && python -c "import pytest; print(pytest.__version__)"'}),
        ('run_command', {'command': pytest_cmd}),
        ('run_command', {'command': "cd /workspace && cat > /tmp/gold.patch <<'GOLD_EOF'\n" + t['patch'] + "\nGOLD_EOF\ngit apply /tmp/gold.patch && git status --short"}),
        ('run_command', {'command': pytest_cmd}),
        ('submit_patch', {}),
    ]
    STEP[0] = 0
    models = ModelRegistry()
    models.register('gemma-4-31b-it-qat-w4a16-ct', ScriptLlm(model='script'))
    limits, gen = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots',
        results_dir=WORKING_DIR / 'envcheck' / tid, submission_dir=AGENT_DIR, models=models, sandbox='subprocess',
        timeout_seconds=240, max_time_minutes=10, max_tool_calls=20, max_turns=40, task_ids=[tid],
        limits=limits, generation_constraints=gen, graph_dir=str(DATA_DIR / 'graphs'),
        embeddings_dir=str(DATA_DIR / 'embeddings'), wheels_dir=DATA_DIR / 'wheels', display_mode='quiet',
    )
    ev = ScorerLikeEvaluator(cfg)
    ev.sandbox = ev.docker = SubprocessManager(timeout_seconds=240, system_site_packages=False)
    try:
        res = asyncio.run(ev.run())
        print('ENVCHECK', tid, 'resolved', res.resolved, flush=True)
    except Exception as e:
        print('ENVCHECK', tid, 'ERROR', repr(e)[:300], flush=True)
log('done')
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', required=True)
    ap.add_argument('--submission', type=Path, default=ROOT / 'submission')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    files = {str(p.relative_to(a.submission)): p.read_text(encoding='utf-8')
             for p in sorted(a.submission.rglob('*')) if p.is_file() and p.suffix in TEXT_SUFFIXES}
    setup = (ROOT / 'scripts/kaggle_eval.py').read_text(encoding='utf-8')
    setup = setup[:setup.index('# ---------------------------------------------------------------- model server')]
    setup = setup.replace('SUBMISSION_FILES: dict[str, str] = {}', f'SUBMISSION_FILES: dict[str, str] = {files!r}', 1)
    setup = setup.replace('TASK_IDS: list[str] = []', f'TASK_IDS: list[str] = {a.tasks!r}', 1)
    setup = setup.replace('''MODEL_PATH = find_dir(
    '/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
    '/kaggle/input/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/*',
)
log(f'data={DATA_DIR} model={MODEL_PATH} tasks={len(TASK_IDS)}')''', "log(f'data={DATA_DIR} tasks={len(TASK_IDS)}')")
    source = setup + '\n' + (ROOT / 'scripts/scorer_env.py').read_text(encoding='utf-8') + RUNNER
    meta = {
        'id': f'{kaggle_username()}/gemma-envcheck', 'title': 'gemma-envcheck', 'code_file': 'envcheck.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:'
                        '37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
        'docker_image_pinning_type': 'original',
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse',
                            'zbigniewkirchner/gemma-4-agent-local-eval-wheels',
                            'zbigniewkirchner/gemma-4-agent-local-eval-pythons'],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': [], 'model_sources': [],
    }
    compile(source, 'envcheck.py', 'exec')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'envcheck.py').write_text(source, encoding='utf-8')
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        print(meta['id'], a.tasks)
        if not a.dry_run:
            subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
