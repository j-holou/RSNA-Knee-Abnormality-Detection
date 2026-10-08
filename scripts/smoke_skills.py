"""Drive the real harness with a scripted fake model to smoke-test submission/.

    SWEGEMMA_TEST_BASE=... .venv/bin/python scripts/smoke_skills.py rich_3063

The fake model issues a fixed list of tool calls (the three swe-tools scripts,
then submit_patch) so skill wiring, argument passing and the sandbox's working
directory can be checked without a GPU. Tool outputs are printed.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
import verify_gold  # noqa: E402,F401  (applies the local sandbox patches)

from adk_submission import ModelRegistry  # noqa: E402
from swegemma.config import EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

CALLS = [
    # finder
    ('list_skills', {}),
    ('run_skill_script', {'skill_name': 'swe-tools', 'file_path': 'scripts/locate.py',
                          'args': {'query': 'Table column width "overflow" Console.print'}}),
    ('TEXT', 'FILES: rich/table.py:1-5\nROOT CAUSE: smoke.\nCHANGE: add a comment.\nCHECK: python -c "import rich"'),
    # fixer
    ('run_command', {'command': 'cd /workspace && echo "# fixed" >> rich/table.py && echo x > repro.py'}),
    ('run_skill_script', {'skill_name': 'swe-tools', 'file_path': 'scripts/check_patch.py',
                          'args': {'clean': 'true'}}),
    ('submit_patch', {}),
]


STEP = [0]  # shared: ADK copies the model object for each agent


class ScriptedLlm(BaseLlm):

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        # Print the previous tool result so the run shows what each call returned.
        last = llm_request.contents[-1] if llm_request.contents else None
        for part in (last.parts or []) if last else []:
            if part.function_response:
                print(f'--- {part.function_response.name} ->')
                print(json.dumps(part.function_response.response, default=str)[:3000])
        si = str(getattr(llm_request.config, 'system_instruction', '') or '')
        print(f'>>> call step={STEP[0]} agent={si[:30]!r} ncontents={len(llm_request.contents)}')
        if '<brief>' in si and not getattr(self, '_shown', False):
            object.__setattr__(self, '_shown', True)
            print('--- fixer system instruction (excerpt) ->')
            print(si[si.index('<issue>'):si.index('</brief>')+8])
        if STEP[0] < len(CALLS):
            name, args = CALLS[STEP[0]]
            STEP[0] += 1
            if name == 'TEXT':
                content = types.Content(role='model', parts=[types.Part(text=args)])
            else:
                content = types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])
        else:
            content = types.Content(role='model', parts=[types.Part(text='done')])
        yield LlmResponse(content=content)


def main() -> None:
    task_id = sys.argv[1] if len(sys.argv) > 1 else 'rich_3063'
    models = ModelRegistry()
    models.register('gemma-4-31b-it-qat-w4a16-ct', ScriptedLlm(model='scripted'))
    limits, gen = build_submission_limits()
    out = Path('/tmp/claude-0/smoke-results')
    config = EvalConfig(
        tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots', results_dir=out,
        submission_dir=ROOT / 'submission', models=models, sandbox='subprocess',
        timeout_seconds=120, max_time_minutes=5, max_tool_calls=60, max_turns=100,
        task_ids=[task_id], limits=limits, generation_constraints=gen,
        graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'),
        wheels_dir=DATA / 'wheels', display_mode='quiet',
    )
    asyncio.run(Evaluator(config).run())
    for p in sorted((out / 'patches').glob('*.patch')):
        print(f'=== {p.name}\n{p.read_text()[:1500]}')


if __name__ == '__main__':
    main()
