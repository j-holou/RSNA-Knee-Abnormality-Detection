"""Generate "oracle" agent trajectories from the gold patches, for LoRA training data.

    SWEGEMMA_TEST_BASE=... .venv/bin/python scripts/oracle_traj.py --tasks eval/sound.txt --out results/oracle

A scripted model plays our agent through the real harness (our submission's prompt,
tools and skill scripts), making the moves we want Gemma to learn: one locate.py call,
one short read around each changed region, an edit_file per hunk by call ~6, check_patch,
submit_patch. The harness grades each run with the hidden tests; only resolved runs are
kept. Each kept run is written as one JSON line holding the system instruction, the tool
declarations and the full conversation (model function calls and tool results), ready to
render with Gemma's chat template.
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import AsyncGenerator

from adk_submission import ModelRegistry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CONTEXT = 2  # context lines kept around each hunk's changed lines in old_string


def parse_patch(patch: str) -> list[dict]:
    """Split a unified diff into files and hunks: {'path', 'new_file', 'hunks': [(start, old, new)]}."""
    files, cur, hunk = [], None, None
    for line in patch.splitlines():
        if line.startswith('--- '):
            cur = {'path': None, 'new_file': line.startswith('--- /dev/null'), 'hunks': []}
            files.append(cur)
        elif line.startswith('+++ ') and cur is not None:
            cur['path'] = line[4:].split('\t')[0].removeprefix('b/')
        elif line.startswith('@@') and cur is not None:
            start = int(re.match(r'@@ -(\d+)', line).group(1))
            hunk = [start, [], []]
            cur['hunks'].append(hunk)
        elif hunk is not None and line[:1] in (' ', '-', '+'):
            if line[0] in ' -':
                hunk[1].append((line[0], line[1:]))
            if line[0] in ' +':
                hunk[2].append((line[0], line[1:]))
        elif hunk is not None and line == '':
            hunk[1].append((' ', '')); hunk[2].append((' ', ''))
    return files


def trim(old, new):
    """Keep only CONTEXT unchanged lines on each side of the changed lines."""
    def first_change(seq):
        return next((i for i, (k, _) in enumerate(seq) if k != ' '), len(seq))
    def last_change(seq):
        return next((len(seq) - i for i, (k, _) in enumerate(reversed(seq)) if k != ' '), 0)
    lead = min(first_change(old), first_change(new))
    tail = min(len(old) - last_change(old), len(new) - last_change(new))
    drop_lead = max(0, lead - CONTEXT)
    drop_tail = max(0, tail - CONTEXT)
    o = old[drop_lead:len(old) - drop_tail]
    n = new[drop_lead:len(new) - drop_tail]
    return drop_lead, '\n'.join(t for _, t in o), '\n'.join(t for _, t in n)


def plan_calls(task: dict) -> list | None:
    files = [f for f in parse_patch(task['patch']) if f['path']]
    if not files or any(f['path'].startswith(('tests/', 'test/')) for f in files):
        return None
    title = task['problem_statement'].strip().splitlines()[0][:200]
    names = sorted(set(re.findall(r'`([A-Za-z_][\w.]{2,60})`', task['problem_statement'])))[:6]
    calls = [('run_skill_script', {'skill_name': 'swe-tools', 'file_path': 'scripts/locate.py',
                                   'args': {'query': ' '.join([title] + names)}})]
    reads, edits = [], []
    for f in files:
        if f['new_file']:
            body = '\n'.join(t for _, t in f['hunks'][0][2]) + '\n'
            edits.append(('write_file', {'filepath': f['path'], 'content': body}))
            continue
        first = f['hunks'][0][0]
        last_h = f['hunks'][-1]
        end = last_h[0] + len(last_h[1])
        if end - first <= 80:
            reads.append(('read_file', {'filepath': f['path'], 'start_line': max(1, first - 10), 'end_line': end + 10}))
        else:
            for start, old, _ in f['hunks'][:3]:
                reads.append(('read_file', {'filepath': f['path'], 'start_line': max(1, start - 10),
                                            'end_line': start + len(old) + 10}))
        for start, old, new in f['hunks']:
            _, o, n = trim(old, new)
            if not o.strip():
                return None  # pure insertion with no anchor; skip rather than guess
            edits.append(('edit_file', {'filepath': f['path'], 'old_string': o, 'new_string': n}))
    if len(reads) > 4 or len(edits) > 8:
        return None  # large multi-file changes don't fit the edit-early pattern
    calls += reads + edits
    calls += [('run_skill_script', {'skill_name': 'swe-tools', 'file_path': 'scripts/check_patch.py',
                                    'args': {'clean': 'true'}}),
              ('submit_patch', {}),
              ('TEXT', f'Fixed: {title}')]
    return calls


# Module-level state: ADK copies the model object for each agent, so instance
# attributes would not be shared with the copy that actually runs.
CALLS: list = []
STEP = [0]
LOG: list = []


class OracleLlm(BaseLlm):

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        LOG.append(llm_request)
        if STEP[0] < len(CALLS):
            name, args = CALLS[STEP[0]]
            STEP[0] += 1
            part = types.Part(text=args) if name == 'TEXT' else \
                types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text='done')
        yield LlmResponse(content=types.Content(role='model', parts=[part]))


def changed_lines(patch: str) -> list[str]:
    return [l for l in patch.splitlines()
            if l[:1] in '+-' and not l.startswith(('+++', '---'))]


def dump_conversation(last_text: str) -> dict:
    req = LOG[-1]
    tools = []
    for t in (req.config.tools or []):
        for fd in (t.function_declarations or []):
            tools.append(fd.model_dump(exclude_none=True, mode='json'))
    msgs = [c.model_dump(exclude_none=True, mode='json') for c in req.contents]
    msgs.append({'role': 'model', 'parts': [{'text': last_text}]})
    return {'system': str(req.config.system_instruction or ''), 'tools': tools, 'contents': msgs}


async def run_task(task: dict, out: Path, submission: Path) -> dict:
    calls = plan_calls(task)
    if calls is None:
        return {'instance_id': task['instance_id'], 'status': 'skipped'}
    CALLS[:] = calls
    STEP[0] = 0
    LOG.clear()
    llm = OracleLlm(model='oracle')
    models = ModelRegistry()
    models.register('gemma-4-31b-it-qat-w4a16-ct', llm)
    limits, gen = build_submission_limits()
    res_dir = out / 'runs' / task['instance_id']
    config = EvalConfig(
        tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots', results_dir=res_dir,
        submission_dir=submission, models=models, sandbox='subprocess',
        timeout_seconds=240, max_time_minutes=10, max_tool_calls=40, max_turns=60,
        task_ids=[task['instance_id']], limits=limits, generation_constraints=gen,
        graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'),
        wheels_dir=DATA / 'wheels', display_mode='quiet',
    )
    result = await Evaluator(config).run()
    resolved = bool(result.resolved)
    # Local grading fails some fastapi tasks on missing test dependencies, so a patch
    # whose changed lines equal the gold patch's also counts.
    patch_file = res_dir / 'patches' / f"{task['instance_id']}.patch"
    same = patch_file.exists() and changed_lines(patch_file.read_text()) == changed_lines(task['patch'])
    status = 'resolved' if resolved else ('gold_match' if same else 'failed')
    rec = {'instance_id': task['instance_id'], 'status': status,
           'n_calls': sum(1 for c in calls if c[0] != 'TEXT')}
    if status != 'failed' and LOG:
        rec['example'] = dump_conversation(calls[-1][1])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'oracle')
    ap.add_argument('--submission', type=Path, default=ROOT / 'submission')
    a = ap.parse_args()
    ids = set(a.tasks.read_text().split())
    tasks = [json.loads(l) for l in (DATA / 'tasks.jsonl').read_text().splitlines()]
    tasks = [t for t in tasks if t['instance_id'] in ids]
    have = {p.stem for p in (DATA / 'snapshots').glob('*.tgz')}
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with open(a.out / 'trajectories.jsonl', 'a') as fout, open(a.out / 'status.jsonl', 'a') as fstat:
        for t in tasks:
            if t['instance_id'] not in have:
                continue
            rec = asyncio.run(run_task(t, a.out, a.submission))
            stats[rec['status']] = stats.get(rec['status'], 0) + 1
            ex = rec.pop('example', None)
            if ex:
                fout.write(json.dumps({'instance_id': t['instance_id'], **ex}) + '\n')
            fstat.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
    print('SUMMARY', json.dumps(stats))


if __name__ == '__main__':
    sys.exit(main())
