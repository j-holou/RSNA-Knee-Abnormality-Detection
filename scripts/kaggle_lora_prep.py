"""Kaggle CPU kernel: render oracle trajectories with Gemma 4's chat template.

Pushed by scripts/push_lora.py --stage prep. Mounts the gemma-oracle-traj kernel output
and the bf16 QAT twin of the scored model (only the tokenizer files are read), converts
each trajectory to chat messages the way LiteLLM sends them to vLLM (assistant tool_calls,
role "tool" results), renders them with the model's own template, and marks which tokens
are trained on: only the assistant turns. Prints a rendered example and token stats, and
writes /kaggle/working/lora_data/train.jsonl (input_ids, labels) for the training kernel.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'jinja2'], check=True)
from transformers import AutoTokenizer  # noqa: E402

MAX_TOKENS = 16384
HOLDOUT_PREFIX = 'rich_'  # rich tasks are kept out of training for the paired eval


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


traj_path = find('/kaggle/input/**/oracle/trajectories.jsonl')
model_dir = find('/kaggle/input/**/gemma-4-31b-it-qat-q4_0-unquantized/**/tokenizer_config.json').parent
print('trajectories:', traj_path, '\nmodel:', model_dir, flush=True)
tok = AutoTokenizer.from_pretrained(model_dir)


def thought_for(name: str, args: dict) -> str:
    """Short synthetic reasoning placed before each call. It is masked out of the loss:
    the model keeps producing its own thoughts and only learns which call follows."""
    if name == 'run_skill_script' and 'locate' in str(args.get('file_path')):
        return 'First I will find the source files most related to the issue.'
    if name == 'read_file':
        return (f"The fix most likely belongs in {args.get('filepath')}. I will read lines "
                f"{args.get('start_line')}-{args.get('end_line')} around the relevant code.")
    if name in ('edit_file', 'write_file'):
        return f"I have seen enough of {args.get('filepath')} to make the change. Editing now instead of searching further."
    if name == 'run_skill_script':
        return 'The edit is in place. I will check the patch before submitting.'
    if name == 'submit_patch':
        return 'The patch is ready, so I will submit it.'
    return 'Next step.'


def to_openai(rec: dict):
    """genai contents -> OpenAI-style messages + tools, as LiteLLM sends them to vLLM."""
    def schema(s):
        if isinstance(s, dict):
            return {k: (v.lower() if k == 'type' and isinstance(v, str) else schema(v)) for k, v in s.items()}
        if isinstance(s, list):
            return [schema(x) for x in s]
        return s
    tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t.get('description', ''),
                                                'parameters': schema(t.get('parameters', {'type': 'object'}))}}
             for t in rec['tools']]
    msgs = [{'role': 'system', 'content': rec['system']}]
    n = 0
    for c in rec['contents']:
        for p in c['parts']:
            if 'text' in p:
                msgs.append({'role': 'assistant' if c['role'] == 'model' else 'user', 'content': p['text']})
            elif 'function_call' in p:
                n += 1
                fc = p['function_call']
                msgs.append({'role': 'assistant', 'content': '', 'reasoning_content': thought_for(fc['name'], fc.get('args', {})),
                             'tool_calls': [{
                    'id': f'call_{n}', 'type': 'function',
                    # vLLM parses the JSON string into a dict before rendering the template.
                    'function': {'name': fc['name'], 'arguments': fc.get('args', {})}}]})
            elif 'function_response' in p:
                fr = p['function_response']
                msgs.append({'role': 'tool', 'tool_call_id': f'call_{n}', 'name': fr['name'],
                             'content': json.dumps(fr.get('response', {}))})
    return msgs, tools


def render(msgs, tools) -> str:
    # Thinking is on when scored (include_thoughts: true), which adds <|think|> to the system turn.
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=True)


out_dir = Path('/kaggle/working/lora_data')
out_dir.mkdir(parents=True, exist_ok=True)
kept, held, too_long, lengths = 0, 0, 0, []
with open(out_dir / 'train.jsonl', 'w') as ftrain, open(out_dir / 'holdout_ids.txt', 'w') as fhold:
    for i, line in enumerate(open(traj_path)):
        rec = json.loads(line)
        if rec['instance_id'].startswith(HOLDOUT_PREFIX):
            fhold.write(rec['instance_id'] + '\n')
            held += 1
            continue
        msgs, tools = to_openai(rec)
        # Token-level mask: render growing prefixes; tokens added by assistant messages are trained.
        ids, labels, prev = [], [], ''
        for k in range(1, len(msgs) + 1):
            text = render(msgs[:k], tools)
            if not text.startswith(prev):
                raise SystemExit(f'template is not prefix-stable at message {k} of {rec["instance_id"]}')
            new = text[len(prev):]
            if msgs[k - 1]['role'] == 'assistant' and '<channel|>' in new:
                # Mask the turn header and the synthetic thought; train on the call that follows.
                head, call = new.rsplit('<channel|>', 1)
                masked = tok(head + '<channel|>', add_special_tokens=False)['input_ids']
                trained = tok(call, add_special_tokens=False)['input_ids']
                ids += masked + trained
                labels += [-100] * len(masked) + trained
            else:
                piece = tok(new, add_special_tokens=False)['input_ids']
                ids += piece
                labels += piece if msgs[k - 1]['role'] == 'assistant' else [-100] * len(piece)
            prev = text
        if i == 0:
            print('=' * 30, 'RENDERED EXAMPLE (first 6000 chars)')
            print(prev[:1500]); print('...'); print(prev[-2500:])
            print('=' * 30, 'TRAINED SPANS (first 3)')
            spans, cur = [], []
            for t, l in zip(ids, labels):
                if l != -100:
                    cur.append(t)
                elif cur:
                    spans.append(cur); cur = []
            for s in spans[:3]:
                print(repr(tok.decode(s)))
        lengths.append(len(ids))
        if len(ids) > MAX_TOKENS:
            too_long += 1
            continue
        ftrain.write(json.dumps({'instance_id': rec['instance_id'], 'input_ids': ids, 'labels': labels}) + '\n')
        kept += 1
lengths.sort()
print(json.dumps({'kept': kept, 'held_out_rich': held, 'too_long': too_long,
                  'tokens_median': lengths[len(lengths) // 2] if lengths else 0,
                  'tokens_max': lengths[-1] if lengths else 0,
                  'tokens_total': sum(lengths)}))
