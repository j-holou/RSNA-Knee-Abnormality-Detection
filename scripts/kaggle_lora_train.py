"""Kaggle GPU kernel (4x L4): QLoRA-train an adapter on the rendered oracle trajectories.

Pushed by scripts/push_lora.py --stage train. Loads the bf16 QAT twin of the scored model
(gemma-4-31b-it-qat-q4_0-unquantized) in 4-bit NF4 spread over the four GPUs, adds a
rank-16 LoRA on the language model's attention and MLP projections, and trains on
/kaggle/input/**/lora_data/train.jsonl (input_ids + labels; -100 = not trained).
The loss is computed only at trained positions, so the 262k-vocabulary logits are never
materialized for the whole sequence. Writes /kaggle/working/adapters/edit_early/
(adapter_config.json + adapter_model.safetensors) after every epoch.
"""
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U',
                'transformers', 'peft', 'bitsandbytes', 'accelerate'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

RANK, ALPHA, LR, EPOCHS, ACCUM = 16, 32, 1e-4, 2, 4
TARGETS = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
TIME_LIMIT_S = 10 * 3600  # stop and save before Kaggle's 12-hour session limit
OUT = Path('/kaggle/working/adapters/edit_early')
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


data_path = Path(sorted(glob.glob('/kaggle/input/**/lora_data/train.jsonl', recursive=True))[0])
model_dir = Path(sorted(glob.glob('/kaggle/input/**/gemma-4-31b-it-qat-q4_0-unquantized/**/config.json',
                                  recursive=True))[0]).parent
rows = [json.loads(l) for l in open(data_path)]
log(f'{len(rows)} examples, {sum(len(r["input_ids"]) for r in rows)} tokens; model {model_dir}; '
    f'transformers {transformers.__version__}; gpus {torch.cuda.device_count()}')

bnb = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
n_gpu = torch.cuda.device_count()
max_memory = {i: '17GiB' for i in range(n_gpu)}  # leave room for activations
load = dict(quantization_config=bnb, device_map='auto', max_memory=max_memory, torch_dtype=torch.bfloat16)
try:  # the full multimodal class keeps weight names as model.language_model.*, which vLLM expects
    model = transformers.AutoModelForImageTextToText.from_pretrained(model_dir, **load)
except Exception as e:
    log(f'AutoModelForImageTextToText failed ({e}); using AutoModelForCausalLM')
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, **load)
log(f'loaded {type(model).__name__}; memory {[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]} GiB')

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
                                         target_modules=TARGETS, task_type='CAUSAL_LM'))
model.print_trainable_parameters()
base = model.get_base_model()
inner = base.model  # transformer stack; base.lm_head projects to the vocabulary
lm_head = base.get_output_embeddings()
text_cfg = getattr(base.config, 'text_config', base.config)
softcap = getattr(text_cfg, 'final_logit_softcapping', None)
log(f'final_logit_softcapping={softcap}')

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
total_steps = math.ceil(len(rows) * EPOCHS / ACCUM)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 5) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))


def loss_for(row):
    dev = lm_head.weight.device
    ids = torch.tensor([row['input_ids']], device=base.get_input_embeddings().weight.device)
    labels = torch.tensor(row['labels'])
    hidden = inner(input_ids=ids).last_hidden_state[0]
    # Position t predicts token t+1: keep positions whose next token is trained.
    target = labels[1:]
    pos = (target != -100).nonzero(as_tuple=True)[0]
    h = hidden[pos.to(hidden.device)].to(dev)
    logits = lm_head(h).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return torch.nn.functional.cross_entropy(logits, target[pos].to(dev)), len(pos)


model.train()
step, stop = 0, False
for epoch in range(EPOCHS):
    random.Random(epoch).shuffle(rows)
    run_loss, run_n = 0.0, 0
    for i, row in enumerate(rows):
        loss, n = loss_for(row)
        (loss / ACCUM).backward()
        run_loss += loss.item(); run_n += 1
        if (i + 1) % ACCUM == 0 or i == len(rows) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            log(f'epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} '
                f'peak {[round(torch.cuda.max_memory_allocated(g) / 2**30, 1) for g in range(n_gpu)]} GiB')
            run_loss, run_n = 0.0, 0
        if time.time() - T0 > TIME_LIMIT_S:
            log('time limit reached; saving and stopping')
            stop = True
            break
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    log(f'saved adapter after epoch {epoch}: {sorted(p.name for p in OUT.iterdir())}')
    if stop:
        break

cfg = json.loads((OUT / 'adapter_config.json').read_text())
log(f"adapter_config: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} targets={cfg.get('target_modules')}")
from safetensors import safe_open  # noqa: E402
with safe_open(str(OUT / 'adapter_model.safetensors'), 'pt') as f:
    keys = list(f.keys())
log(f'{len(keys)} tensors, e.g. {keys[:2]}; size {(OUT / "adapter_model.safetensors").stat().st_size / 2**20:.0f} MiB')
