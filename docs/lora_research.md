# LoRA for Gemma 4 Developer Agent: research notes (2026-10-16)

Labels: **[F]** = fact, with a source. **[G]** = my guess or inference.
Sources: `nb:` = Kaggle notebook (source pulled, outputs stripped, so only the authors' own text is reported). `d/NNN` = discussion
kaggle.com/competitions/gemma-4-developer-agent/discussion/NNN (read through the Kaggle web API). `HR` = data/HARNESS_README.md.

## 0. Bottom line
- [F] No public notebook or post shows a trained LoRA that improves the score. The one measured attempt was a NO-GO: 2/32 runs resolved vs 4/32 for the base model (Dmitrii Gluzdov kit).
- [F] Loading adapters on the scorer has been unreliable. It was fixed several times between Sep 25 and Oct 15, and several LoRA submissions still errored or timed out in mid-October (d/749191, d/749501, d/745364).
- [F] Organizer, Oct 15: "We have seen successful submissions with LoRA adapters up to rank 64." Copy the Getting Started notebook config exactly (d/749191).
- [G] A LoRA is a high-variance bet. Prompt and budget changes probably cap out around 0.18-0.20 (d/748051), and the leaderboard noise is about ±0.08 (nb: hitarthjain0). Spend the first few GPU hours checking that loading works before training anything.

## 1. Public notebooks, datasets and threads on LoRA
| Source | What it does | Reported result |
|---|---|---|
| nb: sauravbellam/tpu-t1-smoke-test… (TpuV5E8, COMPLETE) | Serves `gemma-4-31b-it-qat-q4_0-unquantized` (bf16 QAT twin, model `google/gemma-4/Transformers/…/2`, ~62 GB) with `vllm-tpu==0.31.0`, TP=8, the harness's gemma4 parsers and `--reasoning-config`. Runs the agent on 4 tasks to check that TPU can generate training trajectories | No numbers visible. Notes: the first start compiles XLA for "tens of minutes"; tpu-inference rejects per-request `seed`; the default max_num_batched_tokens of 512 needs raising to 4096 |
| nb: sauravbellam/tpu-t2-more-lora-training-data… (cancelled) | Rejection sampling on TPU: B2 agent × 3 seeds on 56 "rest" tasks, at most 2 resolved trajectories per task. `build_rft_data.py` converts ATIF traces to chat `messages` with `reasoning_content` and `tool_calls` (arguments as a dict, because the Gemma 4 template raises on string args) and puts loss on assistant turns only | No numbers. Its own estimate: about 3×56×434 s / 4 concurrency per pass, in sessions of up to 8 h. Training itself (`train_lora.py`) runs elsewhere |
| nb: dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation + dataset `gemma-4-serving-and-lora-verification-kit` | Audits the 129 tasks (114 "sound": empty patch fails, gold passes). Checks serving fidelity. Runs a short LoRA diagnostic on one A100 (TP1) | **LoRA: r16, α32, lr 1e-4, 412 turns from 23 dev tasks, max_len 8192, 30 steps × GA 8, 3,891 s.** Rich action NLL fell 40.7% and overall NLL ~25-27% on both resolved and unresolved traces ("broad sharpening"). Mini-eval on 16 Rich tasks × 2 seeds: **base 4/32, adapter 2/32 → NO-GO**. Tool-call validity and latency were unchanged. A random rank-4 adapter over 410 projections did change logits, so serving applies adapters |
| same | Agent configurations (no LoRA) | Four-tool single coder: 27/70 dev and 12/44 Rich. Coder-analyzer: 16/70 and 12/44. The sample adapters are zero-delta (`lora_B` = 0, layer 0, q/o only) |
| nb: kevindiazcordova/gemma-4-validated-oracle-trajectories | Synthetic SFT data: an "oracle agent" replays the gold patch through the real harness tools (graph search → read_file → edit_file per hunk → import check → submit_patch) inside the official Evaluator, then verifies with the official Phase 2 | Gold resolves 105/129 after installing missing test dependencies (`inline-snapshot`, `dirty-equals`). Oracle trajectories resolve **103/105**. CPU only, about 1 h. Notes that the Gemma 4 chat template drops past assistant reasoning, so only tool calls become targets |
| nb: denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study, …/gemma-4-offline-qlora-train-verify-submit | QLoRA on **2×T4** with `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (19.1 GB, a non-QAT base), `device_map="balanced"`, r8, seq 1024, 1 epoch. Target is a single-shot unified diff, with Rich held out | Only loss is reported, and it is not visible. The author says it is "only a transfer candidate" because of the base mismatch. Shows that 31B 4-bit training fits on 2×T4 at short sequence lengths |
| nb: keykavus/gemma-4-31b-qat-lora-sft-trainer | Writes an adapter_config only | Fake ("Simulating…"). Ignore |
| nb: nursrijan/…kerashub-lora-training-on-tpu, …adapter-bridge/verifier | KerasHub LoRA on TPU v3-8 | Uses the `gemma2_instruct_2b_en` preset, not Gemma 4 31B. Ignore |
| nb: zhukovoleksiy/gemma-4-walkthrough-findings-submission | Agent notebook | The unmodified sample submission errored. Its adapters are no-ops. The author scored 0.10 without adapters |
| nb: noormamhidayat/sovereign-swe-dual-node-distillation | Prompt "distillation", no LoRA | Self-reported 0.17 |
| d/748051 "top of LB dominated by LoRAs" | Speculation | Replies: "none of the top submissions have managed to get any improvements through finetuning". One author's identical bundle scored 0.17/0.15/0.13/0.15, and LB variance is consistent with binomial noise |
| d/745274 | Distillation from DeepSeek V4.1 Flash | "barely any returns (5 more tasks)" locally |

## 2. Feasibility on free Kaggle hardware
**Which base weights to train against**
- [F] The scorer serves only `gemma-4-31b-it-qat-w4a16-ct` for 31B (HR §3.2).
  - Kaggle also hosts `gemma-4-31b-it-qat-q4_0-unquantized`, the bf16 QAT weights (~62 GB). Saurav calls it the "closest twin of w4a16".
  - Since Oct 11, 12B, E4B and E2B QAT w4a16 variants are also allowed. All agents must use one variant (d/748287).
- [G] Train against the bf16 QAT checkpoint, or that checkpoint quantized on the fly to 4-bit (QLoRA).
  - w4a16 is the int4 rounding of the same QAT weights, so a LoRA delta learned on it should transfer with little error.
  - The Unsloth `gemma-4-31B-it-unsloth-bnb-4bit` checkpoint is a non-QAT base and a larger mismatch. Unsloth has a QAT guide: unsloth.ai/docs/models/gemma-4/qat (cited in d/743186).
- [F] No public evidence yet that a bf16- or QLoRA-trained adapter *helps* on the w4a16 vLLM build. Dmitrii's adapter loaded and changed outputs but did worse.

**Hardware options**
- [F] **2×T4.** Den Pugovkin's recipe fits at seq 1024, r8. Another user could not fit it (device-mismatch errors with `device_map="auto"`, OOM on a single GPU) (d/745364).
- [G] **4×L4 (96 GB) QLoRA.** About 18-20 GB of 4-bit weights split with `device_map="balanced"` leaves ~70 GB for LoRA, optimizer and checkpointed activations. 8-16k-token trajectories look feasible.
  - Speed estimate: Dmitrii's A100 run took 65 min for 240 sequences of ≤8k tokens. Guess L4 is 3-5× slower, so ~3-5 h wall per 240 sequences, which costs 6-10 h of quota (L4×4 counts double).
- [G] **TPU v5e-8 (8×16 GB, separate quota; ~20 h/week per nursrijan).**
  - Saurav showed inference with vllm-tpu only. No public notebook trains Gemma 4 31B LoRA on TPU.
  - bf16 weights are 62 GB, so FSDP over 8 chips needs ~7.8 GB/chip, leaving ~8 GB for activations. It probably works with JAX/Keras-Hub or torch-xla sharding plus remat, but it is new engineering: XLA compiles, and LoRA weights must be renamed to PEFT keys.
  - Best TPU use: generating rejection-sampling data (Saurav's T2), not training.
- [G] **12B QAT.** About 4× cheaper to train, and QLoRA fits on 2×T4. The whole bundle would have to switch to 12B, whose base score is unmeasured. Worth one baseline eval before committing.

**Serving pitfalls**
- [F] The scorer uses Getting Started notebook values (d/749528, d/749191):
  - `max_loras = len(adapters)`, `max_lora_rank = max(detected rank)` (floored to 8 per d/749191), `scheduler_reserve_full_isl=False`
  - env `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, gpu util 0.90 in the notebook (HR says 0.80).
  - HR's "max_loras 8 / rank 128" is outdated.
- [F] KV cache with LoRA:
  - Early on it was 7,600 tokens and requests over ~7.6k hung (d/744331), since fixed.
  - Now about 27.7k tokens with LoRA vs ~36-46k without, because CUDA graphs are captured twice. A single sequential agent fits in 32k.
  - Without the env var, startup OOMs ("warming up sampler with 256 dummy requests", d/746503, d/749191).
  - Decoding is ~11% slower with LoRA (d/749191).
- [F] Failure history:
  - The sample submission failed to score on Sep 25-26 (d/743213).
  - Adapters were silently zeroed by double layer registration; fixed in wheelhouse v23 (d/743508).
  - Through Oct 15, users reported LoRA submissions with "Notebook Threw Exception", "Kaggle Error", or pending >12 h then "exceeded the allowed runtime", while the same bundle without the adapter scored 0.13-0.15 (d/745364, d/749191, d/749501).
  - Tiny rank-4 adapters scored normally (d/749191, Oct 12).
- [F] Tensor names and target modules:
  - Keys look like `base_model.model.model.language_model.layers.N.self_attn.{q,o}_proj.lora_{A,B}.weight`.
  - vLLM merges q/k/v into `qkv_proj`.
  - Confirmed loaded: q/k/v/o/gate/up/down on all 60 text layers (820 tensors = 410 projections; 60×7 = 420, so about 10 layers lack a separate v_proj) (d/749191, Dmitrii kit).
  - Users report that r16 on all modules is 245 MB, and r64 on all modules is 980 MB.
- [G] Restrict `target_modules` to the language model (a regex on `language_model`) so vision or audio towers get no LoRA. Keep rank ≤64, which is what the organizer has seen succeed.
- [F] Rendering fidelity: ADK replays thoughts in `reasoning_content`, while the vLLM template reads `reasoning`. Training renderings must match the served prompt, which Dmitrii's served_history fix matched 803/803 (Dmitrii kit; organizer correction in d/744354).

## 3. Training data options
- [F] **Rules** (organizer/host answers):
  - Only LoRA adapters can be evaluated (d/743383).
  - Training on gemma-4-31b's own outputs is fine (d/743964).
  - Trajectories from other open-weight models, and public HF SWE trajectory datasets, are "Both … permitted" (d/746046, HOST, Oct 5).
  - Proprietary APIs are OK "as long as you are using those models in accordance with their model license terms" (d/742807, ADMIN). The provider terms of OpenAI and Anthropic restrict training competing models, so that path is legally murky, and it is paid anyway.
  - Sending task data to a third-party API (d/746251) is unanswered.
- **Option A: rejection sampling of our own agent** (RFT, Saurav's T2 pipeline). Data is on-policy and in the exact format.
  - We resolve ~11-13 of 48 per eval, mostly on easy tasks, so ~10 new trajectories per eval run.
  - [G] It reinforces what the model already does and does not directly fix "never edits". Dmitrii's version (412 turns, 23 tasks) did not help.
- **Option B: oracle/synthetic trajectories from gold patches** (Kevin Diaz Cordova's method, CPU only, ~103 verified trajectories).
  - [G] Best match for our failure mode: we control the shape (≤6-8 exploration calls, then edit_file, test, submit_patch) and can render them with *our* system prompt and tool list.
  - Risks: thoughts are templated (low-quality reasoning targets; mask them or keep them short); the model may learn to "edit early" without the localization skill; only 4 repos (fastapi-heavy: 67/48/13/1) while hidden tasks come from other repos.
- **Option C: open-weight teacher or public datasets** (allowed). The datasets give repository diversity, which our 4 repos lack.
  - [G] Candidates from memory (not checked here): SWE-Gym, SWE-smith trajectories, nebius SWE-agent-trajectories. They need converting to this harness's 9-tool schema, which is non-trivial.
  - A free teacher needs free GPU time, for example running a larger open model on TPU. That is not free in practice for >100B models.
- **Size** [G]:
  - Behaviour shaping (when to edit or submit) usually needs only ~100-300 good trajectories (~1-4M tokens) and 1-2 epochs at lr 1e-4 to 2e-4, r16.
  - Teaching new localization skill needs thousands.
  - Hold out Rich (as Dmitrii and Den Pugovkin did), or hold out one repo, for every decision.

## 4. Minimal first experiment (free quota)
Goal: find out whether an adapter that makes the agent edit early improves the resolve rate, at about 20 GPU-quota hours.
1. **Loading smoke test** (L4×4, ~1 h wall ≈ 2 quota h).
   - Use the Getting Started serving cell verbatim (with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`).
   - Use a random-B adapter, r16, all 60 language layers, q/k/v/o/gate/up/down, in the PEFT key format above.
   - Check: server starts; the KV-cache line reads ~27k; outputs change against the base; one agent task runs.
   - Then submit our current bundle plus a **zero-B** r16 adapter (one daily slot) to confirm the scorer starts it and scores about our baseline. Stop if it fails.
2. **Data** (CPU, ~1-2 h, 0 GPU).
   - Fork Kevin Diaz Cordova's oracle notebook. Swap in our system prompt and tool set.
   - Cap the oracle at ≤6 exploration calls before the first `edit_file`, then run one targeted test and `submit_patch`.
   - Exclude Rich, which leaves ~60-65 verified trajectories.
   - Add our own resolved non-Rich trajectories from past evals, at most 2 per task.
   - Render with the served chat template (thoughts in the field the server reads). Put loss on assistant tool calls; drop or shorten templated thoughts.
3. **Train** (L4×4 QLoRA, ~3-4 h wall ≈ 6-8 quota h) [G estimate].
   - Base: `gemma-4-31b-it-qat-q4_0-unquantized` in 4-bit NF4, using Unsloth or HF with peft and `device_map="balanced"`.
   - Settings: r16, α32, lr 1e-4, 2 epochs, max_len 16k, grad checkpointing.
   - Save PEFT safetensors, then check the key names and that `lora_B` ≠ 0.
4. **Paired local eval** (L4×4, ≈ 8-10 quota h).
   - Base vs adapter on the same ~20 sound Rich tasks, same seed and our eval_config.
   - Metrics: resolved count, *first-edit call index*, submit rate.
   - Gate: adapter resolved ≥ base and median first-edit call falls. Only then spend a submission.
   - Optional cheaper path: serve the bf16 QAT twin on TPU (Saurav's T1 setup) for evals, if vllm-tpu LoRA works (unverified).
- **Total** ≈ 16-20 L4 quota hours, inside one week's 30 h. TPU quota stays free for Option A data generation.
- **Main risks**:
  - The scorer's LoRA path still fails or times out (d/749191).
  - The adapter sharpens style but not repair skill (Dmitrii's NO-GO).
  - Overfitting to fastapi/rich.
  - A train/serve rendering or base mismatch (bnb vs w4a16).
  - +11% decode time and less KV push long tasks into the time cap.
  - With ~20 eval tasks and ±0.08 LB noise, a real gain under ~3 tasks cannot be detected.
