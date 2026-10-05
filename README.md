# Gemma 4 Developer Agent

Our entry for the Kaggle [Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent)
competition (deadline 2 Dec 2026) and its [paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
(deadline 12 Nov 2026).

The task: a coding agent built on `gemma-4-31b-it-qat-w4a16-ct` fixes real issues in
fastapi, rich, requests and httpx. A submission is a declarative Google ADK agent config
(prompts, sub-agents, skills, optional LoRA adapters); the score is the share of hidden
tasks whose tests pass after the agent's patch.

(This repo was first set up for RSNA Knee Abnormality Detection, which we dropped.)

## Layout

- `submission/` — the agent we submit (`agent.yaml` at the root). Starts as the official
  sample, without its placeholder LoRA adapters.
- `scripts/setup_env.sh` — installs the official harness and downloads the competition data.
- `scripts/verify_gold.py` — runs the harness's Phase 2 verification with each task's
  reference patch (or `--empty` for none), no model needed. Use it to find tasks that are
  unsolvable locally before blaming the agent.

## Running locally without Docker

The harness's `subprocess` sandbox differs from the official Docker one in two ways, which
`verify_gold.py` works around:

1. Sandbox venvs inherit the harness's own site-packages, whose httpx/requests shadow the
   repo under test. We point them at a clean `.testbase` env instead.
2. The official image streams each repo's dependencies from a private cache. We install
   them from the competition wheels first, then PyPI.

Checked on one task per repo: httpx, rich and fastapi verify correctly. requests still
fails 6 network-timing tests outside a Docker network sandbox, and `fastapi_14077`'s tests
pass even without a patch.
