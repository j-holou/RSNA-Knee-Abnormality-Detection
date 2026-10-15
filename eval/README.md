# Evaluation task lists

- `sound.txt`: the 114 public tasks where an empty patch fails and the reference patch
  passes in a Kaggle-like environment, from
  [dmitriigluzdov/gemma-4-task-soundness-and-local-results](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-task-soundness-and-local-results) (CC BY 4.0).
- `unsolvable.txt`: tasks whose tests import names that exist only in the reference fix and
  are never mentioned in the issue (forum thread 745855). No agent can be expected to pass them.
- `dev24.txt`: our fixed development set, 24 tasks drawn from `sound.txt` minus
  `unsolvable.txt` with seed 0, in proportion to the public repo mix
  (12 fastapi, 9 rich, 2 requests, 1 httpx).
- `dev48.txt`: `dev24.txt` plus 24 more tasks from the same pool (seed 1; 12 fastapi,
  10 rich, 2 requests, since the pool has no httpx left). One task is ~2 points instead of ~4.
