"""Print a per-task digest of finished eval kernels into a CPU kernel's log.

    python scripts/digest_runs.py baseline v1 v2   # pushes janholoubek/gemma-eval-digest
    kaggle kernels logs <user>/gemma-eval-digest

Kernel output files are served from www.kaggleusercontent.com, which our
cloud environment may not reach; kernel logs come through the API. This
pushes a CPU-only kernel that mounts the eval kernels' outputs and prints
results, step timelines and patches to stdout. CPU kernels use no GPU quota.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = r'''
import glob, json, os
from pathlib import Path
RUNS = __RUNS__
for run in RUNS:
    hits = glob.glob(f"/kaggle/input/**/gemma-eval-{run}/results", recursive=True) + \
           glob.glob(f"/kaggle/input/gemma-eval-{run}/results")
    print(f"##### RUN {run}: {hits[:1]}")
    if not hits:
        print(os.listdir("/kaggle/input")); continue
    res = Path(hits[0])
    for line in (res / "task_results.jsonl").read_text().splitlines():
        r = json.loads(line)
        print("RESULT", run, json.dumps({k: r.get(k) for k in ("instance_id", "resolved", "agent_patch_size", "duration_seconds", "tool_calls", "total_llm_calls", "error")}))
    for tf in sorted((res / "traces").glob("*.json")):
        d = json.load(open(tf))
        tid = tf.stem.replace("trace_", "")
        prev = 0.0
        out = []
        for s in d.get("steps", []):
            ex = s.get("extra", {})
            for tc in s.get("tool_calls") or []:
                t = tc.get("extra", {}).get("elapsed_s", 0) or 0
                args = json.dumps(tc.get("arguments", {}))[:70]
                obs = len(str((s.get("observation") or {}).get("content", "")))
                m = s.get("metrics") or {}
                out.append(f"{t:6.0f}s +{t-prev:4.0f} {tc.get('function_name')} {args} obs={obs} {json.dumps(m)[:80] if m else ''}")
                prev = t
            if not s.get("tool_calls") and s.get("source") == "agent":
                t = ex.get("elapsed_s", 0) or 0
                out.append(f"{t:6.0f}s +{t-prev:4.0f} TEXT {str(s.get('message',''))[:100]!r}")
                prev = t
        print(f"TRACE {run} {tid} steps={len(out)} metrics={json.dumps(d.get('final_metrics'))}")
        for o in out[:40]:
            print("   ", o)
    for pf in sorted((res / "patches").glob("*.patch")):
        txt = pf.read_text()
        print(f"PATCH {run} {pf.stem} bytes={len(txt)}")
        print(txt[:1200])
'''


def main() -> None:
    runs = sys.argv[1:] or ['baseline', 'v1', 'v2']
    user = os.environ.get('KAGGLE_USERNAME', 'janholoubek')
    meta = {
        'id': f'{user}/gemma-eval-digest', 'title': 'gemma-eval-digest', 'code_file': 'digest.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_internet': False,
        'kernel_sources': [f'{user}/gemma-eval-{r}' for r in runs],
        'dataset_sources': [], 'competition_sources': [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'digest.py').write_text(KERNEL.replace('__RUNS__', repr(runs)))
        (Path(tmp) / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
        subprocess.run(['kaggle', 'kernels', 'push', '-p', tmp], check=True)


if __name__ == '__main__':
    main()
