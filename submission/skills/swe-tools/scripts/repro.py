"""Run a short reproduction script for the issue, and rerun it after the fix.

Usage: repro.py --code "<python code>"     saves the code to /tmp/repro_issue.py and runs it
       repro.py --rerun true               runs the saved script again (after editing the source)

The code runs from /workspace with the repository importable, under a 60-second limit.
Write it so that it FAILS (assertion or exception) while the bug is present and prints
"REPRO OK" once fixed. Prints the exit status and the last lines of output.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SAVED = Path('/tmp/repro_issue.py')
MAX_CHARS = 3000


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--code')
    ap.add_argument('--rerun', default='')
    argv = sys.argv[1:]  # join '--code', value so code starting with '-' still parses
    if '--code' in argv and argv.index('--code') + 1 < len(argv):
        i = argv.index('--code')
        argv[i:i + 2] = ['--code=' + argv[i + 1]]
    a = ap.parse_args(argv)
    if a.code:
        SAVED.write_text(a.code if a.code.endswith('\n') else a.code + '\n', encoding='utf-8')
    elif not SAVED.exists():
        print('ERROR: no saved script yet; pass --code "<python code>" first.')
        return
    ws = workspace()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(p for p in (str(ws), os.environ.get('PYTHONPATH', '')) if p))
    try:
        r = subprocess.run([sys.executable, str(SAVED)], cwd=ws, env=env, capture_output=True, text=True, timeout=60)
        out, code = (r.stdout + r.stderr).strip(), r.returncode
    except subprocess.TimeoutExpired:
        print('TIMEOUT after 60 s: make the script smaller.')
        return
    if len(out) > MAX_CHARS:
        out = '...\n' + out[-MAX_CHARS:]
    print(out or '(no output)')
    if code == 0 and 'REPRO OK' in out:
        print('\nRESULT: PASS (the expected behavior holds). If you have not edited yet, the script does not '
              'capture the bug: tighten the assertion. If you have, run check_patch.py and submit.')
    elif code == 0:
        print('\nRESULT: FAIL (exit 0 but no "REPRO OK" line): end the script with print("REPRO OK").')
    else:
        print(f'\nRESULT: FAIL (exit {code}). Before the fix this is expected and confirms the bug. '
              'After the fix, adjust the source, not the script.')


if __name__ == '__main__':
    main()
