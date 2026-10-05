"""Find existing tests that exercise a source file or symbol.

Usage: find_tests.py --target <path/to/module.py | SymbolName> [--top 6]

Prints test files ranked by how often they reference the target, with a ready
pytest command for the best ones.
"""
import argparse
import os
import re
import sys
from pathlib import Path


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--top', type=int, default=6)
    a = ap.parse_args()

    root = workspace()
    target = a.target.strip()
    needles = []
    if target.endswith('.py'):
        mod = target[:-3].replace('/', '.')
        for prefix in ('src.',):
            mod = mod.removeprefix(prefix)
        stem = Path(target).stem
        needles = [mod, f'from {mod.rsplit(".", 1)[0]} import', stem]
    else:
        needles = [target]

    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('build', 'dist')]
        for name in filenames:
            if not (name.startswith('test_') or name.endswith('_test.py')) or not name.endswith('.py'):
                continue
            f = Path(dirpath) / name
            try:
                text = f.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            count = sum(text.count(n) for n in needles)
            name_bonus = 5 if any(n.split('.')[-1].lower() in name.lower() for n in needles) else 0
            if count or name_bonus:
                funcs = [m for m in re.findall(r'^\s*(?:async\s+)?def (test\w+)', text, re.M)
                         if any(n.split('.')[-1] in text.split(f'def {m}', 1)[1][:2000] for n in needles)]
                results.append((count + name_bonus, f.relative_to(root), funcs[:5]))

    if not results:
        print(f'No tests reference {target!r}. Write an inline check in /tmp instead.')
        return
    results.sort(key=lambda r: r[0], reverse=True)
    for score, rel, funcs in results[: a.top]:
        print(f'{rel}  (refs {score})')
        if funcs:
            print('    tests: ' + ', '.join(funcs))
    best = results[0][1]
    print(f'\nRun: python -m pytest {best} -x -q 2>&1 | tail -30')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'find_tests.py failed: {exc}', file=sys.stderr)
