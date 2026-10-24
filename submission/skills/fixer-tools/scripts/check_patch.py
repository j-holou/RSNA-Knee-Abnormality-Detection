"""Check the current patch before submit_patch.

Usage: check_patch.py [--clean]

Shows the diff against the baseline commit and flags problems the grader would
punish: Python files that no longer compile, edits to test or test-config files
(the grader discards them), and new untracked files (they end up in the patch).
With --clean, deletes new untracked files outside the package source.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROTECTED = ('conftest.py', 'pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg')


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True).stdout


def is_test_path(path: str) -> bool:
    p = Path(path)
    return (p.name.startswith('test_') or p.name.endswith('_test.py')
            or any(part in ('tests', 'test', 'testing') for part in p.parts[:-1])
            or p.name in PROTECTED)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Accept both a bare flag and '--clean true' (dict-style args pass a value).
    ap.add_argument('--clean', nargs='?', const='true', default='false')
    a = ap.parse_args()
    a.clean = str(a.clean).lower() not in ('false', '0', 'no', '')
    root = workspace()

    changed = [l for l in git(root, 'diff', '--name-only', 'HEAD').splitlines() if l]
    untracked = [l for l in git(root, 'ls-files', '--others', '--exclude-standard').splitlines() if l]
    untracked = [u for u in untracked if not u.startswith('.adk_exec_')]

    problems = []
    for path in changed + untracked:
        if path.endswith('.py') and (root / path).exists():
            try:
                compile((root / path).read_bytes(), path, 'exec')
            except SyntaxError as exc:
                problems.append(f'SYNTAX ERROR in {path} line {exc.lineno}: {exc.msg}')
    for path in changed:
        if is_test_path(path):
            problems.append(f'{path} is a test/config file: the grader resets it, so this edit is ignored')
    removed = []
    for path in untracked:
        if a.clean and not is_test_path(path) and (
                '/' not in path or Path(path).name.startswith(('repro', 'reproduce', 'debug', 'scratch', 'tmp'))):
            (root / path).unlink(missing_ok=True)
            removed.append(path)
        else:
            problems.append(f'{path} is a new untracked file and will be part of the patch'
                            + ('' if a.clean else ' (delete it if it is scratch, or rerun with --clean)'))

    source_changes = [p for p in changed if not is_test_path(p)]
    if not source_changes:
        problems.append('NO SOURCE CHANGES: the patch does not modify any library file yet')

    stat = git(root, 'diff', '--stat', 'HEAD').strip()
    print(stat or '(empty diff)')
    diff = git(root, 'diff', 'HEAD', '--', *source_changes) if source_changes else ''
    if diff:
        print(diff[:3500] + ('\n... (diff truncated)' if len(diff) > 3500 else ''))
    for path in removed:
        print(f'removed scratch file {path}')
    print('\n' + ('\n'.join(f'- {p}' for p in problems) if problems else 'OK: ready for submit_patch'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'check_patch.py failed: {exc}', file=sys.stderr)
