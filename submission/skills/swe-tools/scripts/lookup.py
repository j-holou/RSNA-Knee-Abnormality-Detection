"""Answer several code lookups in one call.

Usage: lookup.py --names "Console.print, soft_wrap, 'some error text'" [--file rich/console.py]

For each comma-separated name prints:
- every definition (def / class / NAME = ...) with its body, numbered, up to 40 lines;
- up to 6 places that use it (file:line: code).
A quoted entry is searched as literal text and shown with 2 lines of context.
Test files are skipped. --file restricts the search to one file or directory.
Output stays under ~4,500 characters so nothing is cut off.
"""
import argparse
import os
import re
from pathlib import Path

SKIP_DIRS = {'.git', 'tests', 'test', 'testing', 'build', 'dist', '__pycache__',
             '.venv', 'venv', 'node_modules', 'examples', 'benchmarks'}
BUDGET = 4500
BODY_LINES = 40
USES = 6


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def source_files(root: Path, only: str | None):
    base = root / only if only else root
    if base.is_file():
        yield base
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(filenames):
            if name.endswith('.py') and not name.startswith(('test_', 'conftest', '.')) \
                    and not name.endswith('_test.py'):
                yield Path(dirpath) / name


def block_end(lines: list[str], start: int) -> int:
    """Index after the indented block that starts at lines[start]."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    i = start + 1
    # Skip a multi-line signature: continue until the line ending with ':'.
    while i < len(lines) and not lines[i - 1].rstrip().endswith(':') and i - start < 15:
        i += 1
    while i < len(lines):
        s = lines[i]
        if s.strip() and len(s) - len(s.lstrip()) <= indent and not s.lstrip().startswith((')', ']', '}')):
            break
        i += 1
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', required=True)
    ap.add_argument('--file')
    a = ap.parse_args()
    root = workspace()
    texts = {}
    for f in source_files(root, a.file):
        try:
            texts[f.relative_to(root)] = f.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            pass

    entries = [e.strip() for e in a.names.split(',') if e.strip()]
    per_entry = max(600, BUDGET // max(1, len(entries)))
    body_lines = BODY_LINES if len(entries) <= 2 else 25
    out = []
    for entry in entries:
        chunk = []
        literal = entry[0] in '\'"`' and entry[-1] == entry[0] and len(entry) > 2
        if literal:
            needle = entry[1:-1]
            chunk.append(f'== text {needle!r}')
            hits = 0
            for path, lines in texts.items():
                for i, line in enumerate(lines):
                    if needle in line:
                        hits += 1
                        if hits <= 5:
                            lo, hi = max(0, i - 2), min(len(lines), i + 3)
                            chunk.append(f'{path}:{i + 1}')
                            chunk += [f'{j + 1:>5} {lines[j]}' for j in range(lo, hi)]
            if hits > 5:
                chunk.append(f'... {hits - 5} more matches')
            if not hits:
                chunk.append('not found')
        else:
            name = entry.split('.')[-1]
            owner = entry.split('.')[-2] if '.' in entry else None
            def_re = re.compile(rf'^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*(?::[^=]*)?=[^=].*[^,(]$')
            # Attribute access for Owner.name, else the bare word; prose in comments is skipped.
            use_re = re.compile(rf'\.{re.escape(name)}\b' if owner else rf'\b{re.escape(name)}\b')
            chunk.append(f'== {entry}')
            defs = uses = 0
            use_lines = []
            for path, lines in texts.items():
                for i, line in enumerate(lines):
                    if def_re.search(line):
                        if owner:  # only the definition inside the named class or module
                            if line.startswith((' ', '\t')):
                                cls = next((lines[k] for k in range(i, -1, -1)
                                            if lines[k].startswith('class ')), '')
                                if not re.match(rf'class\s+{re.escape(owner)}\b', cls):
                                    continue
                            elif path.stem != owner:
                                continue
                        defs += 1
                        if defs > 3:
                            continue
                        end = block_end(lines, i) if line.lstrip().startswith(('def', 'class', 'async')) else i + 1
                        shown = min(end, i + body_lines)
                        chunk.append(f'{path}:{i + 1}-{end} (definition)')
                        chunk += [f'{j + 1:>5} {lines[j]}' for j in range(i, shown)]
                        if shown < end:
                            chunk.append(f'      ... {end - shown} more lines (read_file {path} {shown + 1}-{end})')
                    elif use_re.search(line.split('#', 1)[0]) and not line.lstrip().startswith(('"', "'")):
                        uses += 1
                        if len(use_lines) < USES:
                            use_lines.append(f'  {path}:{i + 1}: {line.strip()[:110]}')
            if not defs:
                chunk.append('no definition found')
            if use_lines:  # keep uses near the top so a long body cannot push them out
                chunk[1:1] = [f'used in {uses} places, e.g.:'] + use_lines
        text = '\n'.join(chunk)
        if len(text) > per_entry:
            text = text[:per_entry] + '\n... (cut; narrow with --file or a more specific name)'
        out.append(text)
    print('\n\n'.join(out))


if __name__ == '__main__':
    main()
