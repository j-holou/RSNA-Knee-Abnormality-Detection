"""Replace a range of lines in a file. Use when edit_file cannot find a unique match.

Usage: replace_lines.py --file rich/console.py --start 120 --end 124 --text "<new code>"

Lines start..end (1-based, inclusive) are replaced by --text, written exactly as
given, including indentation. --end start-1 inserts before line start without
removing anything. Prints the region before and after the change and checks that
a Python file still compiles; on a syntax error the file is left unchanged.
"""
import argparse
import os
import sys
from pathlib import Path


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def show(lines, lo, hi):
    return '\n'.join(f'{i + 1:>5} {lines[i]}' for i in range(max(0, lo), min(len(lines), hi)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--text', required=True)
    argv = sys.argv[1:]  # join '--text', value so a value starting with '-' still parses
    if '--text' in argv and argv.index('--text') + 1 < len(argv):
        i = argv.index('--text')
        argv[i:i + 2] = ['--text=' + argv[i + 1]]
    a = ap.parse_args(argv)
    path = Path(a.file)
    if not path.is_absolute():
        path = workspace() / path
    old = path.read_text(encoding='utf-8').split('\n')
    if not (1 <= a.start <= len(old) + 1 and a.start - 1 <= a.end <= len(old)):
        print(f'ERROR: {a.file} has {len(old)} lines; got start={a.start} end={a.end}. Nothing changed.')
        return
    new_lines = a.text.split('\n')
    if new_lines and new_lines[-1] == '':
        new_lines.pop()
    new = old[:a.start - 1] + new_lines + old[a.end:]
    text = '\n'.join(new)
    if path.suffix == '.py':
        try:
            compile(text, str(path), 'exec')
        except SyntaxError as e:
            print(f'ERROR: the change would break syntax ({e.msg} at line {e.lineno}). Nothing changed.')
            print(show(new, (e.lineno or a.start) - 4, (e.lineno or a.start) + 3))
            return
    print('BEFORE:')
    print(show(old, a.start - 3, a.end + 2))
    path.write_text(text, encoding='utf-8')
    print('AFTER:')
    print(show(new, a.start - 3, a.start - 1 + len(new_lines) + 2))
    print(f'OK: {a.file} lines {a.start}-{a.end} replaced by {len(new_lines)} line(s).')


if __name__ == '__main__':
    main()
