"""Read helper the agent runs through run_command (installed as .git/rd.py).

    cd /workspace && python3 .git/rd.py path/to/file.py 120 180

Prints the numbered lines START..END (at most 80; the whole file if it has no more than 80).
Each call counts toward the task's reading budget, shared with locate.py and lookup.py and
stored in .git/reads. Once the budget in .git/read_cap is used up it refuses and says to edit.
"""
import sys
from pathlib import Path

MAX_LINES = 80
GIT = Path(__file__).resolve().parent


def reading_call() -> bool:
    """Count one reading call; False once the budget is used up."""
    try:
        cap = int((GIT / 'read_cap').read_text().strip())
    except (OSError, ValueError):
        cap = 0
    try:
        used = int((GIT / 'reads').read_text().strip())
    except (OSError, ValueError):
        used = 0
    if cap and used >= cap:
        print(f'READING BUDGET USED ({used} of {cap} reading calls). Do not read more code. Edit now with '
              f'.git/sr.py using the lines you have already seen, then run check_patch.py and call submit_patch.')
        return False
    (GIT / 'reads').write_text(str(used + 1))
    if cap:
        print(f'[reading call {used + 1} of {cap}]')
    return True


def main() -> None:
    if len(sys.argv) not in (2, 3, 4):
        print('usage: python3 .git/rd.py path/to/file.py [START [END]]')
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f'{path}: no such file')
        sys.exit(1)
    if not reading_call():
        sys.exit(1)
    lines = path.read_text(encoding='utf-8', errors='replace').split('\n')
    start = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else start + MAX_LINES - 1
    end = min(len(lines), end, start + MAX_LINES - 1)
    print(f'{path} lines {start}-{end} of {len(lines)}:')
    print('\n'.join(f'{i:5} {lines[i - 1]}' for i in range(start, end + 1)))


if __name__ == '__main__':
    main()
