"""Rank the source files and definitions most likely to need the fix.

Usage: locate.py --query "<issue text or key terms>" [--top 8]

Pulls identifiers, dotted names, file paths and quoted strings out of the
query, searches the repository's non-test Python files for them, and prints the
best-scoring files with the matching definitions and lines.
"""
import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STOP = set('''
a an and are as at be but by can could do does for from has have how i if in into is it its
may not of on or should so than that the then there these this to use used using was we when
where which while will with would you your error issue bug fix expected actual behavior
version python true false none self return raise class def import value values type types
default example code test tests case get set new also like just only one same file line
'''.split())
SKIP_DIRS = {'.git', 'tests', 'test', 'testing', 'docs', 'build', 'dist',
             '__pycache__', '.venv', 'venv', 'node_modules', 'scripts', 'benchmarks', 'examples'}
DEF_RE = re.compile(r'^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)')


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def terms_from(query: str) -> Counter:
    terms: Counter = Counter()
    for q in re.findall(r'["\'`]([^"\'`\n]{3,80})["\'`]', query):
        terms[q.strip()] += 3  # quoted strings: error messages, literals
    for dotted in re.findall(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b', query):
        for part in dotted.split('.'):
            if len(part) > 2 and part.lower() not in STOP:
                terms[part] += 2
        terms[dotted] += 2
    for path in re.findall(r'[\w/]+\.py\b', query):
        terms[Path(path).stem] += 3
    for word in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b', query):
        if word.lower() in STOP:
            continue
        code_like = '_' in word or any(c.isupper() for c in word[1:]) or word[0].isupper()
        terms[word] += 2 if code_like else 1
    return terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required=True)
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()

    root = workspace()
    terms = terms_from(a.query)
    if not terms:
        print('No searchable terms in query.')
        return
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
        for name in filenames:
            if name.endswith('.py') and not name.startswith(('test_', 'conftest', '.')):
                files.append(Path(dirpath) / name)

    doc_freq: Counter = Counter()
    contents = {}
    for f in files:
        try:
            text = f.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        contents[f] = text
        for t in terms:
            if t in text:
                doc_freq[t] += 1
    n = max(1, len(contents))

    scores = {}
    hits = defaultdict(list)
    for f, text in contents.items():
        score = 0.0
        rel = str(f.relative_to(root))
        lines = text.splitlines()
        defs = {m.group(2): i for i, line in enumerate(lines) if (m := DEF_RE.match(line))}
        for t, weight in terms.items():
            if t not in text:
                continue
            idf = math.log(1 + n / doc_freq[t])
            score += weight * idf * min(3, text.count(t)) ** 0.5
            if t in defs:  # the term is defined here
                score += 3 * weight * idf
                hits[f].append((defs[t] + 1, lines[defs[t]].strip()))
            if t.lower() == f.stem.lower() or t in rel.replace('/', '.'):
                score += 2 * weight * idf
        if score:
            # docs examples are sometimes the fix site, but rarely the first place to look
            scores[f] = score * (0.6 if rel.startswith('docs_src/') else 1.0)

    ranked = sorted(scores, key=scores.get, reverse=True)[: a.top]
    if not ranked:
        print('No matches. Try different terms (symbol names, error text).')
        return
    for f in ranked:
        print(f'{f.relative_to(root)}  (score {scores[f]:.1f})')
        for lineno, line in sorted(set(hits[f]))[:6]:
            print(f'    {lineno}: {line[:120]}')
    print('\nRead the top files with read_file before editing; scores are lexical, not certain.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # never crash the agent's turn
        print(f'locate.py failed: {exc}', file=sys.stderr)
