"""Rank the functions most likely to need the fix, and the files that hold them.

Usage: locate.py --query "<issue text or key terms>" [--top 6]

Splits every non-test Python file into definitions (functions, methods, class headers)
and scores each with BM25 over identifier sub-words from the query (snake_case and
CamelCase are split), plus bonuses for exact identifiers, dotted names, quoted strings
and file names the query mentions. Prints the best files, each with its best
definitions and their line ranges, ready for reading.
"""
import argparse
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

STOP = set('''
a an and are as at be but by can could do does for from has have how i if in into is it its
may not of on or should so than that the then there these this to use used using was we when
where which while will with would you your error issue bug fix fixes fixed expected actual
behavior behaviour version python true false none self return raise class def import value
values type types default example code test tests case get set new also like just only one
same file line add adds added support make makes work works now see https http www com github
issue issues pull pr thanks please cls args kwargs str int bool list dict any optional
'''.split())
SKIP_DIRS = {'.git', 'tests', 'test', 'testing', 'docs', 'build', 'dist', '__pycache__', '.venv',
             'venv', 'node_modules', 'scripts', 'benchmarks', 'examples', 'site-packages'}
DEF_RE = re.compile(r'^(\s*)(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)')
WORD_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def workspace() -> Path:
    for cand in (Path('/workspace'), Path(os.environ.get('PWD', '.'))):
        if (cand / '.git').exists():
            return cand
    return Path(os.environ.get('PWD', '.'))


def stem(w: str) -> str:
    for suf in ('ing', 'ed', 'es', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def subwords(ident: str):
    out = []
    for part in ident.split('_'):
        out += re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+', part)
    return [stem(w.lower()) for w in out if len(w) > 1]


def clean_query(query: str) -> str:
    """Drop URLs and PR-template boilerplate (checklists, instructions)."""
    query = re.sub(r'https?://\S+', ' ', query)
    keep = []
    for line in query.splitlines():
        if re.match(r'\s*[-*] \[[ xX]\]', line) or 'Please describe your changes' in line:
            continue
        if re.match(r'\s*#+\s*(Type of changes|Checklist|Description)\s*$', line):
            continue
        keep.append(line)
    return '\n'.join(keep)


def query_terms(query: str):
    query = clean_query(query)
    words = Counter()
    idents = Counter()
    for ident in WORD_RE.findall(query):
        code_like = '_' in ident or any(c.isupper() for c in ident[1:])
        if code_like and len(ident) > 3:
            idents[ident] += 1
        for w in subwords(ident):
            if w not in STOP and len(w) > 2:
                words[w] += 2 if code_like else 1
    plain = {stem(w.lower()) for w in WORD_RE.findall(query) if len(w) > 3 and w.lower() not in STOP}
    quoted = [q.strip() for q in re.findall(r'["\'`]([^"\'`\n]{3,80})["\'`]', query)]
    paths = [Path(p).stem for p in re.findall(r'[\w/]+\.py\b', query)]
    return words, idents, quoted, paths, plain


def chunks(text):
    """[(start, end, name, kind)] 1-based inclusive; a class keeps only its header lines."""
    lines = text.split('\n')
    defs = []
    for i, l in enumerate(lines):
        m = DEF_RE.match(l)
        if m:
            defs.append((i, len(m.group(1).expandtabs()), m.group(3), m.group(2)))
    out = []
    for k, (i, ind, name, kind) in enumerate(defs):
        end = len(lines)
        for j, ind2, _, _ in defs[k + 1:]:
            if ind2 <= ind or kind == 'class':
                end = j
                break
        out.append((i + 1, end, name, kind))
    if not defs or defs[0][0] > 0:
        out.append((1, defs[0][0] if defs else len(lines), '<module>', 'module'))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', required=True)
    ap.add_argument('--top', type=int, default=6)
    a = ap.parse_args()
    root = workspace()
    words, idents, quoted, paths, plain = query_terms(a.query)
    if not words and not quoted:
        print('No searchable terms in query.')
        return
    docs = []  # (file, start, end, name, Counter of sub-words, raw text)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
        for fn in filenames:
            if not fn.endswith('.py') or fn.startswith(('test_', 'conftest', '.')):
                continue
            f = Path(dirpath) / fn
            try:
                text = f.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            lines = text.split('\n')
            for s, e, name, kind in chunks(text):
                body = '\n'.join(lines[s - 1:e])
                tf = Counter(w for ident in WORD_RE.findall(body) for w in subwords(ident))
                docs.append((f, s, e, name, tf, body))
    if not docs:
        print('No Python files found.')
        return
    n = len(docs)
    df = Counter()
    for d in docs:
        for w in words:
            if d[4][w]:
                df[w] += 1
    avg = sum(sum(d[4].values()) for d in docs) / n
    scored = []
    for f, s, e, name, tf, body in docs:
        length = sum(tf.values()) or 1
        sc = 0.0
        for w, qw in words.items():
            c = tf[w]
            if not c:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            sc += qw * idf * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * length / avg))
        for ident in idents:
            if ident == name:
                sc += 6
            elif ident in body:
                sc += 1.5
        if stem(name.lower()) in plain and not any(ident == name for ident in idents):
            sc += 4
        for q in quoted:
            if q in body:
                sc += 4
        rel = str(f.relative_to(root))
        if any(p.lower() == f.stem.lower() for p in paths):
            sc += 4
        elif stem(f.stem.lower().strip('_')) in plain:
            sc += 2
        if rel.startswith('docs_src/'):
            sc *= 0.6
        if sc > 0:
            scored.append((sc, rel, s, e, name))
    scored.sort(reverse=True)
    by_file = defaultdict(list)
    for sc, rel, s, e, name in scored:
        by_file[rel].append((sc, s, e, name))
    # file score: best definition plus a share of the next ones
    fscore = {rel: v[0][0] + 0.3 * sum(x[0] for x in v[1:4]) for rel, v in by_file.items()}
    ranked = sorted(fscore, key=fscore.get, reverse=True)[: a.top]
    if not ranked:
        print('No matches. Try different terms (symbol names, error text).')
        return
    for rel in ranked:
        print(f'{rel}  (score {fscore[rel]:.1f})')
        for sc, s, e, name in sorted(by_file[rel][:4], key=lambda x: x[1]):
            print(f'    {s}: {name}  (lines {s}-{e})')
    print('\nRead the relevant lines with `python3 .git/rd.py FILE START END` before editing; scores are lexical, not certain.')


if __name__ == '__main__':
    main()
