#!/usr/bin/env python3
"""전량 홀드아웃 8지점의 짝지음 행렬 — init 대비 / 인접 구간 / 핵심 쌍."""
import glob, json, math, os, sys
sys.path.insert(0, 'scripts')
from eval_paired import load, mcnemar

ORDER = ['init', 'step400', 'step500', 'step600', 'step700', 'step850', 'step900', 'step1000']
files = {}
for t in ORDER:
    g = sorted(glob.glob(f'logs/eval_items_{t}_*.jsonl'))
    if g:
        files[t] = g[-1]
D = {t: load(p) for t, p in files.items()}
print('로드:', {t: len(v) for t, v in D.items()})


def pair(a, b, by=None):
    A, B = D[a], D[b]
    ids = [i for i in A if i in B]
    if by is None:
        return mcnemar([(A[i]['score'], B[i]['score']) for i in ids]), len(ids)
    out = {}
    for i in ids:
        out.setdefault(A[i].get(by, 'all'), []).append((A[i]['score'], B[i]['score']))
    return {k: mcnemar(v) for k, v in sorted(out.items())}, len(ids)


def fmt(r):
    delta, b, c, p, mde = r
    star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    return f'{delta:+6.2f}pp  b={b:<4} c={c:<4} p={p:<9.3g}{star:<3} MDE={mde:.2f}pp'


print('\n' + '=' * 92)
print('A) init 대비 (전량 1,772 짝지음)')
print('=' * 92)
for t in ORDER[1:]:
    if t in D:
        r, n = pair('init', t)
        print(f'  init → {t:<9} n={n:<6} {fmt(r)}')

print('\n' + '=' * 92)
print('B) 인접 구간')
print('=' * 92)
seq = [t for t in ORDER if t in D]
for a, b in zip(seq, seq[1:]):
    r, n = pair(a, b)
    print(f'  {a:<9} → {b:<9} n={n:<6} {fmt(r)}')

print('\n' + '=' * 92)
print('C) 핵심 쌍')
print('=' * 92)
for a, b in [('step400', 'step850'), ('step700', 'step850'), ('step600', 'step850'),
             ('step850', 'step900'), ('step850', 'step1000'), ('step900', 'step1000'),
             ('init', 'step900'), ('init', 'step1000')]:
    if a in D and b in D:
        r, n = pair(a, b)
        print(f'  {a:<9} → {b:<9} n={n:<6} {fmt(r)}')

print('\n' + '=' * 92)
print('D) 소스별 — init 대비')
print('=' * 92)
for t in ORDER[1:]:
    if t not in D:
        continue
    d, n = pair('init', t, by='source')
    print(f'  init → {t}')
    for k, r in d.items():
        print(f'      [{k:<10}] {fmt(r)}')

print('\n' + '=' * 92)
print('E) 층(stratum)별 — init 대비')
print('=' * 92)
for t in ['step850', 'step900', 'step1000']:
    if t not in D:
        continue
    d, n = pair('init', t, by='stratum')
    print(f'  init → {t}')
    for k, r in d.items():
        print(f'      [{k:<10}] {fmt(r)}')
