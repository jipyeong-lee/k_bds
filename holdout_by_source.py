#!/usr/bin/env python3
"""홀드아웃 소스별 성능 추이 — 절대값 / init 대비 / 인접 구간 / 소스×층 교차."""
import glob, sys
from collections import defaultdict
sys.path.insert(0, 'scripts')
from eval_paired import load, mcnemar

ORDER = ['init', 'step400', 'step500', 'step600', 'step700', 'step850', 'step900', 'step1000']
D = {}
for t in ORDER:
    g = sorted(glob.glob(f'logs/eval_items_{t}_*.jsonl'))
    if g:
        D[t] = load(g[-1])
seq = [t for t in ORDER if t in D]

SRC = sorted({r['source'] for r in D['init'].values()})
IDS = {s: [i for i, r in D['init'].items() if r['source'] == s] for s in SRC}

print('소스 구성:', {s: len(v) for s, v in IDS.items()})

# 소스 × 층 교차 — 어떤 층이 어느 소스에서 오는지
cross = defaultdict(lambda: defaultdict(int))
for r in D['init'].values():
    cross[r['source']][r['stratum']] += 1
print('\n소스 × 층 교차')
for s in SRC:
    print(f'  {s:<11}', dict(sorted(cross[s].items(), key=lambda x: -x[1])))


def acc(t, ids):
    return sum(D[t][i]['score'] for i in ids) / len(ids) * 100


def star(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < .05 else ' ⁿˢ'))


print('\n' + '=' * 96)
print('A) 절대 정확도 (%)')
print('=' * 96)
print(f"{'source':<12}" + ''.join(f'{t.replace("step",""):>10}' for t in seq))
for s in SRC:
    print(f'{s:<12}' + ''.join(f'{acc(t, IDS[s]):>10.2f}' for t in seq))
print(f"{'전체':<11}" + ''.join(f'{acc(t, list(D["init"])):>10.2f}' for t in seq))

print('\n' + '=' * 96)
print('B) init 대비 (짝지음, pp)')
print('=' * 96)
print(f"{'source':<12}" + ''.join(f'{t.replace("step",""):>14}' for t in seq[1:]))
for s in SRC:
    row = f'{s:<12}'
    for t in seq[1:]:
        d, b, c, p, mde = mcnemar([(D['init'][i]['score'], D[t][i]['score']) for i in IDS[s]])
        row += f'{d:>+9.2f}{star(p):<5}'
    print(row)

print('\n' + '=' * 96)
print('C) 인접 구간 (짝지음, pp) — 각 소스 안에서')
print('=' * 96)
print(f"{'구간':<20}" + ''.join(f'{s:>18}' for s in SRC))
for a, b_ in zip(seq, seq[1:]):
    row = f'{a.replace("step","")}→{b_.replace("step",""):<14}'
    for s in SRC:
        d, bb, cc, p, mde = mcnemar([(D[a][i]['score'], D[b_][i]['score']) for i in IDS[s]])
        row += f'{d:>+11.2f}{star(p):<7}'
    print(row)

print('\n' + '=' * 96)
print('D) 최고점 대비 — 400→850 / 700→850 / 850→900 (소스별)')
print('=' * 96)
for a, b_ in [('step400', 'step850'), ('step700', 'step850'), ('step850', 'step900')]:
    print(f'  {a} → {b_}')
    for s in SRC:
        d, bb, cc, p, mde = mcnemar([(D[a][i]['score'], D[b_][i]['score']) for i in IDS[s]])
        print(f'      [{s:<11}] {d:>+7.2f}pp  b={bb:<4} c={cc:<4} p={p:<10.3g}{star(p)}  MDE={mde:.2f}pp')

print('\n' + '=' * 96)
print('E) 소스 안 층별 절대 정확도 (%)')
print('=' * 96)
for s in SRC:
    strata = sorted(cross[s], key=lambda k: -cross[s][k])
    for st in strata:
        ids = [i for i in IDS[s] if D['init'][i]['stratum'] == st]
        if len(ids) < 20:
            continue
        print(f'  {s:<11} [{st:<9} n={len(ids):<5}]' + ''.join(f'{acc(t, ids):>9.2f}' for t in seq))
