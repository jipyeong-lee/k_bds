#!/usr/bin/env python3
"""붕괴 체크포인트의 실제 출력 — 같은 프롬프트로 850 vs 900 대조."""
import json, collections, re, pathlib

#  구 계정 절대경로가 박혀 있었다(k252a02) → 이관 후 파일을 못 연다. __file__ 기준으로 바꾼다.
P = str(pathlib.Path(__file__).parent / 'logs/probe_frag_samples_step%s.jsonl')


def load(step):
    rows = collections.defaultdict(list)
    for l in open(P % step):
        d = json.loads(l)
        rows[d['temp']].append(d)
    return rows


s900, s850 = load(900), load(850)
print('step900 temps:', {k: len(v) for k, v in s900.items()})
print('step850 temps:', {k: len(v) for k, v in s850.items()})

A = s850[sorted(s850)[0]]          # 850 의 가장 낮은 temperature 셀
B = s900[0.0]
tA = sorted(s850)[0]
print(f'\n대조: step850 @ T={tA} vs step900 @ T=0.0   (같은 프롬프트, 인덱스 조인)')

n = min(len(A), len(B))
# 850 은 형식 정상 · 900 은 형식 깨진 짝만
cand = [i for i in range(n) if A[i]['fmt'] == 1.0 and B[i]['fmt'] == 0.0
        and not A[i]['trunc'] and not B[i]['trunc']]
print(f'대조 가능 쌍: {len(cand)}/{n}')

by_src = collections.defaultdict(list)
for i in cand:
    by_src[B[i]['src']].append(i)
print('소스별:', {k: len(v) for k, v in by_src.items()})


def tags(t):
    return {k: t.count(k) for k in ('<think>', '</think>', '<answer>', '</answer>')}


for src in ('mmk12', 'pmcvqa', 'deepvision'):
    idxs = sorted(by_src.get(src, []), key=lambda i: len(B[i]['text']))
    if not idxs:
        continue
    i = idxs[len(idxs) // 2]        # 길이 중앙값 사례
    print('\n' + '=' * 96)
    print(f'### [{src}]  prompt #{i}')
    print('=' * 96)
    for lbl, R in (('step 850', A[i]), ('step 900', B[i])):
        t = R['text']
        print(f'\n--- {lbl}  ({len(t):,}자 · {R["ctok"]} tok · fmt={R["fmt"]:.0f} · {tags(t)}) ---')
        print(t[:900] + ('\n   …(중략)…\n' + t[-320:] if len(t) > 1200 else ''))

# 900 의 전형 — 어떻게 끝나는가
print('\n' + '=' * 96)
print('### step 900 형식 실패 사례의 종결부 (마지막 90자) — 20건')
print('=' * 96)
bad = [d for d in B if d['fmt'] == 0.0 and not d['trunc']]
for d in bad[:20]:
    tail = d['text'][-90:].replace('\n', '⏎')
    print(f'  [{d["src"]:<10} {len(d["text"]):>5}자] …{tail}')

ends = collections.Counter()
for d in bad:
    m = re.search(r'([A-Za-z가-힣 ]{0,12}:?)\s*\S{0,20}$', d['text'].strip())
    ends[d['text'].strip()[-1]] += 1
print('\n종결 문자 분포:', dict(ends.most_common(8)))
print('Answer: 로 끝나는 비율:',
      f"{sum(1 for d in bad if re.search(r'Answer\\s*:', d['text'][-200:])) / max(1,len(bad)):.1%}")
print('</think> 를 포함한 비율:',
      f"{sum(1 for d in bad if '</think>' in d['text']) / max(1,len(bad)):.1%}")
