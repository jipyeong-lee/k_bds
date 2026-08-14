#!/usr/bin/env python3
"""checkpoint-900 greedy 출력의 형식 실패 유형 분류.

§8-b 는 학습 롤아웃(T=0.9, step 875~899)에서 "태그 0개 + `Answer: X` 종결"을
서명으로 기록했다. 체크포인트를 직접 greedy 로 돌리면 같은 모양인지 확인한다.
"""
import json, re, collections, pathlib

#  구 계정 절대경로가 박혀 있었다(k252a02) → 이관 후 파일을 못 연다. __file__ 기준으로 바꾼다.
LOG = pathlib.Path(__file__).parent / 'logs/probe_frag_samples_step900.jsonl'
B = [json.loads(l) for l in open(LOG)]
B = [d for d in B if d['temp'] == 0.0]
bad = [d for d in B if d['fmt'] == 0.0 and not d['trunc']]
good = [d for d in B if d['fmt'] == 1.0 and not d['trunc']]
print(f'전체 {len(B)} · 비절단 형식실패 {len(bad)} · 비절단 정상 {len(good)}')

ANS_RE = re.compile(r'Answer\s*:', re.I)
mode = collections.Counter()
for d in bad:
    t = d['text']
    has_close = '</think>' in t
    has_ans = '<answer>' in t
    if not has_close and not has_ans:
        mode['A. 태그 전무'] += 1
    elif has_close and not has_ans:
        mode['B. </think> 까지 쓰고 멈춤'] += 1
    elif has_ans:
        mode['C. <answer> 는 있으나 구조 위반'] += 1
print('\n실패 유형')
for k, v in mode.most_common():
    print(f'  {k:<28} {v:>4}건  {v/len(bad):>6.1%}')

print('\n종결부 특징')
print(f"  `</think>` 로 끝남            {sum(1 for d in bad if d['text'].rstrip().endswith('</think>')):>4}건"
      f"  {sum(1 for d in bad if d['text'].rstrip().endswith('</think>'))/len(bad):>6.1%}")
print(f"  `Answer:` 가 마지막 200자에 있음 {sum(1 for d in bad if ANS_RE.search(d['text'][-200:])):>4}건"
      f"  {sum(1 for d in bad if ANS_RE.search(d['text'][-200:]))/len(bad):>6.1%}")
print(f"  `\\boxed{{` 포함                 {sum(1 for d in bad if '\\boxed{' in d['text']):>4}건")

# 정상 사례는 어떻게 끝나나 (대조)
print('\n대조 — 비절단 정상(fmt=1) 사례의 종결')
for d in good[:5]:
    print(f"  [{d['src']:<10}] …{d['text'][-60:].replace(chr(10), '⏎')}")

# 유형 B 의 원문 하나 통째로
b_only = [d for d in bad if '</think>' in d['text'] and '<answer>' not in d['text']]
if b_only:
    d = sorted(b_only, key=lambda x: len(x['text']))[0]
    print('\n' + '=' * 90)
    print(f"유형 B 최단 사례 전문 — [{d['src']}] {len(d['text'])}자 · {d['ctok']} tok")
    print('=' * 90)
    print(d['text'])
