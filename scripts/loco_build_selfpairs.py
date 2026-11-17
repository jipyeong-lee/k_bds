#!/usr/bin/env python3
"""loco_build_selfpairs.py — 풀이 줄이기 3차: 원본 모델 자신의 정답 중 가장 짧은 것(좋은 답)과 가장 긴 것(나쁜 답)을 짝짓는다. 큰 Qwen 은 쓰지 않는다.
사용: python3 -I scripts/loco_build_selfpairs.py [work/loco2] [최소 단어 차이 200] → <dir>/selfpairs.jsonl (이미지 경로 IMG/<파일>)"""
import json, os, statistics as S, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 로 실행하면 스크립트 폴더가 경로에 없다
from fmt_strict import normalize, strict
D = sys.argv[1] if len(sys.argv) > 1 else 'work/loco2'; GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
w = lambda t: len(t.split())


def form(t):  # 앞의 '<think>\n' 을 빼고 마지막 </answer> 뒤를 자른다. 형식이 맞지 않으면 None
    t = t[len('<think>\n'):] if t.startswith('<think>\n') else t
    i = t.rfind('</answer>')
    t = t[:i + len('</answer>')] if i >= 0 else t
    return t if t.count('</think>') == 1 and t.endswith('</answer>') and '<answer>' in t else None


out = []
for r in map(json.loads, open(f'{D}/rollouts.jsonl')):
    c = [x for x in (form(t) for t, k, f in zip(r['completions'], r['correct'], r['finish']) if k and f == 'stop') if x]
    if len(c) < 2: continue
    g, b = min(c, key=w), max(c, key=w)
    if w(b) - w(g) < GAP: continue
    g = normalize(g)   # 좋은 답은 엄격 형식으로: `</think>` 뒤 설명을 <think> 안으로 옮긴다(단어 수는 같다)
    if g is None: continue
    out.append({'id': r['id'], 'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '<image>' + r['question']}, {'role': 'assistant', 'content': g}],
                'rejected_response': b, 'images': ['IMG/' + r['image']]})
with open(f'{D}/selfpairs.jsonl', 'w') as f:
    for x in out: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(out), '쌍 · 좋은 답 중앙', S.median(w(x['messages'][2]['content']) for x in out), '· 나쁜 답 중앙', S.median(w(x['rejected_response']) for x in out))
assert out and all(strict(x['messages'][2]['content']) for x in out)   # 좋은 답 100% 엄격 형식
