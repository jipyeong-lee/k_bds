#!/usr/bin/env python3
"""loco_build_pairs.py — 풀이 줄이기 B안: 큰 Qwen 의 생각을 <think>..</think><answer>..</answer> 로 바꿔 좋은 답으로, 원본 모델 답을 나쁜 답으로 짝짓는다.
사용: python3 scripts/loco_build_pairs.py [work/loco] [최소 단어 차이 200] [쌍 한쪽 최대 단어 6500]
입력: work/loco/{rollouts.jsonl(원본 모델), api_answers.jsonl(큰 Qwen)} · 출력: work/loco/pairs.jsonl, 이미지 경로는 IMG/<파일명>(학습 pod 에서 바꿔 쓴다)
좋은 답 = 문항마다 정답이고 끝까지 쓴 큰 Qwen 답 중 가장 짧은 것. 형식은 ta_w07 과 같다: 생각 + "\n</think>\n<answer>정답</answer>" (생성 프롬프트가 앞의 '<think>\n' 을 붙이므로 뺀다).
  kind=short: 원본도 정답 → 나쁜 답은 원본의 정답 중 가장 긴 것(좋은 답보다 MIN_GAP 단어 이상 길 때만)
  kind=fix  : 원본이 모두 오답이거나 잘림 → 나쁜 답은 원본 오답 중 가장 짧은 것(없으면 건너뜀. 잘린 답은 너무 길어 쓰지 않는다)
"""
import json, os, re, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'work/loco'; GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 200; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 6500
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
w = lambda t: len(t.split())
strip = lambda t: t[len('<think>\n'):] if t.startswith('<think>\n') else t
# 큰 Qwen 의 혼잣말: ① 첫머리의 "The user wants me to ..." 과제 되풀이 ② 끝머리의 "I'll write reasoning concisely." 같은 출력 지시 문장
META = re.compile(r"^(the user (wants|asks|is asking)\b.*|.*\b(concise|concisely)\b.*|.*\b(write|provide|present|produce|give|output)\b.*\b(reasoning|final)\b.*|.*<answer>.*|reasoning\b.*|(thus|hence|so|now) final\.?)$", re.I)
dropped = 0
def clean(t):
    global dropped
    out = []
    for line in t.split('\n'):
        keep = []
        for s in re.split(r'(?<=[.!?])\s+', line):
            if META.match(s.strip()): dropped += 1
            else: keep.append(s)
        out.append(' '.join(keep))
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip()
def api_form(a):
    m = re.findall(r'<answer>(.*?)</answer>', a['content'], re.S)
    return clean(a['reasoning']) + '\n</think>\n<answer>' + m[-1].strip() + '</answer>'
base = {r['id']: r for r in map(json.loads, open(f'{D}/rollouts.jsonl'))}
out, n = [], {'short': 0, 'fix': 0, 'no_good': 0, 'no_reject': 0}
for r in map(json.loads, open(f'{D}/api_answers.jsonl')):
    good = [api_form(a) for a in r['answers'] if a['correct'] and a['finish'] == 'stop' and a['reasoning'].strip()]
    good = [g for g in good if w(g) <= CAP]
    if not good: n['no_good'] += 1; continue
    g = min(good, key=w); b = base[r['id']]
    right = [t for t, c, f in zip(b['completions'], b['correct'], b['finish']) if c and f == 'stop' and w(t) <= CAP]
    if right:
        kind, rej = 'short', max(right, key=w)
        if w(rej) - w(g) < GAP: n['no_reject'] += 1; continue
    else:
        wrong = [t for t, c, f in zip(b['completions'], b['correct'], b['finish']) if not c and f == 'stop' and w(t) <= CAP]
        if not wrong: n['no_reject'] += 1; continue
        kind, rej = 'fix', min(wrong, key=w)
    n[kind] += 1
    out.append({'id': r['id'], 'kind': kind, 'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': '<image>' + r['question']}, {'role': 'assistant', 'content': g}],
                'rejected_response': strip(rej), 'images': ['IMG/' + r['image']]})
with open(f'{D}/pairs.jsonl', 'w') as f:
    for x in out: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(n, '→', len(out), '쌍', '· 걸러낸 문장', dropped)
if out:
    import statistics as S
    for k in ('short', 'fix'):
        xs = [x for x in out if x['kind'] == k]
        if xs: print(k, len(xs), '좋은 답 중앙 단어', S.median(w(x['messages'][2]['content']) for x in xs), '나쁜 답 중앙 단어', S.median(w(x['rejected_response']) for x in xs))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 로 실행하면 스크립트 폴더가 경로에 없다
from fmt_strict import strict
assert all(strict(x['messages'][2]['content']) for x in out)   # 좋은 답 100% 엄격 형식(`</think>` 와 `<answer>` 사이 설명 금지)
