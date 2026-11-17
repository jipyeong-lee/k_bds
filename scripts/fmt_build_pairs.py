#!/usr/bin/env python3
"""fmt_build_pairs.py — 형식 교정 DPO 짝: GRPO 학습 중 답(completions.jsonl)에서 `</think>` 와 `<answer>` 사이 설명을 <think> 안으로 옮긴다.
왜: 85634(deepvision, 6차 시작)는 답의 66% 가 `</think>` 뒤에 설명(중앙 약 500자)을 붙여 형식 0.5 를 받는다(2026-11-13 실측).
    설명을 지우지 않고 <think> 안으로 옮겨, 모델이 배우는 차이를 '설명을 어디에 쓰나' 하나로 줄인다. 답 글자는 그대로다.
사용: python3 scripts/fmt_build_pairs.py <completions.jsonl> <stage2_deepvision.jsonl> <출력 dir> [시작 step 1000] [최소 정답 보상 1.0]
출력: <dir>/pairs.jsonl(loco_dpo.sh 형식, 이미지 경로 IMG/<파일명>) · <dir>/images.txt(원본 이미지 경로, tar 용)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 로 실행하면 스크립트 폴더가 경로에 없다
from fmt_strict import clean, normalize, strict

SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
USER = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.DOTALL)


def fix(t):
    """원래 답 → (고친 답, 원래 답). 둘 다 앞의 '<think>\\n' 없이(생성 프롬프트가 붙인다). 고칠 수 없거나 이미 정상이면 None."""
    g, b = normalize(t), clean(t)
    return (g, b) if g and g != b else None


def main():
    comp, data, out = sys.argv[1:4]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    min_acc = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    img = {r['messages'][0]['content']: r['images'] for r in map(json.loads, open(data))}
    pairs, seen, n = [], set(), {'rows': 0, 'fmt_ok': 0, 'low_acc': 0, 'unfixable': 0, 'dup': 0, 'no_image': 0}
    for line in open(comp):
        d = json.loads(line)
        for i, s in enumerate(d['step']):
            if int(s) < start:
                continue
            n['rows'] += 1
            if d['FormatThink'][i] >= 1.0: n['fmt_ok'] += 1; continue
            if d['AccuracyMix'][i] < min_acc: n['low_acc'] += 1; continue
            r = fix(d['completion'][i])
            if r is None: n['unfixable'] += 1; continue
            m = USER.search(d['prompt'][i]); q = m and m.group(1)
            if q in seen: n['dup'] += 1; continue   # 문항당 1쌍
            if q not in img: n['no_image'] += 1; continue
            seen.add(q)
            pairs.append({'id': len(pairs), 'step': int(s), 'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q},
                          {'role': 'assistant', 'content': r[0]}], 'rejected_response': r[1], 'images': ['IMG/' + os.path.basename(p) for p in img[q]],
                          '_src': img[q]})
    # 형식 점검: 고친 답은 엄격 형식을 통과하고 원래 답은 통과하지 못한다. <answer> 안 글자는 같다
    for p in pairs:
        g, b = p['messages'][2]['content'], p['rejected_response']
        assert strict(g) and not strict(b), p['id']
        assert g.rsplit('<answer>', 1)[1] == b.rsplit('<answer>', 1)[1].strip(), p['id']
    os.makedirs(out, exist_ok=True)
    with open(f'{out}/pairs.jsonl', 'w') as f:
        for p in pairs:
            f.write(json.dumps({k: v for k, v in p.items() if k != '_src'}, ensure_ascii=False) + '\n')
    with open(f'{out}/images.txt', 'w') as f:
        f.write('\n'.join(sorted({x for p in pairs for x in p['_src']})) + '\n')
    print(n, '→ 쌍', len(pairs))


if __name__ == '__main__':
    main()
