#!/usr/bin/env python3
"""
build_rft_coldstart.py — rejection-sampling 콜드스타트 SFT 데이터 생성 (STaR/ReST/RFT)

진단 결과: base Qwen3.5-VL 이 어려운 시각/기하 문제에 본래 매우 길게 추론 →
  4096/6144 budget 으로도 ~50% 잘림(무답=0점). RL/budget 으로 못 잡음.
대응: 모델 "자기 자신"의 정답+마감+간결 완성문을 모아 SFT → 간결 추론 습관을 재주입.
  (지난 콜드스타트는 쉬운 clevr_math 760자라 어려운 문제로 일반화 실패)

소스: GRPO 롤아웃 로그(work/checkpoints/grpo_general/v*/completions.jsonl)
  필터: AccuracyMix==1.0 (정답) AND <think>..</think><answer>..</answer> 마감 AND len<=MAX_CHARS (간결)
  질문당 가장 짧은 것 위주로 최대 K개. 이미지 경로는 DeepVision 원본과 질문텍스트로 join.

출력: work/data/sft_rft_coldstart_{train,val}.jsonl  (ms-swift {messages, images} 포맷)
"""
import json
import glob
import re
import os

PROJ = "/home01/k266a01/kbds_project"
DEEPVISION = f"{PROJ}/work/data/deepvision103k_train.jsonl"
GLOB = f"{PROJ}/work/checkpoints/grpo_general/v*/completions.jsonl"
OUT_TRAIN = f"{PROJ}/work/data/sft_rft_coldstart_train.jsonl"
OUT_VAL = f"{PROJ}/work/data/sft_rft_coldstart_val.jsonl"
SYSTEM = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason "
          "step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the "
          "final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. "
          "<answer>A</answer>.")
MAX_CHARS = 6000      # 간결 기준(≈2000토큰). 잘리지 않고 마감된 것 중 짧은 것.
PER_Q = 3             # 질문당 최대 보관(가장 짧은 것부터)
VAL_N = 40

_ANS = re.compile(r'<answer>.*?</answer>', re.S)
_USER = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.S)


def closed(c):
    return ('</think>' in c) and bool(_ANS.search(c))


def question_of(prompt):
    m = _USER.search(prompt)
    if not m:
        return None
    return re.sub(r'<image>\s*', '', m.group(1)).strip()


def normalize_assistant(c):
    """저장된 완성문 → 완전한 assistant 타깃. 강제 prefix '<think>\\n' 복원 + im_end 제거."""
    c = c.split('<|im_end|>')[0].rstrip()
    if not c.lstrip().startswith('<think>'):
        c = '<think>\n' + c
    return c


# 1) DeepVision: 질문텍스트 → 이미지경로
q2img = {}
for line in open(DEEPVISION):
    d = json.loads(line)
    user = next((m['content'] for m in d['messages'] if m['role'] == 'user'), '')
    q = re.sub(r'<image>\s*', '', user).strip()
    imgs = d.get('images', [])
    if q and imgs:
        q2img.setdefault(q, imgs)
print(f"[1] DeepVision 질문 인덱스: {len(q2img)}")

# 2) 롤아웃에서 정답+마감+간결 수집 (질문당 짧은 것 위주)
cand = {}   # q -> list of completion strings
total = 0
for f in sorted(glob.glob(GLOB)):
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        for i in range(len(r['completion'])):
            c = r['completion'][i]
            if float(r['AccuracyMix'][i]) >= 1.0 and closed(c) and len(c) <= MAX_CHARS:
                q = question_of(r['prompt'][i])
                if q and q in q2img:
                    cand.setdefault(q, []).append(c)
                    total += 1
print(f"[2] 정답+마감+간결 완성문 {total}개 / 유니크 질문 {len(cand)}개")

# 3) 질문당 짧은 것 PER_Q개 → SFT 예시
examples = []
for q, comps in cand.items():
    comps = sorted(set(comps), key=len)[:PER_Q]
    for c in comps:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<image>\n{q}"},
                {"role": "assistant", "content": normalize_assistant(c)},
            ],
            "images": q2img[q],
        })
print(f"[3] SFT 예시 {len(examples)}개 (질문당 최대 {PER_Q})")

# 4) split (질문 단위로 섞되, 간단히 인덱스 기반 — 재현성 위해 정렬 후 stride)
examples.sort(key=lambda e: e['messages'][1]['content'])
val = examples[::max(1, len(examples) // VAL_N)][:VAL_N]
val_set = set(id(e) for e in val)
train = [e for e in examples if id(e) not in val_set]

with open(OUT_TRAIN, 'w') as f:
    for e in train:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(OUT_VAL, 'w') as f:
    for e in val:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

# 통계
tls = sorted(len(e['messages'][2]['content']) for e in examples)
print(f"[4] train {len(train)} / val {len(val)}  → {OUT_TRAIN}")
print(f"    assistant 길이(char): p50 {tls[len(tls)//2]} | p90 {tls[int(len(tls)*0.9)]} | max {tls[-1]}")
