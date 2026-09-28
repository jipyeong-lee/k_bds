#!/usr/bin/env python3
"""eval_pass_at_k.py — init 모델의 pass@k 측정 (지식 부재 vs reward 신호 부족 가설 검증).

같은 질문을 K번 재샘플링(temperature=1.0, 학습과 동일)해서 pass@k 가 pass@1 보다
확 오르면(지식은 있는데 RL 이 못 끌어냄) / 그대로면(지식 자체가 천장) 을 가른다.
env: EVAL_BASE_URL / EVAL_MODEL / EVAL_DATA / EVAL_K(8) / EVAL_N(all) / EVAL_CONC(8) / EVAL_TAG
"""
import os, sys, json, base64, asyncio, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import accuracy as A  # accuracy_mix scorer  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

K = int(os.environ.get('EVAL_K', '8'))
CONC = int(os.environ.get('EVAL_CONC', '8'))
TAG = os.environ.get('EVAL_TAG', '?')
SYSTEM = os.environ['SYSTEM_PROMPT']
MODEL = os.environ['EVAL_MODEL']

EVAL_DATA = os.environ.get('EVAL_DATA', 'work/data/pmcvqa_only_holdout.jsonl')
rows = [json.loads(l) for l in open(EVAL_DATA)]
_n = os.environ.get('EVAL_N', 'all').strip().lower()
if _n not in ('all', '0', '-1'):
    rows = rows[:int(_n)]

print(f"[passk:{TAG}] model={MODEL} N={len(rows)} K={K} conc={CONC}")

cli = AsyncOpenAI(base_url=os.environ['EVAL_BASE_URL'], api_key=os.environ.get('EVAL_API_KEY', 'EMPTY'))
sem = asyncio.Semaphore(CONC)


def data_url(p):
    with open(p, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(p)[1].lstrip('.').lower() or 'png'
    return f'data:image/{ext};base64,{b}'


async def infer_k(row):
    q = row['messages'][0]['content'].replace('<image>', '').strip()
    content = [{'type': 'image_url', 'image_url': {'url': data_url(row['images'][0])}},
               {'type': 'text', 'text': q}]
    async with sem:
        try:
            r = await cli.chat.completions.create(
                model=MODEL, n=K,
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
                temperature=1.0, max_tokens=int(os.environ.get('EVAL_MAXTOK', '8192')))
            return [(getattr(c.message, 'content', None) or getattr(c.message, 'reasoning_content', None) or '')
                    for c in r.choices]
        except Exception as e:
            return [f'__ERR__{type(e).__name__}'] * K


async def main():
    preds_k = await asyncio.gather(*[infer_k(r) for r in rows])
    sols = [str(r['solution']) for r in rows]
    am = A.AccuracyMix()

    per_item_rate = []  # K개 중 정답 비율(0~1) — 문항별로 "안정적으로 아는지" 지표
    pass1, passk = [], []
    for preds, sol in zip(preds_k, sols):
        scores = am(preds, [sol] * len(preds))
        per_item_rate.append(sum(scores) / len(scores))
        pass1.append(scores[0])
        passk.append(1.0 if sum(scores) > 0 else 0.0)

    p1, pk = st.mean(pass1), st.mean(passk)
    print(f"[passk:{TAG}] pass@1={p1:.4f}  pass@{K}={pk:.4f}  (Δ={pk - p1:+.4f})")

    # 문항별 정답률 분포 — 0/1 에 몰려 있으면 이산적(지식 유무), 중간에 퍼져 있으면
    # 연속적(추론 노이즈). frac_reward_zero_std 관찰(학습로그)의 init-모델판.
    buckets = {'0%(전부 오답)': 0, '중간': 0, '100%(전부 정답)': 0}
    for r in per_item_rate:
        if r == 0:
            buckets['0%(전부 오답)'] += 1
        elif r == 1:
            buckets['100%(전부 정답)'] += 1
        else:
            buckets['중간'] += 1
    n = len(per_item_rate)
    for k, v in buckets.items():
        print(f"          └ {k}: {v}/{n} ({v / n:.1%})")

    with open(os.environ.get('EVAL_RESULT', 'logs/eval_pass_at_k_results.jsonl'), 'a') as f:
        f.write(json.dumps({'tag': TAG, 'model': MODEL, 'n': n, 'k': K,
                            'pass1': round(p1, 4), 'passk': round(pk, 4),
                            'bucket_zero': buckets['0%(전부 오답)'],
                            'bucket_mid': buckets['중간'],
                            'bucket_full': buckets['100%(전부 정답)']}) + '\n')

    items_path = os.environ.get('EVAL_ITEMS')
    if items_path:
        with open(items_path, 'w') as f:
            for r, rate in zip(rows, per_item_rate):
                f.write(json.dumps({'item_id': r.get('images', [None])[0] or r.get('id'),
                                    'success_rate': round(rate, 4)}) + '\n')
        print(f"[passk:{TAG}] 문항별 성공률 → {items_path} ({n} rows)")

asyncio.run(main())
