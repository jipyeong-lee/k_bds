#!/usr/bin/env python3
"""eval_global_bench.py — MedXpertQA-MM + MMMU-H&M(객관식) 글로벌 벤치마크 평가.

원 과제계획서(양식1) 약속: "글로벌 의료 벤치마크(MedXpertQA-MM, MMMU-H&M)로 SFT 모델
대비 성능 향상을 정량 입증". eval_compare.py 와 동일 하니스(vLLM OpenAI 서버 + accuracy_mix)
지만 두 벤치마크는 문항당 이미지가 여러 장이라 다중 이미지를 보낸다.

env: EVAL_BASE_URL / EVAL_MODEL / EVAL_DATA / EVAL_N(all) / EVAL_CONC(8) / SYSTEM_PROMPT / EVAL_TAG
"""
import os, sys, json, base64, asyncio, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import accuracy as A  # accuracy_mix scorer  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

CONC = int(os.environ.get('EVAL_CONC', '8'))
TAG = os.environ.get('EVAL_TAG', '?')
SYSTEM = os.environ['SYSTEM_PROMPT']
MODEL = os.environ['EVAL_MODEL']

EVAL_DATA = os.environ.get('EVAL_DATA', 'work/data/global_bench/global_bench.jsonl')
all_rows = [json.loads(l) for l in open(EVAL_DATA)]

# 소스별(medxpertqa_mm/mmmu_hm) 균등 층화 — eval_compare.py 와 동일 정책.
groups = {}
for r in all_rows:
    groups.setdefault(r.get('_source', 'all'), []).append(r)

_n = os.environ.get('EVAL_N', 'all').strip().lower()
if _n in ('all', '0', '-1'):
    rows = all_rows
else:
    import random
    N = int(_n)
    rng = random.Random(int(os.environ.get('EVAL_SEED', '20260908')))
    per = max(1, N // len(groups))
    rows = []
    for k in sorted(groups):
        g = groups[k][:]
        rng.shuffle(g)
        rows.extend(g[:per])

src_n = {}
for r in rows:
    src_n[r.get('_source', 'all')] = src_n.get(r.get('_source', 'all'), 0) + 1
print(f"[gbench:{TAG}] model={MODEL} N={len(rows)} conc={CONC} sources={src_n}")

cli = AsyncOpenAI(base_url=os.environ['EVAL_BASE_URL'], api_key=os.environ.get('EVAL_API_KEY', 'EMPTY'))
sem = asyncio.Semaphore(CONC)


def data_url(p):
    with open(p, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(p)[1].lstrip('.').lower() or 'png'
    return f'data:image/{ext};base64,{b}'


async def infer(row):
    q = row['messages'][0]['content'].replace('<image>', '').strip()
    content = [{'type': 'image_url', 'image_url': {'url': data_url(p)}} for p in row['images']]
    content.append({'type': 'text', 'text': q})
    async with sem:
        try:
            r = await cli.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
                temperature=0.0, max_tokens=int(os.environ.get('EVAL_MAXTOK', '4096')))
            m = r.choices[0].message
            return (getattr(m, 'content', None) or getattr(m, 'reasoning_content', None) or '')
        except Exception as e:
            return f'__ERR__{type(e).__name__}'


async def main():
    preds = await asyncio.gather(*[infer(r) for r in rows])
    sols = [str(r['solution']) for r in rows]
    errs = sum(1 for p in preds if p.startswith('__ERR__'))
    am = A.AccuracyMix()
    scores = am(preds, sols)
    acc = st.mean(scores)
    import re
    fmt = sum(1 for p in preds if re.search(r'<answer>.*?</answer>', p, re.S)) / len(preds)
    mlen = st.mean(len(p) for p in preds)
    print(f"[gbench:{TAG}] accuracy={acc:.4f}  format(<answer>)={fmt:.2f}  mean_chars={mlen:.0f}  errors={errs}/{len(preds)}")

    srcs = {}
    for r, s in zip(rows, scores):
        srcs.setdefault(r.get('_source', 'all'), []).append(s)
    per_source = {k: round(st.mean(v), 4) for k, v in srcs.items()}
    for k in sorted(srcs):
        print(f"          ▶ [{k:14}] accuracy={st.mean(srcs[k]):.4f}  (n={len(srcs[k])})")

    strata = {}
    for r, s in zip(rows, scores):
        k = r.get('_stratum', 'all')
        strata.setdefault(k, []).append(s)
    per_stratum = {k: round(st.mean(v), 4) for k, v in strata.items()}
    for k in sorted(strata):
        print(f"              └ [{k:14}] accuracy={st.mean(strata[k]):.4f}  (n={len(strata[k])})")

    items_path = os.environ.get('EVAL_ITEMS')
    if items_path:
        with open(items_path, 'w') as f:
            for r, s in zip(rows, scores):
                f.write(json.dumps({
                    'item_id': r.get('images', [None])[0],
                    'tag': TAG, 'source': r.get('_source', 'all'),
                    'stratum': r.get('_stratum', 'all'), 'score': s,
                }) + '\n')
        print(f"[gbench:{TAG}] per-item scores → {items_path} ({len(rows)} rows)")

    with open(os.environ.get('EVAL_RESULT', 'logs/eval_global_bench_results.jsonl'), 'a') as f:
        f.write(json.dumps({'tag': TAG, 'model': MODEL, 'n': len(rows),
                            'accuracy': round(acc, 4), 'format': round(fmt, 3),
                            'mean_chars': round(mlen, 0), 'errors': errs,
                            'per_source': per_source, 'per_stratum': per_stratum}) + '\n')

asyncio.run(main())
