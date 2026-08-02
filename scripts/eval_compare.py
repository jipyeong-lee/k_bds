#!/usr/bin/env python3
"""eval_compare.py — 라이브 vLLM OpenAI 엔드포인트로 DeepVision 홀드아웃 평가.

base vs 학습모델을 동일 조건(시스템프롬프트·생성설정)으로 채점해 비교.
검증가능 정답(DeepVision solution)을 accuracy_mix(math/letter/문자열)로 자동 채점.
env: EVAL_BASE_URL / EVAL_MODEL(served name) / EVAL_N(150) / EVAL_CONC(8) / SYSTEM_PROMPT / EVAL_TAG
"""
import os, sys, json, base64, asyncio, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import accuracy as A  # accuracy_mix scorer  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

N = int(os.environ.get('EVAL_N', '150'))
CONC = int(os.environ.get('EVAL_CONC', '8'))
TAG = os.environ.get('EVAL_TAG', '?')
SYSTEM = os.environ['SYSTEM_PROMPT']
MODEL = os.environ['EVAL_MODEL']

# 평가셋 = 진짜 홀드아웃(학습 trainonly 에서 제외). base/trained 동일 슬라이스.
EVAL_DATA = os.environ.get('EVAL_DATA', 'work/data/deepvision_holdout.jsonl')
all_rows = [json.loads(l) for l in open(EVAL_DATA)]

# ⚠️ 2026-08-02: 종전에는 앞에서 N 줄을 잘랐다. 확장 홀드아웃(stage2_expanded_holdout.jsonl)은
#    deepvision(972) → mmk12(400) → pmcvqa(400) 순으로 정렬돼 있어 N≤972 면 전부 deepvision 이
#    뽑혀 math·의료 효과가 아예 측정되지 않았다. → _source 별 균등 층화 추출로 교체.
#    시드 고정이라 base/init/trained 가 동일 슬라이스를 본다(비교 가능성 보존).
groups = {}
for r in all_rows:
    groups.setdefault(r.get('_source', 'all'), []).append(r)

if len(groups) <= 1:
    rows = all_rows[:N]                      # 구 홀드아웃(_source 없음) — 종전 동작 유지
else:
    import random
    rng = random.Random(int(os.environ.get('EVAL_SEED', '20260802')))
    per = max(1, N // len(groups))
    rows = []
    for k in sorted(groups):
        g = groups[k][:]
        rng.shuffle(g)
        rows.extend(g[:per])

src_n = {}
for r in rows:
    src_n[r.get('_source', 'all')] = src_n.get(r.get('_source', 'all'), 0) + 1
print(f"[eval:{TAG}] model={MODEL} N={len(rows)} conc={CONC} sources={src_n}")

cli = AsyncOpenAI(base_url=os.environ['EVAL_BASE_URL'], api_key='EMPTY')
sem = asyncio.Semaphore(CONC)


def data_url(p):
    with open(p, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(p)[1].lstrip('.').lower() or 'png'
    return f'data:image/{ext};base64,{b}'


async def infer(row):
    q = row['messages'][0]['content'].replace('<image>', '').strip()
    content = [{'type': 'image_url', 'image_url': {'url': data_url(row['images'][0])}},
               {'type': 'text', 'text': q}]
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
    # accuracy_mix 채점
    am = A.AccuracyMix()
    scores = am(preds, sols)
    acc = st.mean(scores)
    # 형식 준수율(닫힌 <answer>)·평균 길이
    import re
    fmt = sum(1 for p in preds if re.search(r'<answer>.*?</answer>', p, re.S)) / len(preds)
    mlen = st.mean(len(p) for p in preds)
    print(f"[eval:{TAG}] accuracy={acc:.4f}  format(<answer>)={fmt:.2f}  mean_chars={mlen:.0f}  errors={errs}/{len(preds)}")
    # 층(stratum)별 분리 보고 — 홀드아웃 라인의 _stratum (math / vl) 기준
    strata = {}
    for r, s in zip(rows, scores):
        k = r.get('_stratum', 'all')
        strata.setdefault(k, []).append(s)
    per_stratum = {k: round(st.mean(v), 4) for k, v in strata.items()}
    for k in sorted(strata):
        print(f"          └ [{k:9}] accuracy={st.mean(strata[k]):.4f}  (n={len(strata[k])})")
    # 소스별 분리 보고 — 확장셋의 핵심 질문(math=mmk12 / 의료=pmcvqa 가 올랐나)
    srcs = {}
    for r, s in zip(rows, scores):
        srcs.setdefault(r.get('_source', 'all'), []).append(s)
    per_source = {k: round(st.mean(v), 4) for k, v in srcs.items()}
    if len(srcs) > 1:
        for k in sorted(srcs):
            print(f"          ▶ [{k:9}] accuracy={st.mean(srcs[k]):.4f}  (n={len(srcs[k])})")
    # 문항별 점수 덤프 — 체크포인트 간 "짝지음(McNemar)" 비교의 전제조건.
    # 집계값만 남기면 비짝지음 검정밖에 못 해 검출 하한이 8.0pp(n=300)로 뜬다.
    # 같은 시드·같은 슬라이스라 item_id 로 조인하면 하한이 5.7pp(n=300)·2.8pp(n=1200)로 내려간다.
    items_path = os.environ.get('EVAL_ITEMS')
    if items_path:
        with open(items_path, 'w') as f:
            for r, s in zip(rows, scores):
                f.write(json.dumps({
                    'item_id': r.get('images', [None])[0] or r.get('id'),
                    'tag': TAG,
                    'source': r.get('_source', 'all'),
                    'stratum': r.get('_stratum', 'all'),
                    'score': s,
                }) + '\n')
        print(f"[eval:{TAG}] per-item scores → {items_path} ({len(rows)} rows)")

    # 결과 파일(머신리더블) append
    with open(os.environ.get('EVAL_RESULT', 'logs/eval_compare_results.jsonl'), 'a') as f:
        f.write(json.dumps({'tag': TAG, 'model': MODEL, 'n': len(rows),
                            'accuracy': round(acc, 4), 'format': round(fmt, 3),
                            'mean_chars': round(mlen, 0), 'errors': errs,
                            'per_stratum': per_stratum, 'per_source': per_source}) + '\n')

asyncio.run(main())
