#!/usr/bin/env python3
"""eval_v3_holdout.py — v3(sft_mixed) 콜드스타트 홀드아웃 평가 (eval_compare 확장판).

eval_compare.py 와 동일 조건(같은 홀드아웃·시스템프롬프트·temp0)으로 채점하되,
v3 의 핵심 질문(콜드스타트 데이터 format_think=1.0 이 실제 모델출력 format_think 를
끌어올렸는가 = 'RL 형식천장' 규명)에 답하기 위해 **엄격 format_think**(configs/accuracy
의 FormatThink 와 동일 규칙: 앵커드 구조 + think 실질길이>=16)를 추가로 보고한다.
또 사후분석용으로 per-sample 예측을 덤프한다.

env: EVAL_BASE_URL / EVAL_MODEL / EVAL_N / EVAL_CONC / SYSTEM_PROMPT / EVAL_TAG
     EVAL_MAXTOK / EVAL_DATA / EVAL_RESULT / EVAL_PRED_DUMP
"""
import os, sys, json, base64, asyncio, re, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))
import accuracy as A  # AccuracyMix scorer  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

N = int(os.environ.get('EVAL_N', '972'))
CONC = int(os.environ.get('EVAL_CONC', '8'))
TAG = os.environ.get('EVAL_TAG', 'v3')
SYSTEM = os.environ['SYSTEM_PROMPT']
MODEL = os.environ['EVAL_MODEL']
EVAL_DATA = os.environ.get('EVAL_DATA', 'work/data/deepvision_holdout.jsonl')
PRED_DUMP = os.environ.get('EVAL_PRED_DUMP', f'logs/eval_v3_preds_{TAG}.jsonl')

# --- 엄격 format_think (configs/accuracy.FormatThink 와 동일 규칙, 독립 구현) ----------
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_FULL_RE = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])', re.DOTALL | re.MULTILINE)
_MIN_THINK_CHARS = 16


def strict_format_think(content: str) -> float:
    text = content if content.lstrip().startswith('<think>') else '<think>\n' + content
    if not _FULL_RE.match(text):
        return 0.0
    m = _THINK_RE.search(text)
    think = re.sub(r'\s+', '', m.group(1) if m else '')
    return 1.0 if len(think) >= _MIN_THINK_CHARS else 0.0


rows = []
for l in open(EVAL_DATA):
    rows.append(json.loads(l))
    if len(rows) >= N:
        break
print(f"[eval:{TAG}] model={MODEL} N={len(rows)} conc={CONC} data={EVAL_DATA}")

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
            # 사고형 모델이 think 를 reasoning_content 로 분리하면 <think> 로 재조립해 형식평가 보존
            c = getattr(m, 'content', None) or ''
            rc = getattr(m, 'reasoning_content', None) or ''
            if rc and '<think>' not in c:
                c = f'<think>\n{rc}\n</think>\n{c}'
            return c or '__EMPTY__'
        except Exception as e:
            return f'__ERR__{type(e).__name__}'


async def main():
    preds = await asyncio.gather(*[infer(r) for r in rows])
    sols = [str(r['solution']) for r in rows]
    errs = sum(1 for p in preds if p.startswith('__ERR__'))
    am = A.AccuracyMix()
    scores = am(preds, sols)
    acc = st.mean(scores)
    fmt_loose = sum(1 for p in preds if re.search(r'<answer>.*?</answer>', p, re.S)) / len(preds)
    fmt_strict = st.mean(strict_format_think(p) for p in preds)   # v3 핵심지표
    mlen = st.mean(len(p) for p in preds)
    print(f"[eval:{TAG}] accuracy={acc:.4f}  format_think(STRICT)={fmt_strict:.3f}  "
          f"format(<answer>)={fmt_loose:.2f}  mean_chars={mlen:.0f}  errors={errs}/{len(preds)}")
    # 층별
    strata = {}
    for r, s in zip(rows, scores):
        strata.setdefault(r.get('_stratum', 'all'), []).append(s)
    per_stratum = {k: round(st.mean(v), 4) for k, v in strata.items()}
    for k in sorted(strata):
        print(f"          └ [{k:5}] accuracy={st.mean(strata[k]):.4f}  (n={len(strata[k])})")
    # per-sample 덤프(사후분석: 어떤 문항서 형식/정답 깨지는지)
    with open(PRED_DUMP, 'w') as f:
        for r, p, s in zip(rows, preds, scores):
            f.write(json.dumps({'stratum': r.get('_stratum', 'all'), 'solution': str(r['solution']),
                                'score': s, 'fmt_strict': strict_format_think(p),
                                'pred': p[:4000]}, ensure_ascii=False) + '\n')
    with open(os.environ.get('EVAL_RESULT', 'logs/eval_v3_results.jsonl'), 'a') as f:
        f.write(json.dumps({'tag': TAG, 'model': MODEL, 'n': len(rows),
                            'accuracy': round(acc, 4), 'format_think_strict': round(fmt_strict, 4),
                            'format_answer': round(fmt_loose, 3), 'mean_chars': round(mlen, 0),
                            'errors': errs, 'per_stratum': per_stratum}) + '\n')
    print(f"[eval:{TAG}] preds -> {PRED_DUMP}")

asyncio.run(main())
