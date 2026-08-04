#!/usr/bin/env python3
"""probe_format_fragility.py — 체크포인트별 "절벽까지의 거리" 실측.

질문: step 850 은 step 700 보다 형식 붕괴에 더 취약한가? 그 취약성이 붕괴 *이전에*
      측정 가능한 양인가?

방법: 동일 프롬프트 집합에 temperature 를 올려가며 롤아웃하고, 학습 때 쓴 것과
      **완전히 같은** FormatThink 보상 함수로 채점한다. temperature 는 증폭기다 —
      깨진 형식 basin 근처에 확률질량이 얼마나 쌓여 있는지를 드러낸다.

주 지표 = **비절단 형식실패율**. 절단(finish_reason=length)은 제외한다.
  학습 로그 분석에서 확인했듯 상시 절단 실패(5~10%)가 신호를 묻기 때문이다.
  조기경보 설계에 쓴 지표와 정의가 같아야 임계값을 그대로 옮겨 쓸 수 있다.

env: PROBE_BASE_URL / PROBE_MODEL / PROBE_TAG / PROBE_TEMPS / PROBE_NS /
     PROBE_MAXTOK / PROBE_CONC / PROBE_SEED / PROBE_DATA / PROBE_OUT / PROBE_RESULT
"""
import os, sys, json, base64, asyncio, random, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs'))
import accuracy as A          # 학습에 쓴 그 FormatThink 를 그대로 import  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

BASE_URL = os.environ['PROBE_BASE_URL']
MODEL    = os.environ['PROBE_MODEL']
TAG      = os.environ.get('PROBE_TAG', '?')
SYSTEM   = os.environ['SYSTEM_PROMPT']
# 학습 롤아웃과 같은 값. max_completion_length=6144 / temperature=0.9 가 학습 조건.
MAXTOK   = int(os.environ.get('PROBE_MAXTOK', '6144'))
CONC     = int(os.environ.get('PROBE_CONC', '32'))
SEED     = int(os.environ.get('PROBE_SEED', '20260804'))
DATA     = os.environ.get('PROBE_DATA', 'work/data/stage2_expanded_train.jsonl')
OUT      = os.environ.get('PROBE_OUT', f'logs/probe_frag_samples_{TAG}.jsonl')
RESULT   = os.environ.get('PROBE_RESULT', 'logs/probe_fragility_results.jsonl')

TEMPS = [float(x) for x in os.environ.get('PROBE_TEMPS', '0.0,0.9,1.2,1.5').split(',')]
NS    = [int(x) for x in os.environ.get('PROBE_NS', '200,350,350,350').split(',')]
assert len(TEMPS) == len(NS), 'PROBE_TEMPS 와 PROBE_NS 길이가 다르다'

FMT = A.FormatThink()


def source_of(row):
    p = (row.get('images') or ['none'])[0]
    parts = p.split('/images/')
    return parts[1].split('/')[0] if len(parts) > 1 else 'none'


def build_master(need):
    """소스 균등 층화 + 시드 고정 마스터 리스트.

    체크포인트·temperature 가 달라도 항상 같은 프롬프트를 같은 순서로 본다.
    셀마다 앞에서 n 개를 자르므로 작은 셀은 큰 셀의 부분집합이다(중첩 = 짝지음 가능).
    """
    buckets = {}
    for line in open(DATA):
        r = json.loads(line)
        buckets.setdefault(source_of(r), []).append(r)
    rng = random.Random(SEED)
    per = need // len(buckets) + 1
    picked = {}
    for k in sorted(buckets):
        g = buckets[k][:]
        rng.shuffle(g)
        picked[k] = g[:per]
    master, i = [], 0
    while len(master) < need:                 # 라운드로빈 → 어떤 prefix 도 소스 균등
        for k in sorted(picked):
            if i < len(picked[k]):
                master.append(picked[k][i])
        i += 1
    return master[:need]


def data_url(p):
    with open(p, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(p)[1].lstrip('.').lower() or 'png'
    return f'data:image/{ext};base64,{b}'


cli = AsyncOpenAI(base_url=BASE_URL, api_key='EMPTY', timeout=1800)
sem = asyncio.Semaphore(CONC)


async def gen(row, temp):
    q = row['messages'][0]['content'].replace('<image>', '').strip()
    content = [{'type': 'image_url', 'image_url': {'url': data_url(row['images'][0])}},
               {'type': 'text', 'text': q}]
    async with sem:
        try:
            r = await cli.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'system', 'content': SYSTEM},
                          {'role': 'user', 'content': content}],
                temperature=temp, max_tokens=MAXTOK)
            ch = r.choices[0]
            m = ch.message
            return {
                'text': getattr(m, 'content', None) or '',
                'reasoning': getattr(m, 'reasoning_content', None) or '',
                'finish': ch.finish_reason,
                'ctok': getattr(r.usage, 'completion_tokens', None) if r.usage else None,
                'src': source_of(row),
            }
        except Exception as e:
            return {'text': '', 'reasoning': '', 'finish': f'__ERR__{type(e).__name__}',
                    'ctok': None, 'src': source_of(row)}


def rep_ratio(s, n=30):
    """문자 n-gram 중복률. 공백 없는 태그 스팸(<think></think><think>…)은
    단어 단위로는 한 '단어'라 0% 가 나온다 — 반드시 문자 단위로 잰다."""
    if len(s) < n * 2:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


async def warmup():
    """reasoning parser 가 켜져 있으면 content 에서 <think> 가 사라져 형식 측정이 무의미해진다.
    조용히 재조립하면 실패를 성공으로 둔갑시키므로, 여기서 즉시 죽는다."""
    rows = build_master(2)
    outs = await asyncio.gather(*[gen(r, 0.0) for r in rows])
    for o in outs:
        if o['finish'].startswith('__ERR__'):
            print(f"[probe:{TAG}] ❌ warmup 실패: {o['finish']}", flush=True)
            sys.exit(1)
        if o['reasoning']:
            print(f"[probe:{TAG}] ❌ reasoning_content 가 채워졌다(파서 활성). "
                  f"content 에서 <think> 가 제거되어 형식 측정 불가.", flush=True)
            sys.exit(2)
    head = outs[0]['text'][:120].replace('\n', '\\n')
    print(f"[probe:{TAG}] warmup OK · starts_with_think="
          f"{outs[0]['text'].lstrip().startswith('<think>')} · head={head!r}", flush=True)


async def main():
    await warmup()
    master = build_master(max(NS))
    fout = open(OUT, 'w')
    for temp, n in zip(TEMPS, NS):
        rows = master[:n]
        outs = await asyncio.gather(*[gen(r, temp) for r in rows])
        errs = [o for o in outs if o['finish'].startswith('__ERR__')]
        ok = [o for o in outs if not o['finish'].startswith('__ERR__')]
        if not ok:
            print(f"[probe:{TAG}] T={temp} ❌ 전부 실패 {errs[0]['finish']}", flush=True)
            continue
        scores = FMT([o['text'] for o in ok])
        trunc = [o['finish'] == 'length' for o in ok]

        nt = [s for s, t in zip(scores, trunc) if not t]     # 비절단만
        prim = (1 - st.mean(nt)) if nt else float('nan')     # ← 주 지표
        rec = {
            'tag': TAG, 'temp': temp, 'n': len(ok), 'errors': len(errs),
            'fail_nontrunc': round(prim, 5),                 # 비절단 형식실패율
            'n_nontrunc': len(nt),
            'fail_all': round(1 - st.mean(scores), 5),
            'trunc_rate': round(st.mean(trunc), 5),
            'mean_chars': round(st.mean(len(o['text']) for o in ok), 0),
            'mean_ctok': round(st.mean([o['ctok'] for o in ok if o['ctok']] or [0]), 0),
            'p90_chars': round(sorted(len(o['text']) for o in ok)[int(.9 * len(ok))], 0),
            'rep_rate': round(st.mean(rep_ratio(o['text']) > 0.5 for o in ok), 4),
            'starts_think': round(st.mean(o['text'].lstrip().startswith('<think>') for o in ok), 3),
        }
        by = {}
        for o, s, t in zip(ok, scores, trunc):
            if not t:
                by.setdefault(o['src'], []).append(s)
        rec['by_source'] = {k: round(1 - st.mean(v), 4) for k, v in sorted(by.items())}
        print(f"[probe:{TAG}] T={temp:<4} n={len(ok):<4} 비절단실패={prim:.4f} "
              f"({sum(1 for s,t in zip(scores,trunc) if not t and s==0)}/{len(nt)}) "
              f"절단={rec['trunc_rate']:.3f} chars={rec['mean_chars']:.0f} "
              f"반복={rec['rep_rate']:.3f} src={rec['by_source']}", flush=True)
        with open(RESULT, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        for o, s, t in zip(ok, scores, trunc):
            fout.write(json.dumps({'tag': TAG, 'temp': temp, 'src': o['src'],
                                   'fmt': s, 'trunc': t, 'ctok': o['ctok'],
                                   'text': o['text'][:4000]}) + '\n')
        fout.flush()
    fout.close()
    print(f"[probe:{TAG}] done → {RESULT} / {OUT}", flush=True)


asyncio.run(main())
