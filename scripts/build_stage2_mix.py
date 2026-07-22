#!/usr/bin/env python3
"""build_stage2_mix.py — Stage-2 풀확장 데이터 조립 + 소스별 클린 홀드아웃.

train   = DeepVision trainonly(102,531) + MMK12 + ThinkLite  (신규는 홀드아웃 이미지해시 제외)
holdout = DeepVision holdout(972·기존) + MMK12 holdout + ThinkLite holdout
  · 신규 2종 홀드아웃은 **이미지 바이트해시 dedup**(경로 아닌 해시 기준 → 22% 오염 재발 방지)
  · 각 홀드아웃 행에 _source·_stratum(정답유형) 태그 → 소스/층별 리포트용
  · DeepVision 은 기존 split 재사용(102k 재해싱 회피, 기존 Stage-2 와 연속성)
seed=42. 산출: work/data/stage2_expanded_{train,holdout}.jsonl
"""
import json, hashlib, collections, random, re
random.seed(42)
D = 'work/data'
HOLD_MMK12, HOLD_THINKLITE = 400, 300


def imghash(p):
    try:
        with open(p, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def load(f):
    return [json.loads(l) for l in open(f)]


def ans_kind(s):
    s = str(s).strip()
    if re.fullmatch(r'[A-Ha-h]', s):        return 'letter'
    if re.fullmatch(r'-?[\d,]+\.?\d*', s):  return 'numeric'
    if re.search(r'[√π/^=]|\\', s):         return 'symbolic'
    return 'other'


def carve(rows, n_hold, source):
    """정답유형 층화 홀드아웃 n_hold 뽑고, 그 이미지해시를 train 에서 제외."""
    for r in rows:
        r['_source'], r['_stratum'] = source, ans_kind(r['solution'])
    bystrat = collections.defaultdict(list)
    for r in rows:
        bystrat[r['_stratum']].append(r)
    N, hold = len(rows), []
    for s, rs in bystrat.items():
        random.shuffle(rs)
        hold += rs[:max(1, round(n_hold * len(rs) / N))]
    hh = {imghash(r['images'][0]) for r in hold}
    hh.discard(None)
    train = [r for r in rows if imghash(r['images'][0]) not in hh]
    return train, hold


def dump(rows, path, tags):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            o = {'messages': r['messages'], 'images': r['images'], 'solution': r['solution']}
            if tags:
                o['_source'] = r.get('_source', '?')
                o['_stratum'] = r.get('_stratum', 'all')
            f.write(json.dumps(o, ensure_ascii=False) + '\n')


print('[mix] 신규 소스 홀드아웃 분리(bytehash dedup)...')
mm_tr, mm_ho = carve(load(f'{D}/mmk12_train.jsonl'), HOLD_MMK12, 'mmk12')
tl_tr, tl_ho = carve(load(f'{D}/thinklite_train.jsonl'), HOLD_THINKLITE, 'thinklite')

dv_tr = load(f'{D}/deepvision103k_trainonly.jsonl')
for r in dv_tr:
    r['_source'] = 'deepvision'
dv_ho = load(f'{D}/deepvision_holdout.jsonl')      # 이미 _stratum(math/vl) 보유
for r in dv_ho:
    r['_source'] = 'deepvision'

dump(dv_tr + mm_tr + tl_tr, f'{D}/stage2_expanded_train.jsonl', tags=False)
dump(dv_ho + mm_ho + tl_ho, f'{D}/stage2_expanded_holdout.jsonl', tags=True)

print(f'[mix] ✅ train   = DeepVision {len(dv_tr):,} + MMK12 {len(mm_tr):,} + ThinkLite {len(tl_tr):,} '
      f'= {len(dv_tr)+len(mm_tr)+len(tl_tr):,}')
print(f'[mix] ✅ holdout = DeepVision {len(dv_ho)} + MMK12 {len(mm_ho)} + ThinkLite {len(tl_ho)} '
      f'= {len(dv_ho)+len(mm_ho)+len(tl_ho)}')
