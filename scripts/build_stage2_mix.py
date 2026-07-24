#!/usr/bin/env python3
"""build_stage2_mix.py — Stage-2 풀확장 데이터 조립 + 소스별 클린 홀드아웃.

구성 (2026-07-24, 의료 27% — docs/stage2_data.md):
  train   = DeepVision(서브샘플 DV_CAP) + MMK12(전량) + PMC-VQA(서브샘플 PMC_CAP)   ← ThinkLite 드롭
  holdout = DeepVision(972·기존) + MMK12(HOLD) + PMC-VQA(HOLD)
  · **전 소스 이미지 바이트해시 dedup**: DeepVision 도 홀드아웃 이미지해시를 train 서브샘플에서
    제외 → 구 22% 오염(경로만 다른 동일 그림)을 이번에 최종 해소.
  · 각 홀드아웃 행에 _source·_stratum(정답유형) 태그 → 소스/층별 리포트용.
seed=42. env 캡: DV_CAP(40000) PMC_CAP(20000). 산출: work/data/stage2_expanded_{train,holdout}.jsonl
"""
import json, hashlib, collections, random, re, os
random.seed(42)
D = 'work/data'
DV_CAP = int(os.environ.get('DV_CAP', 40000))
PMC_CAP = int(os.environ.get('PMC_CAP', 20000))
HOLD = {'mmk12': 400, 'pmcvqa': 400}


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


print(f'[mix] 신규 소스 홀드아웃 분리(bytehash dedup)  DV_CAP={DV_CAP:,} PMC_CAP={PMC_CAP:,}')
mm = load(f'{D}/mmk12_train.jsonl'); random.shuffle(mm)
mm_tr, mm_ho = carve(mm, HOLD['mmk12'], 'mmk12')

pmc = load(f'{D}/pmcvqa_train.jsonl'); random.shuffle(pmc)
pmc = pmc[:PMC_CAP]
pmc_tr, pmc_ho = carve(pmc, HOLD['pmcvqa'], 'pmcvqa')

# DeepVision: 홀드아웃(972·기존) 재사용 + 그 이미지해시를 trainonly 서브샘플에서 제외(22% 오염 최종 해소)
dv_ho = load(f'{D}/deepvision_holdout.jsonl')
ho_hashes = {imghash(r['images'][0]) for r in dv_ho}; ho_hashes.discard(None)
for r in dv_ho:
    r['_source'] = 'deepvision'
dv = load(f'{D}/deepvision103k_trainonly.jsonl'); random.shuffle(dv)
dv_tr, skipped = [], 0
for r in dv:
    if len(dv_tr) >= DV_CAP:
        break
    if imghash(r['images'][0]) in ho_hashes:   # 홀드아웃과 동일 이미지 → 오염 제외
        skipped += 1; continue
    r['_source'] = 'deepvision'
    dv_tr.append(r)

dump(dv_tr + mm_tr + pmc_tr, f'{D}/stage2_expanded_train.jsonl', tags=False)
dump(dv_ho + mm_ho + pmc_ho, f'{D}/stage2_expanded_holdout.jsonl', tags=True)

tot_tr = len(dv_tr) + len(mm_tr) + len(pmc_tr)
print(f'[mix] ✅ train   = DeepVision {len(dv_tr):,}(오염제외 {skipped}) + MMK12 {len(mm_tr):,} '
      f'+ PMC-VQA {len(pmc_tr):,} = {tot_tr:,}')
print(f'[mix]   비율: 일반 {len(dv_tr)/tot_tr*100:.0f} / math {len(mm_tr)/tot_tr*100:.0f} '
      f'/ 의료 {len(pmc_tr)/tot_tr*100:.0f}')
print(f'[mix] ✅ holdout = DeepVision {len(dv_ho)} + MMK12 {len(mm_ho)} + PMC-VQA {len(pmc_ho)} '
      f'= {len(dv_ho)+len(mm_ho)+len(pmc_ho)}')
