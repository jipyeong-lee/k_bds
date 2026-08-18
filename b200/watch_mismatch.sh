#!/bin/bash
# watch_mismatch.sh — 롤아웃/학습 괴리 붕괴를 조기 경보한다. 정상이면 무출력.
# 실행 위치: KISTI(/scratch/migrate_k266_to_gpu) — 플랫폼 자격증명이 거기 있고 pull_log.sh·parse_log.py 를 쓴다.
#
# 단독으로 붕괴를 판별하는 지표는 tr_ppl/ro_ppl · fmt · len 셋뿐이다(1차 실측):
#   tr/ro  1.098 → 1.162 → 1.348 → 208.5   (단조 발산)
#   fmt    0.993 → 0.957 → 0.936           (형식 붕괴)
#   len    1487 → 665 → 502                (길이 붕괴)
# ess·ppl_abs_diff·clipped_frac 은 단독으로 쓰면 오탐이다 — 1차는 붕괴가 다 진행된 bin 1000~1200 에서도
# ess 가 0.9646/0.9665/0.9645 였고, 2차는 건강한 상태에서 0.9620 을 찍었다(2026-08-19 실측).
# clipped_frac 은 발산이 깊어지는 동안 오히려 내려간다(README 발견 ⑨). → 이 셋은 2개 이상 겹칠 때만 건다.
cd /scratch/migrate_k266_to_gpu || { echo "ALERT cd 실패"; exit 0; }
bash pull_log.sh train_deepvision_ep1_gdpo_async_tis_entmask.log t2.log >/dev/null 2>&1 \
  || { echo "ALERT 로그 pull 실패"; exit 0; }
python3 parse_log.py t2.log m.csv >/dev/null 2>&1 || { echo "ALERT parse 실패"; exit 0; }
python3 - <<'PY'
import csv, time
r=list(csv.DictReader(open('m.csv')))
def f(x):
    try: return float(x)
    except: return None
def m(c, n=50):
    v=[f(x[c]) for x in r[-n:] if f(x[c]) is not None]
    return sum(v)/len(v) if v else None
step=int(r[-1]['step'])
pd, ro, tr, es, cf = m('ppl_abs_diff'), m('ro_ppl'), m('tr_ppl'), m('ess'), m('clipped_frac')
fm, ln = m('fmt'), m('len')
rat = tr/ro if ro and tr else None

a=[]                                   # 단독 트리거 — 1차에서 단조였던 것만
if rat and rat > 1.15 : a.append(f"tr_ppl/ro_ppl={rat:.4f} > 1.15")
if fm  and fm  < 0.98 : a.append(f"fmt={fm:.4f} < 0.98")
if ln  and ln  < 1200 : a.append(f"len={ln:.0f} < 1200")

weak=[]                                # 보조 지표 — 2개 이상 겹칠 때만 경보
if pd and pd > 0.12  : weak.append(f"ppl_abs_diff={pd:.4f}")
if es and es < 0.965 : weak.append(f"ess={es:.4f}")
if cf and cf > 0.02  : weak.append(f"clipped_frac={cf:.5f}")
if len(weak) >= 2: a.append("보조지표 " + "+".join(weak))

# 정체 판정은 시각 기준이다 — job 교체는 정상적으로 10~30분 걸린다(TIME_WAIT/vLLM 재기동).
p='.watch_prev_step'; now=time.time()
try:    prev, since = open(p).read().split()
except Exception: prev, since = '', now
prev, since = (prev or str(step)), float(since)
if str(step) != prev: since = now
open(p,'w').write(f"{step} {since}")
if str(step) == prev and now - since > 2400:
    a.append(f"step {step} 에서 {int((now-since)/60)}분째 정체")
if a: print(f"ALERT step={step} | " + " | ".join(a))
PY
[ "$(pgrep -fc 'chain_epoch.sh deepvision' || echo 0)" -eq 0 ] && echo "ALERT chain 프로세스 죽음"
exit 0
