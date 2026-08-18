#!/bin/bash
# watch_mismatch.sh — 롤아웃/학습 괴리 붕괴를 조기 경보한다. 정상이면 무출력.
# 실행 위치: KISTI(/scratch/migrate_k266_to_gpu) — 플랫폼 자격증명이 거기 있고 pull_log.sh·parse_log.py 를 쓴다.
# 임계는 1차 붕괴 실측에서 뽑았다(bin 700 = 붕괴 시작점):
#   ppl_abs_diff 0.0756→0.1267 · tr_ppl/ro_ppl 1.0975→1.1623 · ess 0.9741→0.9559
# clipped_frac 은 알람으로 쓰지 않는다 — 1차에서 0.0148 을 찍고 붕괴가 깊어지는 동안 0.009 로 내려갔다.
cd /scratch/migrate_k266_to_gpu || { echo "ALERT cd 실패"; exit 0; }
bash pull_log.sh train_deepvision_ep1_gdpo_async_tis_entmask.log t2.log >/dev/null 2>&1 \
  || { echo "ALERT 로그 pull 실패"; exit 0; }
python3 parse_log.py t2.log m.csv >/dev/null 2>&1 || { echo "ALERT parse 실패"; exit 0; }
python3 - <<'PY'
import csv, os
r=list(csv.DictReader(open('m.csv')))
def f(x):
    try: return float(x)
    except: return None
def m(c, n=50):
    v=[f(x[c]) for x in r[-n:] if f(x[c]) is not None]
    return sum(v)/len(v) if v else None
step=int(r[-1]['step'])
pd, ro, tr, es, cf = m('ppl_abs_diff'), m('ro_ppl'), m('tr_ppl'), m('ess'), m('clipped_frac')
rat = tr/ro if ro and tr else None
a=[]
if pd  and pd  > 0.12  : a.append(f"ppl_abs_diff={pd:.4f} > 0.12")
if rat and rat > 1.15  : a.append(f"tr_ppl/ro_ppl={rat:.4f} > 1.15")
if es  and es  < 0.965 : a.append(f"ess={es:.4f} < 0.965")
if cf  and cf  > 0.02  : a.append(f"clipped_frac={cf:.5f} > 0.02")
# 정체 판정은 시각 기준이다 — job 교체는 정상적으로 10~30분 걸린다(TIME_WAIT/vLLM 재기동).
p='.watch_prev_step'; import time; now=time.time()
try:    prev, since = open(p).read().split()
except Exception: prev, since = '', now
prev, since = (prev or str(step)), float(since)
if str(step) != prev: since = now          # 진행했으면 시계 리셋
open(p,'w').write(f"{step} {since}")
if str(step) == prev and now - since > 2400:
    a.append(f"step {step} 에서 {int((now-since)/60)}분째 정체")
if a: print(f"ALERT step={step} | " + " | ".join(a))
PY
[ "$(pgrep -fc 'chain_epoch.sh deepvision' || echo 0)" -eq 0 ] && echo "ALERT chain 프로세스 죽음"
exit 0
