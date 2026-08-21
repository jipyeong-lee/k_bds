#!/bin/bash
# watch_expert.sh <arm> — 도메인 전문가 본실행의 붕괴를 조기 경보한다. 정상이면 무출력.
#
# 실행 위치: KISTI(~/kbds_project). 체인이 job 을 갈아타므로 arm 이름으로 현재 RUNNING 잡을 찾는다.
#
# ⚠️ 임계를 **절대값으로 고정하지 않는다.** 정상 길이가 도메인마다 다르다 —
#    deepvision 2,000~3,000 vs mmk12 855~1,532(job 75762 스모크). B200 에서 쓴 `len<900` 을
#    mmk12 에 그대로 쓰면 정상 상태에서 계속 울린다. 그래서 길이는 **자기 실행의 초기 구간 대비**로 본다.
#
# B200 실측에서 붕괴를 단조로 따라간 지표만 쓴다(README 발견 ⑨·⑪·⑫):
#   ppl_ratio(=tr/ro) · completions/mean_length.  ess·clipped_frac 은 정상/붕괴 구간이 겹쳐 못 쓴다.
#   지표는 평균이 아니라 **중앙값** — step 하나가 200 step 평균을 321 로 만든 적이 있다.
#
# 형식(fmt)은 **일부러 보지 않는다.** job 내부 watchdog(watch_format_collapse.py)이 fmt<0.85 로 이미 보고,
# 그 임계는 73924/73925 로그 재생 스윕으로 교정된 값이다. 여기서 더 조이면 오탐만 난다 —
# mmk12 스모크의 정상 FormatThink 가 0.898~1.0 이었다(job 75762). 대신 verdict 파일 생성만 감시한다.
set -u
ARM="${1:-mmk12}"
cd ~/kbds_project 2>/dev/null || { echo "ALERT cd 실패"; exit 0; }

JID=$(squeue -u "$USER" -h -n "e-$ARM" -t RUNNING -o %i 2>/dev/null | head -1)
if [[ -z "$JID" ]]; then
  # RUNNING 이 없으면 대기 중인지 확인한다. 둘 다 없으면 체인이 끝났거나 죽은 것이다.
  PEND=$(squeue -u "$USER" -h -n "e-$ARM" -o %i 2>/dev/null | wc -l)
  [[ "$PEND" -eq 0 ]] && echo "ALERT $ARM 체인에 job 이 없다 — 완주했거나 전부 실패했다(sacct 확인 필요)"
  exit 0
fi

LOG="logs/grpo_adv_${JID}.log"
[[ -f "$LOG" ]] || { echo "ALERT $ARM job $JID RUNNING 인데 로그 없음: $LOG"; exit 0; }

python3 - "$LOG" "$ARM" "$JID" <<'PY'
import re, sys, time, os, statistics as st
log, arm, jid = sys.argv[1], sys.argv[2], sys.argv[3]
s=open(log, encoding='utf-8', errors='replace').read()
recs=re.findall(r"\{'loss'.*?\}", s)
def g(r,k):
    m=re.search(re.escape("'"+k+"':")+r"\s*'?([^,'}]+)", r)
    try: return float(m.group(1))
    except Exception: return None
def med(rows,k):
    v=[g(r,k) for r in rows]; v=[x for x in v if x is not None]
    return st.median(v) if v else None
step=None
for r in reversed(recs):
    m=re.search(r"'global_step/max_steps':\s*'(\d+)/(\d+)", r)
    if m: step, total = int(m.group(1)), int(m.group(2)); break
a=[]

# 학습이 살아 있는가 — RUNNING 인데 로그가 멎으면 이상이다(job 교체 중이 아니다).
age=time.time()-os.path.getmtime(log)
if age > 2400: a.append(f"로그가 {int(age/60)}분째 갱신 안 됨")

if len(recs) >= 80:
    base, recent = recs[30:130], recs[-50:]
    lb, lr = med(base,'completions/mean_length'), med(recent,'completions/mean_length')
    if lb and lr and lr < 0.55*lb:
        a.append(f"길이 붕괴: {lr:.0f} < 초기({lb:.0f})의 55%")
    pr = med(recent,'rollout_correction/ppl_ratio')
    if pr and pr > 1.30: a.append(f"ppl_ratio={pr:.3f} > 1.30")   # 스모크 1.05 · B200 정상 1.10~1.16
    zs = med(recent,'frac_reward_zero_std')
    if zs is not None and zs > 0.3: a.append(f"frac_reward_zero_std={zs:.3f} > 0.3 (학습 신호 소실)")

if os.path.exists(f"logs/verdict_{jid}.json"):
    a.append(f"job 내부 watchdog 이 붕괴 판정을 기록했다 → logs/verdict_{jid}.json")

if a: print(f"ALERT {arm} job={jid} step={step}/{total if step else '?'} | " + " | ".join(a))
PY
exit 0
