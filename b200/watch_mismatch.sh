#!/bin/bash
# watch_mismatch.sh — 롤아웃/학습 괴리 붕괴를 조기 경보한다. 정상이면 무출력.
#
# 실행 위치: **로컬**(저장소 루트). 자격증명은 ./.env 에서 읽고 B200 API 를 직접 친다.
#   이전 판은 KISTI ssh 를 거쳤는데 세션이 만료되자 감시가 2일간 통째로 멎었다(2026-08-21).
#
# 임계는 1차(붕괴)와 2차(정상)의 **100 step 중앙값 궤적**을 겹쳐 보고 뽑았다.
# 평균은 못 쓴다 — 2차 step 1775 한 개(tr_ppl=119,300)가 200 step 평균을 321 로 만들었다.
#   지표별 분리 가능 여부:
#     tr/ro   1차 1.152~1.545  vs 2차 1.046~1.121   → 분리됨
#     len     1차  482~687     vs 2차  940~2974     → 분리됨
#     fmt     1차 0.946~1.000  vs 2차 0.987~1.000   → 겹침(붕괴 중반에야 내려간다)
#     ess     1차 0.947~0.968  vs 2차 0.953~0.986   → 겹침, 단독 사용 금지
#     clipf   1차 0.0084~0.0148 vs 2차 0.0022~0.0137 → 겹침(발산 깊어지면 오히려 하락)
set -u
cd "$(dirname "$0")/.." || { echo "ALERT cd 실패"; exit 0; }
[ -f .env ] || { echo "ALERT .env 없음"; exit 0; }
set -a; . ./.env; set +a
: "${ORCH_BASE_URL:?}" "${ORCH_PAT:?}"
LOG_NAME=train_deepvision_ep1_gdpo_async_tis_entmask.log
W="${TMPDIR:-/tmp}/kbds_watch"; mkdir -p "$W"

code=$(curl -k -s --max-time 180 "$ORCH_BASE_URL/me/data/file?path=$LOG_NAME" \
       -H "Authorization: Bearer $ORCH_PAT" -o "$W/t.log" -w '%{http_code}')
[ "$code" = 200 ] || { echo "ALERT 로그 다운로드 실패 (HTTP $code)"; exit 0; }
curl -k -s --max-time 60 "$ORCH_BASE_URL/me/data?path=" -H "Authorization: Bearer $ORCH_PAT" -o "$W/root.json"
python3 b200/parse_log.py "$W/t.log" "$W/m.csv" >/dev/null 2>&1 || { echo "ALERT parse 실패"; exit 0; }

python3 - "$W" "$LOG_NAME" <<'PY'
import csv, json, sys, time, statistics as st
W, LOG = sys.argv[1], sys.argv[2]
r=[x for x in csv.DictReader(open(f"{W}/m.csv"))]
if not r: print("ALERT CSV 비어 있음"); sys.exit()
def f(x):
    try: return float(x)
    except: return None
def med(c, n=50):
    v=[f(x[c]) for x in r[-n:] if f(x[c]) is not None]
    return st.median(v) if v else None
def ratio(n=50):
    v=[f(x['tr_ppl'])/f(x['ro_ppl']) for x in r[-n:]
       if f(x['tr_ppl']) and f(x['ro_ppl'])]
    return st.median(v) if v else None
step=int(r[-1]['step'])
rat, ln, fm = ratio(), med('len'), med('fmt')
pd, es, cf  = med('ppl_abs_diff'), med('ess'), med('clipped_frac')

a=[]                                    # 단독 — 1차 붕괴 중반 이후 값, 2차와 완전 분리
if rat and rat > 1.30 : a.append(f"tr/ro={rat:.3f} > 1.30")
if ln  and ln  < 900  : a.append(f"len={ln:.0f} < 900")
if fm  and fm  < 0.97 : a.append(f"fmt={fm:.3f} < 0.97")

w=[]                                    # 경고 — 2개 이상 겹칠 때만
if rat and rat > 1.14  : w.append(f"tr/ro={rat:.3f}")
if pd  and pd  > 0.125 : w.append(f"ppl_abs_diff={pd:.4f}")
if es  and es  < 0.950 : w.append(f"ess={es:.4f}")
if cf  and cf  > 0.015 : w.append(f"clipped_frac={cf:.5f}")
if len(w) >= 2: a.append("경고지표 " + " + ".join(w))

# 학습이 살아 있는가 — 원격 로그 mtime 으로 본다(체인이 다른 호스트라 pgrep 불가).
try:
    ent={e["name"]: e for e in json.load(open(f"{W}/root.json"))["entries"]}
    age=time.time()-ent[LOG]["mtime"]
    if age > 2400: a.append(f"로그가 {int(age/60)}분째 갱신 안 됨")
except Exception as e:
    a.append(f"로그 mtime 확인 실패({type(e).__name__})")

if a: print(f"ALERT step={step} | " + " | ".join(a))
PY
exit 0
