#!/usr/bin/env bash
# run_pod.sh — 로컬에서 kgpu pod 1개를 빌려 ~/files/kbds/<script> 를 돌리고, 끝나면(또는 상한 시간) 반납한다.
# 사용: setsid nohup bash kgpu/run_pod.sh <script> [KEY=VAL ...] > out 2>&1 < /dev/null & disown
#   KEY=VAL 은 pod env 로 넘어간다(비밀값은 이름만 적으면 로컬 env 값을 쓴다: JUDGE_API_KEY HF_TOKEN).
# env(선택): GPU_MODEL(rtxpro6000) GPU_COUNT(1) MEM(60Gi) CPU(8) DISK(200) MAX_SEC(36000) POLL(180) GRACE(600)
# 반납 이중화: 로컬 trap + pod 자기 반납(스크립트 끝 → .done → GRACE 초 뒤 in-pod 토큰으로 DELETE).
#   로컬 PC 가 잠들어 trap 이 안 돌아도 GRACE 만큼만 과금된다. curl 은 setup_venv.sh 가 깐다.
# 에이전트 세션 백그라운드로 띄우면 세션 종료 때 trap 이 pod 를 반납한다 → 반드시 setsid nohup 으로 분리(CLAUDE.md §2.5).
set -uo pipefail
SCRIPT=${1:?script name under ~/files/kbds}; shift
cd "$(dirname "$0")/.." && set -a && . ./.env && set +a
B=https://api.kgpu.net/v1
api(){ curl -sS --config <(printf 'header = "Authorization: Bearer %s"\n' "$KGPU_API_TOKEN") "$@"; }
ENV_JSON=$(python3 -c '
import json,os,sys
e={}
for a in sys.argv[1:]:
    k,_,v=a.partition("=")
    e[k]=v if "=" in a else os.environ[k]
print(json.dumps(e))' "$@") || exit 1
BODY=$(python3 -c 'import json,os,sys;print(json.dumps({"name":"kbds-"+sys.argv[1].split(".")[0].replace("_","-")[:20],
 "image":"docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04","gpu_model":os.environ.get("GPU_MODEL","rtxpro6000"),
 "gpu_count":int(os.environ.get("GPU_COUNT",1)),"cpu":os.environ.get("CPU","8"),"memory":os.environ.get("MEM","60Gi"),
 "ephemeral_storage_gib":int(os.environ.get("DISK",200)),"env":json.loads(sys.argv[2])}))' "$SCRIPT" "$ENV_JSON")
# pod_id 가 있으면 성공(접두사는 name 에 따라 바뀐다). 없으면 'ERR <응답>'
POD=$(printf '%s' "$BODY" | api -X POST $B/pods -H 'Content-Type: application/json' -d @- | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("pod_id") or "ERR %s" % d)')
case $POD in ERR*|'') echo "❌ pod create: $POD"; exit 1;; esac
trap 'api -X DELETE $B/pods/$POD >/dev/null; echo "[$(date +%T)] released $POD"' EXIT
echo "[$(date +%T)] pod $POD"
ex(){ python3 -c 'import json,sys;print(json.dumps({"cmd":["bash","-c",sys.argv[1]],"timeout_seconds":120}))' "$1" |
      api -X POST $B/pods/$POD/exec -H 'Content-Type: application/json' -d @- | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("stdout",d),end="")'; }

for i in $(seq 60); do
  [ "$(api $B/pods/$POD | python3 -c 'import json,sys;print(json.load(sys.stdin).get("ready"))')" = True ] && break; sleep 10
done
# ~/files 에서 바로 실행하면 Errno 121 → pod 로컬로 복사해서 실행
ex "mkdir -p /root/w && cp /root/files/kbds/$SCRIPT /root/w/" >/dev/null
RUN='bash /root/w/'"$SCRIPT"'; touch /root/w/.done; sleep '"${GRACE:-600}"'; curl -sS --retry 5 --retry-all-errors -X DELETE -H "Authorization: Bearer $KGPU_API_TOKEN" "$KGPU_API_BASE/pods/$KGPU_RENT_ID"'
PID=$(python3 -c 'import json,sys;print(json.dumps({"cmd":["bash","-c",sys.argv[1]],"background":True}))' "$RUN" |
      api -X POST $B/pods/$POD/exec -H 'Content-Type: application/json' -d @- | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("pid") or d)')
case $PID in ''|*[!0-9]*) echo "❌ launch: $PID"; exit 1;; esac
echo "[$(date +%T)] pid $PID"

T0=$(date +%s)
while :; do
  sleep "${POLL:-180}"
  # pod 가 밖에서 끝났으면(power_limit·evicted·크레딧·자기 반납) exec 는 계속 실패한다 → end_reason 으로 바로 끝낸다
  END=$(api $B/pods/$POD | python3 -c 'import json,sys;print(json.load(sys.stdin).get("end_reason") or "")' 2>/dev/null)
  [ -n "$END" ] && { echo "[$(date +%T)] pod 종료됨: end_reason=$END (pod 로컬 로그는 사라짐, ~/files/kbds/logs 확인)"; exit 0; }
  echo "[$(date +%T)] $(ex "tail -n 3 /root/w/*.log 2>/dev/null | grep -v '^\s*\$' | tail -n 1 | cut -c1-200")"
  # 빈 출력(일시적 API 오류)은 종료로 보지 않는다 — 명시적 DEAD 만 종료
  # kill -0 은 좀비도 살았다고 본다(23 함정) → /proc 상태로 판정. 래퍼는 GRACE 동안 살아 있으니 .done 도 종료로 본다
  [ "$(ex "[ ! -e /root/w/.done ] && grep -qs '^State:.*[^Z] (' /proc/$PID/status && echo ALIVE || echo DEAD")" = DEAD ] && break
  (( $(date +%s) - T0 > ${MAX_SEC:-36000} )) && { echo "⏰ MAX_SEC 초과"; break; }
done
echo "[$(date +%T)] 종료. 로그 마지막 40줄:"; ex "tail -n 40 /root/w/*.log"
