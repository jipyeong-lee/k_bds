#!/bin/bash
# run_demo.sh — 로컬에서 실행. 플랫폼 API 로 8장 세션을 열고 학습을 돌려 결과를 받아온다.
#
# 플랫폼 사용 흐름은 항상 이 5단계다:
#   ① 코드 업로드   POST /me/data/upload
#   ② 세션 생성     POST /nodes/<id>/sessions   {"gpu_count":8}
#   ③ 명령 실행     POST /sessions/<sid>/exec   → job_id 즉시 반환(비동기)
#   ④ 폴링          GET  /jobs/<job_id>         → running|succeeded|failed|killed
#   ⑤ 세션 반납     DELETE /sessions/<sid>      ← 안 하면 GPU 를 계속 점유한다
#
# ⚠️ ⑤가 이 스크립트의 핵심이다. 폴링 도중 스크립트가 죽으면 DELETE 가 실행되지 않아
#    세션이 GPU 8장을 계속 잡고, 이후 모든 세션 생성이 실패한다. 그래서 trap 을 건다.
# ⚠️ job 은 timeout_sec 에 무조건 killed 되고, 그때 stdout_tail 이 통째로 빈다.
#    그래서 결과는 화면이 아니라 $ORCH_HOME 의 파일로 받는다(⑥).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] || { echo "❌ .env 없음 (.env.example 참고)"; exit 1; }
set -a; . ./.env; set +a
: "${ORCH_BASE_URL:?}" "${ORCH_PAT:?}"
AUTH="Authorization: Bearer $ORCH_PAT"
CURL=(curl -k -s)          # -k: 플랫폼이 self-signed 인증서를 쓴다
TIMEOUT="${TIMEOUT:-3600}"

api() { "${CURL[@]}" --max-time 60 -H "$AUTH" "$@"; }

echo "── ① 코드 업로드 ──"
for f in lora_demo.py payload.sh; do
  #  path 는 폼 필드가 아니라 **쿼리 파라미터**이고 값은 대상 디렉터리다(파일명은 업로드에서 온다).
  api -X POST "$ORCH_BASE_URL/me/data/upload?path=demo" -F "file=@b200/demo/$f" >/dev/null
  echo "   demo/$f"
done

echo "── ② 세션 생성 (GPU 8장) ──"
NODE=$(api "$ORCH_BASE_URL/nodes" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')
RESP=$(api -X POST "$ORCH_BASE_URL/nodes/$NODE/sessions" -H 'Content-Type: application/json' \
         -d '{"gpu_count":8}')
SID=$(printf '%s' "$RESP" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("id",""))
except Exception: print("")')
if [ -z "$SID" ]; then
  #  응답을 그대로 보여준다. 삼키면 KeyError 만 남아 원인을 알 수 없다.
  echo "❌ 세션 생성 실패: $RESP"
  echo "   409 conflict 면 GPU 가 이미 점유된 것이다 — 고아 세션을 확인·반납할 것:"
  echo "     bash b200/release_session.sh            (조회)"
  echo "     bash b200/release_session.sh --all-gpu  (강제 반납)"
  exit 1
fi
echo "   session=$SID"
#  EXIT 만으로는 부족하다 — Ctrl-C·timeout·파이프 끊김(`| head`)으로 죽으면 DELETE 가 실행되지 않아
#  세션이 GPU 8장을 계속 점유하고, 이후 모든 세션 생성이 409 로 실패한다. 실제로 한 번 겪었다.
trap 'echo "── ⑤ 세션 반납 ──"; api -X DELETE "$ORCH_BASE_URL/sessions/$SID" >/dev/null; echo "   반납 완료"' EXIT INT TERM

echo "── ③ 명령 실행 ──"
JOB=$(api -X POST "$ORCH_BASE_URL/sessions/$SID/exec" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({
          "command": "mkdir -p \"$ORCH_HOME/demo_run\"; bash \"$ORCH_HOME/demo/payload.sh\" > \"$ORCH_HOME/demo_run/demo.log\" 2>&1; echo EXIT=$? >> \"$ORCH_HOME/demo_run/demo.log\"",
          "timeout_sec": int(sys.argv[1])}))' "$TIMEOUT")" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "   job=$JOB (timeout ${TIMEOUT}s)"

echo "── ④ 폴링 ──"
for _ in $(seq 1 2000); do
  ST=$(api "$ORCH_BASE_URL/jobs/$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job"]["state"])')
  case "$ST" in
    succeeded|failed|killed) echo "   state=$ST"; break;;
  esac
  printf '.'; sleep 10
done
echo

echo "── ⑥ 결과 회수 (파일 API — 세션이 없어도 읽힌다) ──"
mkdir -p b200/demo/out
api --max-time 120 "$ORCH_BASE_URL/me/data/file?path=demo_run/demo.log" -o b200/demo/out/demo.log || true
[ -s b200/demo/out/demo.log ] && tail -20 b200/demo/out/demo.log || echo "   (로그 없음)"
api "$ORCH_BASE_URL/me/data?path=demo_run%2Flora_demo_out" | python3 -c '
import json, sys
try:
    for e in json.load(sys.stdin)["entries"]:
        print("   어댑터 %-28s %10d B" % (e["name"], e["size"]))
except Exception:
    print("   (어댑터 없음 — demo.log 를 볼 것)")'
