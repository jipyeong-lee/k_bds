#!/bin/bash
# cleanup_runs.sh — $ORCH_HOME/runs 의 죽은 실행 디렉터리를 지운다.
#
# 왜 스크립트인가: `DELETE /me/data` 는 runs/ 아래에서 거짓 200 을 주고 아무것도 안 지운다(CLAUDE.md §1.5).
#   실제 삭제는 세션 exec 의 rm -rf 로만 된다.
# 왜 지금 되는가: **gpu_count 0 세션은 학습이 8장을 잡고 있어도 열린다**(2026-08-21 실측).
#   "arm 교체 틈에만 가능"이라던 이전 판단은 틀렸다 — GPU 를 요구하는 세션만 막힌다.
#
# 실행: bash b200/cleanup_runs.sh          (자격증명은 저장소 루트 .env)
#       bash b200/cleanup_runs.sh --dry    (지우지 않고 대상만 보여준다)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "❌ .env 없음"; exit 1; }
set -a; . ./.env; set +a
: "${ORCH_BASE_URL:?}" "${ORCH_PAT:?}"
AUTH="Authorization: Bearer $ORCH_PAT"
DRY=""; [ "${1:-}" = "--dry" ] && DRY=1

# 지울 것 — 전부 지난 실험의 잔해다. 와일드카드를 쓰지 않는다(활성 디렉터리와 접두사가 겹친다).
DEAD=(ckpt.txt ckpt2.txt exp1 bench_p4a4 bench_p8a2 bench_p2a8
      deepvision_ep1_gdpo deepvision_smoke pmcvqa_smoke deepvision_server7_smoke
      deepvision_ep1_gdpo_async deepvision_ep1)
# 남길 것 (참고): deepvision_ep1_gdpo_async_tis      24G — 1차 붕괴 어댑터, 재생성 불가
#                deepvision_ep1_gdpo_async_tis_entmask 53G — 실행 중

NODE=$(curl -k -s --max-time 20 "$ORCH_BASE_URL/nodes" -H "$AUTH" \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')

if [ -n "$DRY" ]; then CMD='du -sh "$ORCH_HOME/runs"/* 2>/dev/null | sort -h'
else CMD="cd \"\$ORCH_HOME/runs\" && rm -rf ${DEAD[*]}; echo '=== 남은 것 ==='; du -sh \"\$ORCH_HOME/runs\"/* 2>/dev/null | sort -h; df -h \"\$ORCH_HOME\" | tail -1"
fi

SID=$(curl -k -s --max-time 30 -X POST "$ORCH_BASE_URL/nodes/$NODE/sessions" -H "$AUTH" \
      -H 'Content-Type: application/json' -d '{"gpu_count":0}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "session=$SID (GPU 0장 — 학습에 영향 없음)"
trap 'curl -k -s -X DELETE "$ORCH_BASE_URL/sessions/$SID" -H "$AUTH" >/dev/null; echo "session 반납됨"' EXIT

JOB=$(curl -k -s --max-time 30 -X POST "$ORCH_BASE_URL/sessions/$SID/exec" -H "$AUTH" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"command":sys.argv[1],"timeout_sec":600}))' "$CMD")" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

for _ in $(seq 1 120); do
  R=$(curl -k -s --max-time 20 "$ORCH_BASE_URL/jobs/$JOB" -H "$AUTH")
  ST=$(printf '%s' "$R" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job"]["state"])')
  case "$ST" in
    succeeded|failed|killed)
      echo "state=$ST"
      printf '%s' "$R" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("stdout_tail","")); e=(d.get("stderr_tail") or "").strip(); print("STDERR:", e[:400]) if e else None'
      exit 0;;
  esac
  sleep 5
done
echo "❌ 폴링 시간 초과 — job=$JOB 를 직접 확인할 것"
