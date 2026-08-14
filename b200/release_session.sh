#!/usr/bin/env bash
# release_session.sh — 반납되지 않은 세션을 조회/해제한다.
#
#   ORCH_BASE_URL=https://... ORCH_PAT=pat_xxx bash release_session.sh            # 목록만
#   ORCH_BASE_URL=... ORCH_PAT=... bash release_session.sh <session_id>           # 하나 반납
#   ORCH_BASE_URL=... ORCH_PAT=... bash release_session.sh --all-gpu              # GPU 8장짜리 전부
#
# 왜 필요한가: drive_node.sh 는 폴링이 끝난 뒤 DELETE 로 세션을 반납한다. 그런데 폴링 프로세스가
# 죽으면(체인 스크립트를 kill 하거나 ssh 가 끊기면) DELETE 가 실행되지 않아 **세션이 GPU 를 계속
# 붙든 채로 남는다.** 그 뒤 job 은 전부 `session create failed` 로 죽는다. 실제로 겪었다.
#
# ⚠️ 엔드포인트: `/sessions` 다. `/nodes/<id>/sessions` 는 Method Not Allowed 를 돌려준다.
# ⚠️ 목록에는 이미 반납된 세션도 이력으로 남는다 — state 로 걸러야 "지금 잡고 있는 것"이 나온다.
set -u
BASE="${ORCH_BASE_URL:?set ORCH_BASE_URL to the platform API base (do not hardcode/commit it)}"
PAT="${ORCH_PAT:?set ORCH_PAT to your platform PAT (do not hardcode/commit it)}"
AUTH="Authorization: Bearer $PAT"
CURL="curl -sk --max-time 60"

list(){ $CURL "$BASE/sessions" -H "$AUTH" | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: print('  (응답 파싱 실패)'); sys.exit(0)
rows = d if isinstance(d,list) else d.get('sessions', d.get('items', []))
only_gpu = '${1:-}' == 'gpu'
for s in rows:
    gi = s.get('gpu_indices') or []
    st = (s.get('state') or '').lower()
    if only_gpu and (len(gi) < 8 or st in ('released','closed','terminated')): continue
    print(s.get('id'), st or '-', gi, (s.get('created_at') or '')[:19])
"; }

case "${1:-}" in
  "")         echo "=== 세션 목록 (이력 포함) ==="; list ;;
  --all-gpu)  echo "=== GPU 8장 점유 세션 반납 ==="
              list gpu | awk '{print $1}' | while read -r id; do
                [ -n "$id" ] || continue
                printf '  %s -> ' "$id"; $CURL -X DELETE "$BASE/sessions/$id" -H "$AUTH"; echo
              done ;;
  *)          echo "반납: $1"; $CURL -X DELETE "$BASE/sessions/$1" -H "$AUTH"; echo ;;
esac
