#!/bin/bash
# wipe_b200.sh — B200 플랫폼의 $ORCH_HOME 을 전부 비운다. **되돌릴 수 없다.**
#
# 이 스크립트는 지우기 전에 **KISTI 이관본의 무결성을 스스로 검증하고, 실패하면 삭제를 거부한다.**
#   1) KISTI 측 파일 수가 3,259 이상인가
#   2) 최종 어댑터(checkpoint-2192/adapter_model.safetensors) 의 sha256 이 B200 원본과 같은가
# 둘 중 하나라도 어긋나면 아무것도 지우지 않고 종료한다.
#
# 이관 검증 근거(2026-08-21):
#   runs/ 3,259 파일 85 GiB · sha256 170/170 일치 · 크기 불일치 0
#   kbds_project/work 64G 는 KISTI 에 273,411 파일 전부 존재(누락 0) → 이관 대상이 아니었다
#   kbds_project/ 코드는 GitHub 에 있고 B200 쪽이 오히려 구버전이다(구 계정 절대경로가 박혀 있음)
#   .venv 는 node_setup_and_smoke.sh 로 멱등 재구축 가능(CLAUDE.md §1.4 의 함정 5개 반영본)
#
# 실행: bash b200/wipe_b200.sh --dry     (지우지 않고 대상만 보여준다)
#       bash b200/wipe_b200.sh --yes     (실제로 지운다)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "❌ .env 없음"; exit 1; }
set -a; . ./.env; set +a
: "${ORCH_BASE_URL:?}" "${ORCH_PAT:?}"
AUTH="Authorization: Bearer $ORCH_PAT"
MODE="${1:-}"
[ "$MODE" = "--dry" ] || [ "$MODE" = "--yes" ] || { echo "사용법: $0 --dry | --yes"; exit 1; }

MIG='~/kbds_project/work/b200_migrate'
CK='runs/deepvision_ep1_gdpo_async_tis_entmask/v38-20260821-064935/checkpoint-2192/adapter_model.safetensors'

echo "── 1단계: KISTI 이관본 검증 ──"
N=$(ssh kbds "find $MIG -type f 2>/dev/null | wc -l")
echo "  KISTI 파일 수: $N"
[ "$N" -ge 3259 ] || { echo "❌ 3,259 미만 — 이관이 불완전하다. 삭제하지 않는다."; exit 1; }
KSUM=$(ssh kbds "sha256sum $MIG/$CK 2>/dev/null | cut -d' ' -f1")
BSUM=$(ssh kbds "grep -F '$CK' $MIG/b200_st_sha256.txt 2>/dev/null | cut -d' ' -f1")
echo "  최종 어댑터 sha256  KISTI=${KSUM:0:16}…  B200원본=${BSUM:0:16}…"
[ -n "$KSUM" ] && [ "$KSUM" = "$BSUM" ] || { echo "❌ 체크섬 불일치/누락 — 삭제하지 않는다."; exit 1; }
echo "  ✅ 검증 통과"

# 최상위 19개 항목 전부. 와일드카드를 쓰지 않는다(닷파일이 빠지거나 예상 외 항목이 걸린다).
TARGETS='runs st_stage kbds_project safe_ckpt uploads
         .venv .cache .config .local .modelscope .nv .triton
         b200_st_sha256.txt b200_work_manifest.tsv
         train_deepvision_ep1_gdpo_async_tis_entmask.log train_deepvision_ep1_gdpo_async_tis.log
         rollout_deepvision_ep1_gdpo_async_tis_entmask.log rollout_deepvision_ep1_gdpo_async_tis.log
         watchdog_deepvision.log'

echo "── 2단계: 삭제 대상 ──"
for t in $TARGETS; do echo "  $t"; done
if [ "$MODE" = "--dry" ]; then echo "(--dry: 아무것도 지우지 않았다)"; exit 0; fi

echo "── 3단계: gpu_count 0 세션에서 rm -rf ──"
NODE=$(curl -k -s --max-time 20 "$ORCH_BASE_URL/nodes" -H "$AUTH" \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')
SID=$(curl -k -s --max-time 30 -X POST "$ORCH_BASE_URL/nodes/$NODE/sessions" -H "$AUTH" \
      -H 'Content-Type: application/json' -d '{"gpu_count":0}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "  session=$SID (GPU 0장)"
trap 'curl -k -s -X DELETE "$ORCH_BASE_URL/sessions/$SID" -H "$AUTH" >/dev/null; echo "  세션 반납"' EXIT

CMD="cd \"\$ORCH_HOME\" && rm -rf $(echo $TARGETS | tr '\n' ' ') ; echo '=== 남은 것 ==='; ls -A \"\$ORCH_HOME\" | head -20; echo '=== 용량 ==='; du -sh \"\$ORCH_HOME\" 2>/dev/null; df -h \"\$ORCH_HOME\" | tail -1"
JOB=$(curl -k -s --max-time 30 -X POST "$ORCH_BASE_URL/sessions/$SID/exec" -H "$AUTH" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"command":sys.argv[1],"timeout_sec":1800}))' "$CMD")" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
for _ in $(seq 1 300); do
  R=$(curl -k -s --max-time 20 "$ORCH_BASE_URL/jobs/$JOB" -H "$AUTH")
  ST=$(printf '%s' "$R" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job"]["state"])')
  case "$ST" in
    succeeded|failed|killed)
      echo "  exec=$ST"
      printf '%s' "$R" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("stdout_tail","")); e=(d.get("stderr_tail") or "").strip(); print("STDERR:", e[:400]) if e else None'
      break;;
  esac
  sleep 6
done

echo "── 4단계: 재조회로 삭제 확인 (200 을 증거로 쓰지 않는다) ──"
curl -k -s --max-time 30 "$ORCH_BASE_URL/me/data?path=" -H "$AUTH" \
 | python3 -c 'import json,sys; e=json.load(sys.stdin)["entries"]; print(f"  남은 항목 {len(e)}개:", " ".join(sorted(x["name"] for x in e)) or "(없음)")'
