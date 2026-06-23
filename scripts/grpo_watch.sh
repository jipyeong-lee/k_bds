#!/bin/bash
# =============================================================================
# grpo_watch.sh — DAPO 본실행(57527) 모니터링 watcher.
#   새 checkpoint-N00(100의 배수)이 저장되거나 잡이 종료되면:
#     1) plot 재생성 (plot_grpo_compare.py, 컨테이너 내)
#     2) README AUTO 마커 블록 갱신 (grpo_ab_update.py, 호스트 python3)
#     3) 변경분 git commit + push origin HEAD:master
#   tmux 세션에서 상주 실행 → 클로드 세션이 끊겨도 계속 동작.
#
# 사용:  tmux new -s grpo_watch -d 'bash scripts/grpo_watch.sh'
#        tmux attach -t grpo_watch        # 로그 확인
# 상태/로그: logs/grpo_watch.log
# =============================================================================
set -u
cd /home01/k252a01/kbds_project

JOB_ID="${JOB_ID:-57527}"
DAPO_DIR="${DAPO_DIR:-work/checkpoints/grpo_general_adv_dapo/v1-20260622-154040}"
BASE_LOG="${BASE_LOG:-logs/grpo_stage2_57249.log}"
DAPO_LOG="${DAPO_LOG:-logs/grpo_adv_57527.log}"
PLOT="${PLOT:-docs/assets/grpo_dapo_vs_baseline.png}"
IMG="${IMG:-work/images/ms-swift-413-sandbox}"
POLL="${POLL:-600}"                       # 폴링 간격(초). 369s/it → 100step≈10h, 10분 폴링이면 충분
STATE="logs/grpo_watch.state"             # 마지막으로 처리한 100-step 경계 기록
LOG="logs/grpo_watch.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 마지막 100-step 경계 체크포인트 번호 (없으면 0)
last_ckpt() {
  ls -1d "$DAPO_DIR"/checkpoint-* 2>/dev/null \
    | sed -E 's/.*checkpoint-//' | grep -E '^[0-9]+$' \
    | awk '$1%100==0' | sort -n | tail -1
}

job_alive() { squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING|COMPLETING'; }

do_update() {
  local tag="$1"   # 커밋 메시지용 라벨 (예: step 300 / final)
  log "update 시작 ($tag) — plot 재생성"
  singularity exec "$IMG" python scripts/plot_grpo_compare.py \
    "$BASE_LOG" baseline "$DAPO_LOG" DAPO "$PLOT" >>"$LOG" 2>&1 \
    || log "WARN plot 재생성 실패(계속 진행)"
  local out
  out=$(python3 scripts/grpo_ab_update.py 2>>"$LOG")
  log "updater: $out"
  git add README.md "$PLOT" >>"$LOG" 2>&1
  if git diff --cached --quiet; then
    log "변경 없음 — 커밋 생략"
    return
  fi
  git commit -m "docs: DAPO 57527 진행 자동갱신 ($tag)" >>"$LOG" 2>&1
  if git push origin HEAD:master >>"$LOG" 2>&1; then
    log "push 완료 ($tag)"
  else
    log "WARN push 실패 — 다음 주기에 재시도"
  fi
}

mkdir -p logs
PREV=$(cat "$STATE" 2>/dev/null || echo 0)
log "watcher 시작: JOB=$JOB_ID DAPO_DIR=$DAPO_DIR PREV_CKPT=$PREV POLL=${POLL}s"

while true; do
  CUR=$(last_ckpt); CUR=${CUR:-0}
  if [[ "$CUR" -gt "$PREV" ]]; then
    log "신규 100-step 체크포인트 감지: checkpoint-$CUR (이전 $PREV)"
    do_update "step $CUR"
    PREV="$CUR"; echo "$PREV" > "$STATE"
  fi

  if ! job_alive; then
    log "잡 $JOB_ID 종료 감지 — 최종 갱신 후 watcher 종료"
    do_update "final (job 종료)"
    log "watcher 정상 종료"
    exit 0
  fi
  sleep "$POLL"
done
