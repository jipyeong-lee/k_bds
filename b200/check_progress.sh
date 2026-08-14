#!/usr/bin/env bash
# check_progress.sh — 학습 job 이 도는 중에 짧은 job 으로 진행 상황만 훔쳐본다.
# 학습 job 이 killed 되면 stdout 을 잃으므로, 진행 확인은 반드시 이렇게 별도로 해야 한다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
ARM="${ARM:-deepvision}"
SCALE="${SCALE:-gdpo}"
ASYNC="${ASYNC:-true}"
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)"
OUT="$ORCH_HOME/runs/$RUNTAG"
LOG="$ORCH_HOME/train_${RUNTAG}.log"

echo "=== $ARM 진행 ($(date +%T)) ==="
echo "--- 체크포인트 ---"
ls -d "$OUT"/*/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -5 | tr '\n' ' ' || echo "(아직 없음)"
echo
du -sh "$OUT" 2>/dev/null || echo "(output_dir 없음)"

echo "--- 최근 step (필드만) ---"
# 한 줄이 수천 자라 통째로 흘리면 stdout_tail 을 잡아먹는다 → 필요한 필드만 뽑는다.
grep -oE "'(global_step/max_steps|step_time|reward|rewards/AccuracyMix/mean|rewards/FormatThink/mean|completions/mean_length|completions/clipped_ratio|memory\(GiB\)|learning_rate|elapsed_time|remaining_time)': '[^']*'" "$LOG" 2>/dev/null \
  | tail -22 | tr '\n' ' ' | sed "s/'//g"
echo
echo "--- 학습 프로세스 ---"
pgrep -fa "swift rlhf" >/dev/null 2>&1 && echo "  실행 중 ($(pgrep -fc 'swift rlhf') proc)" || echo "  없음 (job 종료됨)"
pgrep -fa "swift rollout" >/dev/null 2>&1 && echo "  rollout 실행 중" || echo "  rollout 없음"
echo "--- 붕괴 감시 판정 ---"
[ -f "$ORCH_HOME/verdict_${RUNTAG}.json" ] && cat "$ORCH_HOME/verdict_${RUNTAG}.json" || echo "  (verdict 없음 — 아직 판정 전이거나 watchdog 미기동)"
echo "--- 학습·추론 확률 불일치(off-policy) ---"
grep -oE "'(rollout_offpolicy[^']*|kl)': '[^']*'" "$LOG" 2>/dev/null | tail -6 | tr '\n' ' ' | sed "s/'//g" || echo "  (계측 없음)"
echo
echo "--- 에러 ---"
grep -iE "out of memory|traceback|❌" "$LOG" 2>/dev/null | tail -3 || echo "  (없음)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
