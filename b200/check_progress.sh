#!/usr/bin/env bash
# check_progress.sh — 학습 job 이 도는 중에 짧은 job 으로 진행 상황만 훔쳐본다.
# 학습 job 이 killed 되면 stdout 을 잃으므로, 진행 확인은 반드시 이렇게 별도로 해야 한다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
ARM="${ARM:-deepvision}"
SCALE="${SCALE:-gdpo}"
ASYNC="${ASYNC:-true}"
# run_epoch.sh 와 같은 규칙으로 RUNTAG 를 만들어야 로그를 찾는다 — 어긋나면 조용히 빈 결과가 나온다.
ISMODE="${ISMODE:-token_truncate}"
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)$([ -n "$ISMODE" ] && echo _tis)"
OUT="$ORCH_HOME/runs/$RUNTAG"
LOG="$ORCH_HOME/train_${RUNTAG}.log"

echo "=== $ARM 진행 ($(date +%T)) ==="
echo "--- 체크포인트 ---"
ls -d "$OUT"/*/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -5 | tr '\n' ' ' || echo "(아직 없음)"
echo
du -sh "$OUT" 2>/dev/null || echo "(output_dir 없음)"

echo "--- 최근 step (필드만) ---"
# 한 줄이 수천 자라 통째로 흘리면 stdout_tail 을 잡아먹는다 → 필요한 필드만 뽑는다.
grep -oE "'(global_step/max_steps|step_time|reward|rewards/AccuracyMix/mean|rewards/FormatThink/mean|completions/mean_length|completions/clipped_ratio|clip_ratio/region_mean|clip_ratio/low_mean|clip_ratio/high_mean|memory\(GiB\)|learning_rate|elapsed_time|remaining_time)': '[^']*'" "$LOG" 2>/dev/null \
  | tail -22 | tr '\n' ' ' | sed "s/'//g"
echo
echo "--- 학습 프로세스 ---"
pgrep -fa "swift rlhf" >/dev/null 2>&1 && echo "  실행 중 ($(pgrep -fc 'swift rlhf') proc)" || echo "  없음 (job 종료됨)"
pgrep -fa "swift rollout" >/dev/null 2>&1 && echo "  rollout 실행 중" || echo "  rollout 없음"
echo "--- 붕괴 감시 판정 ---"
[ -f "$ORCH_HOME/verdict_${RUNTAG}.json" ] && cat "$ORCH_HOME/verdict_${RUNTAG}.json" || echo "  (verdict 없음 — 아직 판정 전이거나 watchdog 미기동)"
echo "--- 학습·추론 확률 불일치(off-policy) ---"
# 실제 로그 키는 rollout_correction/* 다(rollout_offpolicy 가 아니다 — 처음에 이걸 틀려 빈 결과를 봤다).
# IS 보정을 켜면 is_weight/clipped_frac/ess 가 함께 나온다 → 보정이 실제로 걸렸는지의 증거.
grep -oE "'(rollout_correction/[a-z0-9_]*|is_weight[a-z_]*|clipped_frac|ess|kl)': '[^']*'" "$LOG" 2>/dev/null | tail -9 | tr '\n' ' ' | sed "s/'//g" || echo "  (계측 없음)"
echo
echo "--- 에러 ---"
grep -iE "out of memory|traceback|❌" "$LOG" 2>/dev/null | tail -3 || echo "  (없음)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
