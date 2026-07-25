#!/bin/bash
# =============================================================================
# watch_train.sh — 학습 라이브 모니터 (tmux용). 30초마다 squeue + step 지표 + 로그꼬리 갱신.
#   사용:  bash scripts/watch_train.sh [logfile]
#     logfile 생략 시 logs/grpo_stage2_*.log 중 최신을 매 갱신마다 자동 추적(잡 재시작 대응).
#   tmux:  tmux attach -t kbds   /   detach: Ctrl-b 누른 뒤 d
# =============================================================================
PROJ=/home01/k252a02/kbds_project
USER_ID=k252a01
FIXED_LOG="${1:-}"

while true; do
  LOG="${FIXED_LOG:-$(ls -t $PROJ/logs/grpo_stage2_*.log 2>/dev/null | head -1)}"
  clear
  echo "==================== K-BDS 학습 모니터  $(date '+%F %T') ===================="
  echo "LOG: $LOG"
  echo "------------------------------ squeue -------------------------------"
  squeue -u "$USER_ID" -o "%.8i %.14j %.5T %.11M %.4D %R" 2>/dev/null
  echo "--------------------------- 최근 step 지표 --------------------------"
  python3 - "$LOG" <<'PY' 2>/dev/null
import re, sys
L = sys.argv[1]
try:
    lines = [l for l in open(L) if "'global_step/max_steps'" in l and "'reward'" in l]
except Exception:
    lines = []
if not lines:
    print("  (아직 step 지표 없음 — 초기화/로딩 중)")
for l in lines[-10:]:
    m = re.search(r"\{'loss'.*?\}", l)
    if m:
        try:
            d = eval(m.group(0))
            print(f"  step {d['global_step/max_steps']:>11} | reward {d['reward']:>7} | "
                  f"Acc {d['rewards/AccuracyMix/mean']:>7} | Fmt {d['rewards/Format/mean']:>6} | "
                  f"clip {d['completions/clipped_ratio']:>6} | mem {d['memory(GiB)']:>6} | s/it {d.get('train_speed(s/it)','?')}")
        except Exception:
            pass
PY
  echo "----------------------------- 로그 끝 6줄 ---------------------------"
  tail -6 "$LOG" 2>/dev/null | grep -vE "Memory cleanup|fla/ops|return fn|UserWarning" | cut -c1-100 | tail -4
  echo ""
  echo "(30초 갱신 · detach: Ctrl-b 뒤 d · 종료: Ctrl-c)"
  sleep 30
done
