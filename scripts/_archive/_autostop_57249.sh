#!/bin/bash
# step 1000 도달(checkpoint-1000 저장) 시 job 57249 자동 정지
JOB=57249
CKPT=work/checkpoints/grpo_general/v11-20260616-165537/checkpoint-1000
LOG=logs/grpo_stage2_57249.log
STAMP=logs/_autostop_57249.status
echo "[$(date '+%F %T')] autostop watcher 시작 (target=checkpoint-1000)" > "$STAMP"
while true; do
  # 잡이 이미 끝났으면 종료
  if ! squeue -j "$JOB" -h 2>/dev/null | grep -q "$JOB"; then
    echo "[$(date '+%F %T')] job $JOB 이미 종료됨. watcher 종료." >> "$STAMP"
    exit 0
  fi
  # 현재 step 기록
  step=$(grep -oE "'global_step/max_steps': '[0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  echo "[$(date '+%F %T')] step=${step:-?}" >> "$STAMP"
  # checkpoint-1000 저장되면 정지
  if [ -d "$CKPT" ]; then
    sleep 30   # 저장 완료 대기 여유
    scancel "$JOB"
    echo "[$(date '+%F %T')] checkpoint-1000 감지 → scancel $JOB 실행. watcher 종료." >> "$STAMP"
    exit 0
  fi
  sleep 300
done
