#!/bin/bash
# =============================================================================
# launch_stage2_epoch.sh — dr_grpo Stage-2 를 1 epoch 까지 resume 체인 학습.
#   현재 step~650(TIMEOUT 중단) → MAX_STEPS(=1 epoch) 까지 이어학습.
#   1 epoch = 103,503 / 32(step당 프롬프트) ≈ 3,235 step. 시간제한(70h≈680 step)을 넘으므로
#   afterany 의존 체인으로 N개 잡 제출: 각 잡이 최신 checkpoint 에서 resume,
#   MAX_STEPS 도달 시 이후 잡들은 자동 no-op(빠른 종료).
# 사용:  bash scripts/launch_stage2_epoch.sh        # 기본 1 epoch
#        MAX_STEPS=2000 N_JOBS=3 bash scripts/launch_stage2_epoch.sh
# =============================================================================
set -euo pipefail
cd /home01/k252a02/kbds_project
MAX_STEPS="${MAX_STEPS:-3235}"
N_JOBS="${N_JOBS:-5}"

echo "[chain] dr_grpo 이어학습 → MAX_STEPS=$MAX_STEPS, 체인 $N_JOBS 잡 (afterany)"
PREV=""
for i in $(seq 1 "$N_JOBS"); do
  DEP=""; [[ -n "$PREV" ]] && DEP="--dependency=afterany:$PREV"
  JID=$(sbatch --parsable $DEP \
        --export=ALL,RECIPE=dr_grpo,RESUME=1,MAX_STEPS="$MAX_STEPS" \
        scripts/21_rlvr_grpo_adv.slurm)
  echo "[chain] job $i = $JID (dep=${PREV:-none})"
  PREV="$JID"
done
echo "[chain] 제출 완료."
echo "[chain] 모니터: squeue -u \$USER ;  최신 step: ls -d work/checkpoints/grpo_general_adv_dr_grpo/*/checkpoint-* | sort -t- -k2 -n | tail"
echo "[chain] 완료 후: 최종 checkpoint 재병합(merge_drgrpo.slurm ADAPTER=<최종ckpt>) → Stage-3 init 갱신"
