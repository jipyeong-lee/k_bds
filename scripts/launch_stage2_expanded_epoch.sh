#!/bin/bash
# =============================================================================
# launch_stage2_expanded_epoch.sh — Stage-2 풀확장(GDPO)을 1 epoch 까지 resume 체인 학습
# -----------------------------------------------------------------------------
#  launch_stage2_expanded.sh 의 에포크 확장판. 검증된 21_rlvr_grpo_adv.slurm 을 제출하되
#  RESUME=1 + MAX_STEPS 로 여러 잡을 afterany 체인으로 이어 붙여 walltime 한계를 넘긴다.
#
#  1 epoch = 74,787 / 32(step당 프롬프트: 1 × accum4 × 8gpu) ≈ 2,337 step
#            ≈ 215h wall (331 s/it 실측, job 72832) ≈ 1,719 노드시간 (예산 5,000 의 34%)
#  잡당 walltime 70h ≈ 760 step → 4 잡이면 충분(280h 용량). MAX_STEPS 도달 후 잡은 자동 no-op.
#
#  ⚠️ LR 스케줄은 MAX_STEPS 기준으로 감쇠한다. 목표 스텝을 처음부터 지정해야 매끄럽다
#     (600 으로 돌린 뒤 2337 로 늘리면 LR 이 0 까지 갔다가 다시 튀어 불연속).
#
#  ⚠️ 학습량을 늘리는 근거와 한계:
#     - 구 데이터셋(DeepVision 단독)에선 홀드아웃이 step600 에서 포화(0.38~0.39, step800 동일).
#     - 그러나 확장셋은 MMK12(math)·PMC-VQA(의료)가 새로 들어가 구성이 다르므로 포화점이
#       달라질 수 있다 → 에포크 확장의 근거.
#     - 외부 문헌은 RLVR 장기학습에서 diversity collapse 경고(arXiv 2606.15455): 후반 구간은
#       Pass@1 이득 없이 high-k Pass@k 를 깎는다. → **반드시 중간 체크포인트를 평가하고
#       포화하면 조기중단**할 것. save_steps 50 이라 체크포인트는 촘촘히 남는다.
#     상세 = docs/rlvr_hparams_external.md
#
#  사용:  bash scripts/launch_stage2_expanded_epoch.sh              # 1 epoch(2337) 체인 4잡
#         MAX_STEPS=1200 N_JOBS=2 bash scripts/launch_stage2_expanded_epoch.sh
#  중단:  scancel <체인 잡 ID들>  (남은 체인 잡도 함께 취소해야 이어학습이 멈춘다)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/00_common.sh

INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/stage2_expanded_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/grpo_expanded_gdpo}"
MAX_STEPS="${MAX_STEPS:-2337}"        # 1 epoch
N_JOBS="${N_JOBS:-4}"                 # 70h × 4 = 280h ≥ 215h 필요분

[ -f "$INIT_MODEL/config.json" ] || { echo "❌ INIT 병합본 없음: $INIT_MODEL"; exit 1; }
[ -f "$DATASET_FILE" ] || { echo "❌ 확장셋 없음: $DATASET_FILE"; exit 1; }

echo "[chain] GDPO 확장셋 이어학습 → MAX_STEPS=$MAX_STEPS (1 epoch≈2337), 체인 $N_JOBS 잡"
echo "[chain] INIT=$INIT_MODEL"
echo "[chain] DATA=$DATASET_FILE ($(wc -l <"$DATASET_FILE") 건)"
echo "[chain] OUT =$OUTPUT_DIR"
echo "[chain] knobs: NUM_GEN=${NUM_GEN:-4}  TEMPERATURE=${TEMPERATURE:-0.9}  BETA=${BETA:-0.04} (검증값)"

PREV=""
for i in $(seq 1 "$N_JOBS"); do
  DEP=""; [[ -n "$PREV" ]] && DEP="--dependency=afterany:$PREV"
  JID=$(sbatch --parsable $DEP \
        --export=ALL,RECIPE=dr_grpo,SCALE_REWARDS=gdpo,RESUME=1,MAX_STEPS="$MAX_STEPS",INIT_MODEL="$INIT_MODEL",DATASET_FILE="$DATASET_FILE",OUTPUT_DIR="$OUTPUT_DIR" \
        scripts/21_rlvr_grpo_adv.slurm)
  echo "[chain] job $i = $JID (dep=${PREV:-none})"
  PREV="$JID"
done

echo "[chain] 제출 완료."
echo "[chain] 모니터: squeue -u \$USER ; ls -d $OUTPUT_DIR/*/checkpoint-* | sort -t- -k2 -n | tail"
echo "[chain] ⚠️ 중간 체크포인트를 홀드아웃(_source 별)으로 평가하고 포화 시 조기중단할 것."
