#!/bin/bash
# =============================================================================
# 40_eval.slurm — [4단계] 벤치마크 평가 (의료 멀티모달)
#   계획서 산정: 1노드 × 10h × 8GPU × 8회 = 640 노드시간
#   ms-swift eval (EvalScope 백엔드) 로 SFT 대비 성능 향상 정량 비교.
# =============================================================================
#SBATCH --job-name=kbds-eval
#SBATCH --partition=8gpu          # A100 80GB ×8
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/home01/k252a02/kbds_project/logs/eval_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/eval_%j.log

set -euo pipefail
source /home01/k252a02/kbds_project/scripts/00_common.sh
# 호스트 모듈은 conda 폴백 시에만 필요(컨테이너는 자체 cuda/gcc 보유)
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# 평가 대상 체크포인트 (인자로 단계별 비교: sft / grpo_general / medical_rl)
EVAL_CKPT="${EVAL_CKPT:-$CKPT_DIR/medical_rl}"
# 의료/일반 멀티모달 벤치마크 (실제 사용 가능 dataset 명으로 교체)
EVAL_DATASETS="${EVAL_DATASETS:-MMMU MMStar}"

run_py swift eval \
  --model "$EVAL_CKPT" \
  --model_type "$MODEL_TYPE" \
  --eval_backend VLMEvalKit \
  --eval_dataset $EVAL_DATASETS \
  --infer_backend vllm \
  --max_pixels "$MAX_PIXELS" \
  --eval_output_dir "$WORK_DIR/eval_results/$(basename "$EVAL_CKPT")"

echo "[eval] done -> $WORK_DIR/eval_results/$(basename "$EVAL_CKPT")"
