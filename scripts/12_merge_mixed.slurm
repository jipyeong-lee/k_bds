#!/bin/bash
# =============================================================================
# 12_merge_mixed.slurm — 콜드스타트 v3(sft_mixed) LoRA 병합 + 형식/길이 프로브
#   1) sft_mixed_lora/checkpoint-* → base 병합 → sft_mixed_merged (Stage-2 init 후보)
#   2) 홀드아웃 6샘플 생성 → format_think·길이 즉시 확인
#      (v2 비교: 학습데이터 format_think 0.473, 모델출력 0.333/n6. v3 목표: ↑)
# =============================================================================
#SBATCH --job-name=mixed-merge
#SBATCH --partition=debug-1gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/merge_mixed_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/merge_mixed_%j.log

set -uo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh
export NPROC_PER_NODE=1
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

ADAPTER=$(ls -td "$CKPT_DIR"/sft_mixed_lora/*/checkpoint-* | head -1)
MERGED="$CKPT_DIR/sft_mixed_merged"
echo "[merge] adapter=$ADAPTER -> $MERGED"

run_py swift export \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --output_dir "$MERGED" || { echo "[merge] ❌ export 실패"; exit 1; }

# 프로브: 홀드아웃에서 6샘플 뽑아 생성 (누수 없는 실제 평가 분포)
PROBE="$DATA_DIR/_infer_probe_holdout6.jsonl"
head -6 "$DATA_DIR/deepvision_holdout.jsonl" > "$PROBE"

echo "[probe] enable_thinking=true 생성 (홀드아웃 6)"
run_py swift infer \
  --model "$MERGED" \
  --model_type "$MODEL_TYPE" \
  --val_dataset "$PROBE" \
  --system "$SYSTEM_PROMPT" \
  --enable_thinking true \
  --infer_backend pt \
  --max_new_tokens 6144 \
  --temperature 0.7 \
  --max_pixels "$MAX_PIXELS" \
  --attn_impl "$ATTN_IMPL" \
  --result_path "$PROJ_DIR/logs/probe_mixed_think_true.jsonl"

echo "[merge_probe] done -> $MERGED"
