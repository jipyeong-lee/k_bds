#!/bin/bash
# =============================================================================
# merge_probe_rft.slurm — RFT 간결 콜드스타트 LoRA 병합 + 길이 검증 인퍼런스
#   1) sft_rft_coldstart_lora/checkpoint-48 → base 병합 → sft_rft_coldstart_merged
#   2) 동일 6샘플(_infer_probe6) enable_thinking=true 생성 → 추론 길이 측정
#      (비교군: 옛 콜드스타트 p50 5099자, 13000+ 폭주)
# =============================================================================
#SBATCH --job-name=rft-merge-probe
#SBATCH --partition=debug-1gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/rftmerge_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/rftmerge_%j.log

set -euo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh
export NPROC_PER_NODE=1
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

ADAPTER=$(ls -td "$CKPT_DIR"/sft_rft_coldstart_lora/*/checkpoint-* | head -1)
MERGED="$CKPT_DIR/sft_rft_coldstart_merged"
echo "[merge] adapter=$ADAPTER -> $MERGED"

run_py swift export \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --output_dir "$MERGED"

echo "[probe] enable_thinking=true 생성"
run_py swift infer \
  --model "$MERGED" \
  --model_type "$MODEL_TYPE" \
  --val_dataset "$DATA_DIR/_infer_probe6.jsonl" \
  --system "$SYSTEM_PROMPT" \
  --enable_thinking true \
  --infer_backend pt \
  --max_new_tokens 6144 \
  --temperature 0.7 \
  --max_pixels "$MAX_PIXELS" \
  --attn_impl "$ATTN_IMPL" \
  --result_path "$PROJ_DIR/logs/probe_rft_think_true.jsonl"

echo "[merge_probe] done"
