#!/bin/bash
# =============================================================================
# probe_coldstart_infer.slurm — 콜드스타트 병합본 격리 인퍼런스 진단
#   목적: 콜드스타트 모델이 (a)간결히 추론하는가 (b)enable_thinking 영향이 큰가 확인.
#   DeepVision 6샘플(_infer_probe6.jsonl)에 대해 enable_thinking true/false 각각 생성.
# =============================================================================
#SBATCH --job-name=cs-probe
#SBATCH --partition=debug-1gpu     # 40GB ×1, 9B(18GB) 충분
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home01/k266a01/kbds_project/logs/probe_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/probe_%j.log

set -euo pipefail
source /home01/k266a01/kbds_project/scripts/00_common.sh
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi
export NPROC_PER_NODE=1            # debug-1gpu: 단일 GPU (00_common 기본 8 override)

MODEL="$CKPT_DIR/sft_rft_coldstart_merged"
PROBE="$DATA_DIR/_infer_probe6.jsonl"

for ET in true false; do
  echo "================= enable_thinking=$ET ================="
  run_py swift infer \
    --model "$MODEL" \
    --model_type "$MODEL_TYPE" \
    --val_dataset "$PROBE" \
    --system "$SYSTEM_PROMPT" \
    --enable_thinking "$ET" \
    --infer_backend pt \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --max_pixels "$MAX_PIXELS" \
    --attn_impl "$ATTN_IMPL" \
    --result_path "$PROJ_DIR/logs/probe_coldstart_think_${ET}.jsonl"
done
echo "[probe] done"
