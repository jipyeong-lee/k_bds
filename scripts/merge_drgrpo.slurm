#!/bin/bash
# =============================================================================
# merge_drgrpo.slurm — Stage-2 승자 dr_grpo checkpoint-600(LoRA) → base 병합.
#   결과 work/checkpoints/dr_grpo_merged 를 Stage-3 GRPO 의 init 으로 사용.
#   (sft_rft_coldstart_merged 패턴 동일. 어댑터 base 경로는 adapter_config.json 에 기록됨.)
# =============================================================================
#SBATCH --job-name=merge-drgrpo
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/merge_drgrpo_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/merge_drgrpo_%j.log

set -euo pipefail
source /home01/k266a01/kbds_project/scripts/00_common.sh
export NPROC_PER_NODE=1
ADAPTER="${ADAPTER:-$CKPT_DIR/grpo_general_adv_dr_grpo/v0-20260625-081142/checkpoint-600}"
MERGED="${MERGED:-$CKPT_DIR/dr_grpo_merged}"

echo "[merge] adapter=$ADAPTER -> merged=$MERGED"
run_py swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED"
echo "[merge] done -> $MERGED"
