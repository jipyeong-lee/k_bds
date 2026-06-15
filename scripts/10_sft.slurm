#!/bin/bash
# =============================================================================
# 10_sft.slurm — [1단계] 베이스 모델 환경 구성 및 SFT (지도 미세조정)
#   계획서 산정: 1노드 × 24h × 8GPU × 5사이클 = 960 노드시간
#   본 스크립트 1회 제출 = 1사이클(하이퍼파라미터 1세트). 5회 반복하여 튜닝.
# =============================================================================
#SBATCH --job-name=kbds-sft
#SBATCH --partition=8gpu          # A100 80GB ×8, 896G, 128core (계획서 요구사양 충족)
#SBATCH --comment=pytorch         # 가이드: 사용 애플리케이션명 명시
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/sft_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/sft_%j.log

set -euo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh

# 호스트 모듈은 conda 폴백 시에만 필요(컨테이너는 자체 cuda/gcc 보유)
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# A100 80GB ×8 기준. VRAM 여유로 batch/length 상향 가능(아래는 안정 기본값).
run_py swift sft \
  --model "$BASE_MODEL" \
  --model_type "$MODEL_TYPE" \
  --dataset "$DATA_DIR/sft_train.jsonl" \
  --val_dataset "$DATA_DIR/sft_val.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --max_length 8192 \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --deepspeed "$PROJ_DIR/configs/ds_zero3.json" \
  --attn_impl "$ATTN_IMPL" \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 8 \
  --output_dir "$CKPT_DIR/sft" \
  --report_to tensorboard

echo "[sft] done -> $CKPT_DIR/sft"
