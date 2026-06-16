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

# format cold-start SFT: <think>간결추론</think><answer>답</answer> 형식+결론 습관 주입.
#  데이터 = VLAA-Thinking clevr_math(~2.9K, build_coldstart_sft.py). system 은 데이터에 내장.
#  소규모이므로 다중 epoch. 이 산출물(sft_coldstart)이 GRPO 의 INIT_MODEL 이 됨:
#    INIT_MODEL=$CKPT_DIR/sft_coldstart sbatch scripts/20_rlvr_grpo.slurm
SFT_DATA="${SFT_DATA:-$DATA_DIR/sft_coldstart_train.jsonl}"
SFT_VAL="${SFT_VAL:-$DATA_DIR/sft_coldstart_val.jsonl}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
TUNER_TYPE="${TUNER_TYPE:-full}"        # full | lora
SFT_OUT="${SFT_OUT:-$CKPT_DIR/sft_coldstart_$TUNER_TYPE}"

# tuner 별 분기: LoRA 는 base 동결 → 단일GPU 적재, DeepSpeed 없이 DDP(통신 최소). full 은 ZeRO-2.
EXTRA_TUNER=""
DS_ARG="--deepspeed $PROJ_DIR/configs/ds_zero2.json"
if [[ "$TUNER_TYPE" == lora ]]; then
  EXTRA_TUNER="--lora_rank ${LORA_RANK:-16} --lora_alpha ${LORA_ALPHA:-32} --target_modules all-linear"
  LR="${LR:-1e-4}"                              # LoRA 는 더 높은 LR
  [[ "${LORA_NO_DS:-1}" == 1 ]] && DS_ARG=""    # LoRA 는 DDP (DeepSpeed 생략)
else
  LR="${LR:-1e-5}"                              # full-FT SFT
fi

# A100 80GB ×8 기준. VRAM 여유로 batch/length 상향 가능(아래는 안정 기본값).
run_py swift sft \
  --model "$BASE_MODEL" \
  --model_type "$MODEL_TYPE" \
  --dataset "$SFT_DATA" \
  --val_dataset "$SFT_VAL" \
  --tuner_type "$TUNER_TYPE" $EXTRA_TUNER \
  --torch_dtype bfloat16 \
  --num_train_epochs "$SFT_EPOCHS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate "$LR" \
  --max_length "${SFT_MAX_LEN:-2048}" \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing false \
  $DS_ARG \
  --attn_impl "$ATTN_IMPL" \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 8 \
  --output_dir "$SFT_OUT" \
  --report_to tensorboard

echo "[sft] done -> $SFT_OUT"
