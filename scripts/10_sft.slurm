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

# cold-start SFT: <think>간결추론</think><answer>답</answer> 형식+결론 습관 주입.
#  현행 데이터 = v3 혼합(sft_mixed_*, scripts/build_mixed_coldstart.py). system 은 데이터에 내장.
#  이 산출물(sft_mixed_lora → 병합)이 Stage-2 GRPO 의 INIT_MODEL 이 됨:
#    INIT_MODEL=$CKPT_DIR/sft_mixed_merged sbatch scripts/20_rlvr_grpo.slurm
#  (폐기 빌더 v1/v2 는 scripts/_archive/ 참고)
# ── 기본값 = 현행 채택 경로(일반+의료 혼합 콜드스타트). 과거 기본값은 폐기된 v1(clevr_math 단일)
#    이었어서 그냥 sbatch 하면 "일반화 실패로 버린 실험"이 재현됐다. 이력:
#      v1  sft_coldstart_*        : VLAA clevr_math 만 → 도메인 단일, 일반화 실패 → 폐기
#      v2  sft_rft_coldstart_*    : 자기증류 727건 → format_think 0.473(느슨한 필터) → RL 형식 0.425 정체
#      v3  sft_mixed_*  (현행)     : OpenMedReason+VisualWebInstruct+VLAA 혼합, format_think 1.0 강제
#                                   (scripts/build_mixed_coldstart.py)
SFT_DATA="${SFT_DATA:-$DATA_DIR/sft_mixed_train.jsonl}"
SFT_VAL="${SFT_VAL:-$DATA_DIR/sft_mixed_val.jsonl}"
# val 은 epoch1 에 수렴(구 v2 는 4epoch 태웠으나 epoch1 이후 평탄) → 기본 2.
SFT_EPOCHS="${SFT_EPOCHS:-2}"
TUNER_TYPE="${TUNER_TYPE:-lora}"        # full | lora — NVLink 無 클러스터라 전 단계 LoRA 채택
SFT_OUT="${SFT_OUT:-$CKPT_DIR/sft_mixed_$TUNER_TYPE}"
SFT_MAX_LEN="${SFT_MAX_LEN:-4096}"

# 유효배치 = per_device(1) × GRAD_ACCUM × NPROC_PER_NODE. 8gpu 기본에서 1×8×8 = 64.
# 파티션을 바꿔 GPU 수가 달라지면 유효배치가 따라 변하므로 GRAD_ACCUM 으로 64를 유지한다.
#   8gpu: GRAD_ACCUM=8   4gpu: GRAD_ACCUM=16   2gpu: GRAD_ACCUM=32
# 예) 4gpu 로 제출(8gpu 큐가 길 때):
#   sbatch --partition=4gpu --gres=gpu:4 --cpus-per-task=32 \
#          --export=ALL,NPROC_PER_NODE=4,GRAD_ACCUM=16 scripts/10_sft.slurm
GRAD_ACCUM="${GRAD_ACCUM:-8}"

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
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --max_length "$SFT_MAX_LEN" \
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
