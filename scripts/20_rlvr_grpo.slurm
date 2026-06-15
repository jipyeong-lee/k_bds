#!/bin/bash
# =============================================================================
# 20_rlvr_grpo.slurm — [2단계] 범용 멀티모달 RLVR/GRPO 훈련
#   데이터: DeepVision-103K (검증가능 정답 → 규칙기반 보상)
#   계획서 산정: 1노드 × 70h × 8GPU × 6사이클(ablation) = 3,360 노드시간
#   GRPO: 정책/참조 모델 + vLLM rollout 동시 적재 → VRAM 압박 큼.
#         8gpu=A100 80GB라 7B는 server 모드(GPU 분리)로 안정. colocate도 가능.
#   ※ ablation 6사이클을 동시 제출하면 8gpu 동시제출 한도(6개)에 도달.
# =============================================================================
#SBATCH --job-name=kbds-grpo
#SBATCH --partition=8gpu          # A100 80GB ×8 — GRPO에 충분
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00           # 5일(120h) 한도 내. 사이클당 70h 예상
#SBATCH --output=/home01/k252a01/kbds_project/logs/grpo_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/grpo_%j.log

set -euo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh
# 호스트 모듈은 conda 폴백 시에만 필요(컨테이너는 자체 cuda/gcc 보유)
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# SFT 생략(R1-zero) → 베이스 모델에서 바로 GRPO. (SFT 도입 시 INIT_MODEL 로 체크포인트 지정)
INIT_MODEL="${INIT_MODEL:-$BASE_MODEL}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/deepvision103k_train.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # 스모크 테스트: EXTRA_ARGS="--max_steps 3"
# vLLM rollout 를 별도 GPU에 두는 colocate 모드(vLLM이 학습 GPU 공유, sleep_level로 교대). server 모드는 외부 vLLM 서버 필요.

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATASET_FILE" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --reward_funcs accuracy format \
  --num_generations 8 \
  --max_completion_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --max_length 8192 \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --deepspeed "$PROJ_DIR/configs/ds_zero3.json" \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.6 \
  --sleep_level 1 \
  --save_strategy steps --save_steps 200 --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR:-$CKPT_DIR/grpo_general}" \
  --report_to tensorboard \
  $EXTRA_ARGS

echo "[grpo] done -> $CKPT_DIR/grpo_general"
