#!/bin/bash
# =============================================================================
# 30_medical_rl.slurm — [3단계] 의료 특화 개방형 RL (복합 보상 / LLM-as-a-judge)
#   데이터: medix-rl-data (자유형식 임상 QA). 정답 일치가 아닌 의미론적·임상적 평가.
#   복합 보상 = 규칙(format) + LLM 심판(임상 유효성). 심판은 vLLM 서버로 가동.
#   2단계 산출물(grpo_general)을 베이스로 의료 도메인 전이.
# =============================================================================
#SBATCH --job-name=kbds-medrl
#SBATCH --partition=8gpu          # A100 80GB ×8
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/medrl_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/medrl_%j.log

set -euo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh
# 호스트 모듈은 conda 폴백 시에만 필요(컨테이너는 자체 cuda/gcc 보유)
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

GEN_CKPT="${GEN_CKPT:-$CKPT_DIR/grpo_general}"

# 복합 보상:
#   - format       : ms-swift 내장 형식 보상
#   - external      : configs/medical_reward.py 의 LLM-as-a-judge (임상 유효성)
#   심판 모델은 --reward_model 또는 외부 vLLM judge 서버로 연동. 아래는 plugin 방식 예시.
run_py swift rlhf \
  --rlhf_type grpo \
  --model "$GEN_CKPT" \
  --model_type "$MODEL_TYPE" \
  --dataset "$DATA_DIR/medix_rl_train.jsonl" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/medical_reward.py" \
  --reward_funcs format clinical_judge \
  --num_generations 8 \
  --max_completion_length 3072 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 \
  --max_length 12288 \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --deepspeed "$PROJ_DIR/configs/ds_zero3.json" \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.6 \
  --sleep_level 1 \
  --save_strategy steps --save_steps 200 --save_total_limit 3 \
  --logging_steps 5 \
  --output_dir "$CKPT_DIR/medical_rl" \
  --report_to tensorboard

echo "[medrl] done -> $CKPT_DIR/medical_rl"
