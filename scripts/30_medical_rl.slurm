#!/bin/bash
# =============================================================================
# 30_medical_rl.slurm — [3단계] 의료 특화 개방형 RL (RaR 루브릭 보상 / 멀티모달 judge)
#   init  = Stage-2 승자 dr_grpo 병합본(dr_grpo_merged)
#   데이터 = medix_rl_train.jsonl (단답 의료 VQA)
#   보상  = format_think(0.2) + clinical_judge(1.0, RaR 루브릭, 외부 멀티모달 judge)
#   judge = 별도 컴퓨트노드 vLLM 서버(judge_server.slurm). endpoint 는 logs/judge_endpoint.txt
#           또는 --export JUDGE_BASE_URL 로 주입. 학습 노드↔judge 노드 내부망 통신(검증됨).
#   학습 방식 = LoRA-DDP (NVLink 없음 대응, Stage-2 와 동일). 정책 rollout 은 colocate vLLM.
#   ▶ 보통 launch_stage3.sh 가 judge 띄우고 endpoint 주입해 제출. 수동 시 JUDGE_BASE_URL 필수.
# =============================================================================
#SBATCH --job-name=kbds-medrl
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00
#SBATCH --output=/home01/k266a01/kbds_project/logs/medrl_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/medrl_%j.log

set -euo pipefail
source /home01/k266a01/kbds_project/scripts/00_common.sh
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/dr_grpo_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/medix_rl_train.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- judge 엔드포인트 (env 주입 우선, 없으면 endpoint 파일) ------------------
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-$(cat "$PROJ_DIR/logs/judge_endpoint.txt" 2>/dev/null || true)}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen36-judge}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
if [[ -z "$JUDGE_BASE_URL" ]]; then
  echo "[medrl] ❌ JUDGE_BASE_URL 없음 — judge 잡 먼저 띄우세요(launch_stage3.sh 권장)"; exit 1
fi
echo "[medrl] init=$INIT_MODEL  judge=$JUDGE_BASE_URL  model=$JUDGE_MODEL"
[[ -d "$INIT_MODEL" ]] || { echo "[medrl] ❌ init 모델 없음: $INIT_MODEL (merge_drgrpo.slurm 먼저)"; exit 1; }

# ---- LoRA (Stage-2 와 동일: no-NVLink → DDP) --------------------------------
LORA_RANK="${LORA_RANK:-16}"; LORA_ALPHA="${LORA_ALPHA:-32}"; LR="${LR:-1e-5}"

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATASET_FILE" \
  --tuner_type lora \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules all-linear \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" "$PROJ_DIR/configs/medical_reward.py" \
  --reward_funcs format_think clinical_judge \
  --reward_weights 0.2 1.0 \
  --enable_thinking true \
  --num_generations "${NUM_GEN:-4}" \
  --max_completion_length "${MAX_COMPLETION:-4096}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "$LR" \
  --beta "${BETA:-0.04}" \
  --max_length "${MAX_LEN:-10240}" \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP:-1}" \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.4}" \
  --sleep_level 1 \
  --log_completions true \
  --save_strategy steps --save_steps 100 --save_total_limit 5 \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR:-$CKPT_DIR/medical_rl}" \
  --report_to tensorboard \
  $EXTRA_ARGS

echo "[medrl] done -> ${OUTPUT_DIR:-$CKPT_DIR/medical_rl}"
