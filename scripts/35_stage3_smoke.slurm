#!/bin/bash
# =============================================================================
# 35_stage3_smoke.slurm — Stage-3(RaR) 배선 end-to-end 스모크.
#   목적: ms-swift GRPO 루프 안에서 clinical_judge(RaR) 플러그인이 실제로 호출되고,
#         swift 가 넘기는 kwargs(특히 images 경로)가 판정에 도달하는지 실증(=정적검증 보완).
#   구성: idle 8gpu 노드 1대에서 GPU0=judge(Qwen3.6-27B-FP8), GPU1=소형 GRPO 트레이너(max_steps 2).
#   판정: 로그에 [JUDGE_DEBUG] images 경로 도달 + clinical_judge reward 값이 흐르면 PASS.
# =============================================================================
#SBATCH --job-name=s3-smoke
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/stage3_smoke_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/stage3_smoke_%j.log

set -uo pipefail   # -e 제외: 트레이너 실패해도 judge teardown 보장
cd /home01/k252a01/kbds_project
source scripts/00_common.sh
PORT="${JUDGE_PORT:-8177}"

# ---- 1) judge 서버: GPU0 ----------------------------------------------------
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
JUDGE_PORT=$PORT JUDGE_TP=1 bash scripts/judge_server.sh > logs/smoke_judge_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!
READY=0
for i in $(seq 1 90); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen36-judge && { READY=1; echo "[smoke] judge READY (~$((i*10))s)"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[smoke] ❌ judge 프로세스 종료됨"; tail -25 logs/smoke_judge_${SLURM_JOB_ID}.log; break; }
  sleep 10
done
if [ "$READY" != 1 ]; then echo "[smoke] ❌ judge not ready — abort"; kill $SRV 2>/dev/null; exit 1; fi

# ---- 2) 소형 GRPO 트레이너: GPU1, max_steps 2, clinical_judge 실호출 ---------
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1
export NPROC_PER_NODE=1
export JUDGE_BASE_URL="http://localhost:$PORT/v1" JUDGE_MODEL=qwen36-judge JUDGE_API_KEY=EMPTY
export JUDGE_DEBUG=1          # ← images kwarg 도달 여부 로깅
INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/dr_grpo_merged}"
echo "[smoke] 트레이너 시작: init=$INIT_MODEL judge=$JUDGE_BASE_URL"

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATA_DIR/medix_smoke.jsonl" \
  --tuner_type lora \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" "$PROJ_DIR/configs/medical_reward.py" \
  --reward_funcs format_think clinical_judge \
  --reward_weights 0.2 1.0 \
  --enable_thinking true \
  --num_generations 2 \
  --max_completion_length "${SMOKE_MAXTOK:-4096}" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --beta 0.04 \
  --max_length 8192 \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  --vllm_mm_processor_cache_gb 0 --vllm_gpu_memory_utilization 0.4 --sleep_level 1 \
  --log_completions true --logging_steps 1 \
  --max_steps 2 \
  --save_strategy no \
  --output_dir "$CKPT_DIR/_smoke_stage3" \
  --report_to none
TR=$?
echo "[smoke] 트레이너 종료코드=$TR"

# ---- 3) teardown -----------------------------------------------------------
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null || true
echo "[smoke] done (트레이너 exit=$TR)"
