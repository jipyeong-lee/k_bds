#!/bin/bash
# =============================================================================
# 40_eval_compare.slurm — base(Qwen3.5-9B) vs 학습모델(dr_grpo_merged) 벤치마크 비교.
#   각 모델을 vLLM OpenAI 서버로 띄우고 DeepVision 홀드아웃 N건 동일조건 평가 → accuracy_mix.
#   결과: logs/eval_compare_results.jsonl + 로그. (Stage-2 학습 효과 정량화)
# =============================================================================
#SBATCH --job-name=eval-cmp
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a01/kbds_project/logs/eval_compare_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/eval_compare_%j.log

set -u
cd /home01/k252a01/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8155}"
EVAL_N="${EVAL_N:-150}"
: > logs/eval_compare_results.jsonl   # 초기화

run_eval() {  # $1=tag  $2=model(경로 or HF id)
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    "$SB" vllm serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_serve_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  local srv=$!
  local ready=0
  for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] server READY (~$((i*10))s)"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_serve_${tag}_${SLURM_JOB_ID}.log; break; }
    sleep 10
  done
  if [ "$ready" = "1" ]; then
    singularity exec --nv --env HF_HUB_OFFLINE=1 \
      --env EVAL_BASE_URL=http://localhost:$PORT/v1 --env EVAL_MODEL=evalmodel \
      --env EVAL_N=$EVAL_N --env EVAL_CONC="${EVAL_CONC:-8}" --env EVAL_TAG="$tag" \
      --env SYSTEM_PROMPT="$SYSTEM_PROMPT" \
      "$SB" python scripts/eval_compare.py
  fi
  kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null || true
  sleep 5
}

run_eval base    "Qwen/Qwen3.5-9B"
run_eval trained "$CKPT_DIR/dr_grpo_merged"

echo ""
echo "=====================  비교 결과  ====================="
cat logs/eval_compare_results.jsonl
echo "[eval-cmp] done"
