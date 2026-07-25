#!/bin/bash
# =============================================================================
# 52_eval_v2_baseline.slurm — v2(sft_rft_coldstart) 홀드아웃 재측정 [v3 와 동일조건 A/B]
#   목적: v3(0.348 / format_think 0.909)의 대조군 v2 를 **완전 동일 하니스**로 재측정.
#         과거 v2 0.22 는 조건 불명 → 같은 holdout·system·temp0·eval_v3_holdout.py 로 못박음.
#   v2 는 이미 병합돼 있음(sft_rft_coldstart_merged) → 병합 생략, serve+채점만.
#   temp=0 greedy 라 conc 는 속도만 바꾸고 결과 불변 → conc16 으로 벽시계 여유 확보.
# =============================================================================
#SBATCH --job-name=eval-v2base
#SBATCH --partition=debug-1gpu,1gpu,2gpu,4gpu,8gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
# ↑ 1차 시도(70342)는 1:20 으로 냈다가 TIMEOUT. v2 는 형식이 깨져 있어 생성이 max_tok(2048)
#   까지 자주 차서 v3(mean 1,982자)보다 훨씬 느리다 → conc16 으로도 74분 안에 972건 미완주.
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/eval_v2base_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/eval_v2base_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8197}"
EVAL_N="${EVAL_N:-972}"
RESULT=logs/eval_v3_results.jsonl        # v3 와 같은 파일에 append → 나란히 비교

MODEL="$CKPT_DIR/sft_rft_coldstart_merged"
[ -f "$MODEL/config.json" ] || { echo "❌ v2 merged 없음: $MODEL"; exit 1; }
echo "[serve] v2 baseline = $MODEL (port $PORT)"

singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
  "$SB" vllm serve "$MODEL" --served-model-name evalmodel \
    --max-model-len 6144 --max-num-seqs 16 --gpu-memory-utilization 0.92 --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
    > logs/eval_v2base_serve_${SLURM_JOB_ID}.log 2>&1 &
srv=$! ready=0
for i in $(seq 1 120); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[serve] READY (~$((i*10))s)"; break; }
  kill -0 $srv 2>/dev/null || { echo "❌ vLLM 종료됨"; tail -25 logs/eval_v2base_serve_${SLURM_JOB_ID}.log; break; }
  sleep 10
done

if [ "$ready" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env EVAL_BASE_URL=http://localhost:$PORT/v1 --env EVAL_MODEL=evalmodel \
    --env EVAL_N=$EVAL_N --env EVAL_CONC="${EVAL_CONC:-16}" --env EVAL_TAG=sft_rft_v2 \
    --env EVAL_MAXTOK="${EVAL_MAXTOK:-2048}" --env SYSTEM_PROMPT="$SYSTEM_PROMPT" \
    --env EVAL_RESULT="$RESULT" --env EVAL_PRED_DUMP="logs/eval_v2base_preds_${SLURM_JOB_ID}.jsonl" \
    "$SB" python scripts/eval_v3_holdout.py
fi
kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null || true

echo ""; echo "=====  v2/v3 홀드아웃 나란히 (동일 N=$EVAL_N·하니스)  ====="
tail -2 "$RESULT" 2>/dev/null
echo "[eval-v2base] done"
