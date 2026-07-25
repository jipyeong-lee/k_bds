#!/bin/bash
# =============================================================================
# 48_eval_gspo_holdout.slurm — GSPO step600 홀드아웃 (미측정 갭 보완)
#   GSPO 는 on-policy 판정창(Acc 0.500 vs 0.487)만 봤고 층화 홀드아웃 미측정.
#   dr_grpo(0.380)·GDPO(0.390)와 동일 조건(colds init·trainonly·step600·N=200)으로 대조.
# =============================================================================
#SBATCH --job-name=eval-gspo
#SBATCH --partition=8gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/eval_gspo_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/eval_gspo_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8195}"
EVAL_N="${EVAL_N:-200}"
RESULT=logs/eval_gspo_results.jsonl
: > "$RESULT"

GSPO_CKPT="${GSPO_CKPT:-$(ls -d $CKPT_DIR/grpo_general_adv_gspo_he/*/checkpoint-600 2>/dev/null | head -1)}"
[ -z "$GSPO_CKPT" ] && { echo "❌ GSPO checkpoint-600 없음"; exit 1; }
export NPROC_PER_NODE=1
GM="$CKPT_DIR/_gspo600_merged"

echo "[gspo] merge $GSPO_CKPT -> $GM"
rm -rf "$GM"
run_py swift export --adapters "$GSPO_CKPT" --merge_lora true --output_dir "$GM" || { echo "[gspo] ❌ merge 실패"; exit 1; }

echo "=================  [gspo600] $GM  ================="
singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
  "$SB" vllm serve "$GM" --served-model-name evalmodel \
    --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
    > logs/eval_gspo_serve_${SLURM_JOB_ID}.log 2>&1 &
srv=$! ready=0
for i in $(seq 1 90); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[gspo600] READY"; break; }
  kill -0 $srv 2>/dev/null || { echo "[gspo600] ❌ server died"; tail -15 logs/eval_gspo_serve_${SLURM_JOB_ID}.log; break; }
  sleep 10
done
if [ "$ready" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env EVAL_BASE_URL=http://localhost:$PORT/v1 --env EVAL_MODEL=evalmodel \
    --env EVAL_N=$EVAL_N --env EVAL_CONC="${EVAL_CONC:-16}" --env EVAL_TAG=gspo600 \
    --env EVAL_MAXTOK="${EVAL_MAXTOK:-2048}" --env SYSTEM_PROMPT="$SYSTEM_PROMPT" \
    --env EVAL_RESULT="$RESULT" \
    "$SB" python scripts/eval_compare.py
fi
kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null || true

echo ""; echo "=====  GSPO step600 홀드아웃 (N=$EVAL_N)  ====="
cat "$RESULT"
echo "(비교: dr_grpo 0.380 / GDPO 0.390 — 동일 step600·N=200)"
rm -rf "$GM"
echo "[gspo] done"
