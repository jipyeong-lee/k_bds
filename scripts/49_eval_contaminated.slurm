#!/bin/bash
# =============================================================================
# 49_eval_contaminated.slurm — DAPO·baseline 홀드아웃 (⚠️ 오염, 참고용)
#   DAPO(57527)·baseline(57249)은 홀드아웃 분리 前 구 데이터(deepvision103k_train,
#   홀드아웃 972건 포함)로 학습 → 홀드아웃 채점은 **학습 데이터로 시험(누수)** = 상향편향.
#   그럼에도 완결성 위해 측정(⚠️ 오염 표기). clean 3종(dr_grpo 0.38/GDPO 0.39/GSPO 0.29)과
#   직접 비교 불가 — DAPO/baseline은 leaked upper-bound.
#   step: DAPO=600(clean과 동일), baseline=1000(600 롤오프됨, 완주점).
# =============================================================================
#SBATCH --job-name=eval-contam
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/eval_contam_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/eval_contam_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8196}"
EVAL_N="${EVAL_N:-200}"
RESULT=logs/eval_contam_results.jsonl
: > "$RESULT"
export NPROC_PER_NODE=1

DAPO_CKPT=$(ls -d $CKPT_DIR/grpo_general_adv_dapo/*/checkpoint-600 2>/dev/null | head -1)
BASE_CKPT=$(ls -d $CKPT_DIR/grpo_general/v11-20260616-165537/checkpoint-1000 2>/dev/null | head -1)

merge() { rm -rf "$2"; echo "[contam] merge $1 -> $2"
  run_py swift export --adapters "$1" --merge_lora true --output_dir "$2" || { echo "[contam] ❌ merge 실패: $1"; return 1; }; }

run_eval() {  # $1=tag $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
    "$SB" vllm serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_contam_serve_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 90); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_contam_serve_${tag}_${SLURM_JOB_ID}.log; break; }
    sleep 10
  done
  if [ "$ready" = "1" ]; then
    singularity exec --nv --env HF_HUB_OFFLINE=1 \
      --env EVAL_BASE_URL=http://localhost:$PORT/v1 --env EVAL_MODEL=evalmodel \
      --env EVAL_N=$EVAL_N --env EVAL_CONC="${EVAL_CONC:-16}" --env EVAL_TAG="$tag" \
      --env EVAL_MAXTOK="${EVAL_MAXTOK:-2048}" --env SYSTEM_PROMPT="$SYSTEM_PROMPT" \
      --env EVAL_RESULT="$RESULT" \
      "$SB" python scripts/eval_compare.py
  fi
  kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null || true; sleep 5
}

DM="$CKPT_DIR/_contam_dapo600_merged"; BM="$CKPT_DIR/_contam_base1000_merged"
[ -n "$DAPO_CKPT" ] && merge "$DAPO_CKPT" "$DM" && run_eval dapo600_LEAK "$DM"
[ -n "$BASE_CKPT" ] && merge "$BASE_CKPT" "$BM" && run_eval baseline1000_LEAK "$BM"

echo ""; echo "=====  ⚠️ DAPO·baseline 홀드아웃 (오염/누수, 참고용, N=$EVAL_N)  ====="
cat "$RESULT"
echo "(⚠️ leaked: 구 데이터에 홀드아웃 포함 → 상향편향. clean: dr_grpo 0.38/GDPO 0.39/GSPO 0.29)"
rm -rf "$DM" "$BM"
echo "[contam] done"
