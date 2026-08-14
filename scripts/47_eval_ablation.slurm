#!/bin/bash
# =============================================================================
# 47_eval_ablation.slurm — Stage-1 순가치 ablation 홀드아웃 평가
#   base→RL(콜드스타트 無) checkpoint-200 두 arm(dr_grpo/GDPO)을 DeepVision 층화
#   홀드아웃(N=200)에서 채점. base 0.15 / 콜드스타트 0.22 / 콜드스타트+RL(step600) 0.38 과 비교.
#   목적: 콜드스타트 없이 RL 하면 홀드아웃이 콜드스타트 경로에 못 미침을 수치로 확정.
# =============================================================================
#SBATCH --job-name=eval-ablation
#SBATCH --partition=debug-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/eval_ablation_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/eval_ablation_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8185}"
EVAL_N="${EVAL_N:-200}"
RESULT=logs/eval_ablation_results.jsonl
: > "$RESULT"

DR_CKPT=$(ls -d work/checkpoints/grpo_ablation_base_noclod/*/checkpoint-200 2>/dev/null | head -1)
GD_CKPT=$(ls -d work/checkpoints/grpo_ablation_base_noclod_gdpo/*/checkpoint-200 2>/dev/null | head -1)
export NPROC_PER_NODE=1

merge() { rm -rf "$2"; echo "[abl] merge $1 -> $2"
  run_py swift export --adapters "$1" --merge_lora true --output_dir "$2" || { echo "[abl] ❌ merge 실패: $1"; exit 1; }; }

run_eval() {  # $1=tag $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
    "$SB" vllm serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_abl_serve_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 90); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_abl_serve_${tag}_${SLURM_JOB_ID}.log; break; }
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

DRM="$CKPT_DIR/_abl_drgrpo200_merged"
GDM="$CKPT_DIR/_abl_gdpo200_merged"
merge "$DR_CKPT" "$DRM"
merge "$GD_CKPT" "$GDM"
run_eval abl_drgrpo200 "$DRM"
run_eval abl_gdpo200   "$GDM"

echo ""; echo "=====  Stage-1 ablation (base→RL, 콜드스타트 無) step200 홀드아웃  ====="
cat "$RESULT"
echo "(비교 기준선: base 0.15 / 콜드스타트 0.22 / 콜드스타트+dr_grpo step600 0.38)"
rm -rf "$DRM" "$GDM"
echo "[abl] done"
