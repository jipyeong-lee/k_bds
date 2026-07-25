#!/bin/bash
# =============================================================================
# 46_eval_gdpo_ab.slurm — GDPO vs dr_grpo 최종 판정 (step600 층화 홀드아웃)
#   clean A/B: 둘 다 init=sft_rft_coldstart_merged · trainonly · loss_type=dr_grpo · step600,
#   scale_rewards 만 none(dr_grpo) vs gdpo. 판정창 on-policy 는 동률 → 홀드아웃이 결정.
#   각 체크포인트 병합 → vLLM 서빙 → deepvision_holdout(math/vl 층화) accuracy_mix 채점.
#   base 0.15 / init 0.22 는 기존 중간벤치(eval_midtrain) 수치 재사용(동일 홀드아웃).
# =============================================================================
#SBATCH --job-name=eval-gdpo-ab
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/eval_gdpo_ab_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/eval_gdpo_ab_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8175}"
EVAL_N="${EVAL_N:-200}"
RESULT=logs/eval_gdpo_ab_results.jsonl
: > "$RESULT"

DRGRPO_CKPT="${DRGRPO_CKPT:-$CKPT_DIR/grpo_general_adv_dr_grpo_he/v0-20260629-170227/checkpoint-600}"
GDPO_CKPT="${GDPO_CKPT:-$CKPT_DIR/grpo_general_adv_gdpo_he/v0-20260704-090102/checkpoint-600}"
export NPROC_PER_NODE=1

merge() {  # $1=ckpt $2=out
  rm -rf "$2"; echo "[ab] merge $1 -> $2"
  run_py swift export --adapters "$1" --merge_lora true --output_dir "$2" \
    || { echo "[ab] ❌ merge 실패: $1"; exit 1; }
}

run_eval() {  # $1=tag $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
    "$SB" vllm serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_gdpoab_serve_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 90); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY (~$((i*10))s)"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_gdpoab_serve_${tag}_${SLURM_JOB_ID}.log; break; }
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

DRM="$CKPT_DIR/_ab_drgrpo600_merged"
GDM="$CKPT_DIR/_ab_gdpo600_merged"
merge "$DRGRPO_CKPT" "$DRM"
merge "$GDPO_CKPT"   "$GDM"

run_eval dr_grpo600 "$DRM"
run_eval gdpo600    "$GDM"

echo ""; echo "===============  GDPO vs dr_grpo (step600, 홀드아웃 math/vl)  ==============="
cat "$RESULT"
echo "(참고 기준선: base 0.15 / init 0.22 — eval_midtrain 동일 홀드아웃)"
rm -rf "$DRM" "$GDM"
echo "[ab] done"
