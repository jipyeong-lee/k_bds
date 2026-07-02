#!/bin/bash
# =============================================================================
# eval_midtrain.slurm — Stage-2 중간 홀드아웃 벤치마크 (조기 판단용)
#   dr_grpo 1 epoch 재학습이 길어(~10일) → 중간 체크포인트에서 홀드아웃 정확도가
#   오르는지 조기 확인. base / init(RL 0%) / trained(중간 체크포인트) 3개를 층화
#   홀드아웃에서 동일조건 채점(accuracy_mix, math/vl 층별 분리).
#   핵심 질문: init 대비 trained 가 홀드아웃 Acc 를 올렸나?(=Stage-2 RL 효과 격리)
# =============================================================================
#SBATCH --job-name=eval-mid
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a01/kbds_project/logs/eval_midtrain_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/eval_midtrain_%j.log

set -u
cd /home01/k252a01/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
PORT="${EVAL_PORT:-8165}"
EVAL_N="${EVAL_N:-100}"
RESULT=logs/eval_midtrain_results.jsonl
: > "$RESULT"

# ---- 0) 중간 체크포인트 병합 (복사본에서, 학습의 checkpoint 롤링과 무관) ----
CKPT="${MID_CKPT:-$CKPT_DIR/_mideval_ckpt_step800}"
MERGED="$CKPT_DIR/_midtrain_merged"
export NPROC_PER_NODE=1
rm -rf "$MERGED"
echo "[mid] merge $CKPT -> $MERGED"
run_py swift export --adapters "$CKPT" --merge_lora true --output_dir "$MERGED" \
  || { echo "[mid] ❌ merge 실패"; exit 1; }

# ---- 평가 함수 (40_eval_compare.slurm 과 동일 로직, 결과 파일만 별도) --------
run_eval() {  # $1=tag  $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
    "$SB" vllm serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_mid_serve_${tag}_${SLURM_JOB_ID}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY (~$((i*10))s)"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_mid_serve_${tag}_${SLURM_JOB_ID}.log; break; }
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

BASE_PATH=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
run_eval base    "$BASE_PATH"
run_eval init    "$CKPT_DIR/sft_rft_coldstart_merged"
run_eval trained "$MERGED"

echo ""; echo "=====================  중간 벤치마크 결과 (holdout, math/vl 층별)  ====================="
cat "$RESULT"
rm -rf "$MERGED"   # 임시 병합본 정리
echo "[mid] done"
