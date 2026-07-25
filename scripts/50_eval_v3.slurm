#!/bin/bash
# =============================================================================
# 50_eval_v3.slurm — 콜드스타트 v3(sft_mixed) 병합 + 홀드아웃 평가 (단일 GPU 통합잡)
#   왜 통합: 평가(eval_compare)는 병합 가중치를 vLLM 로 serve 해야 하므로 병합이 어차피
#   1단계. merge→serve→홀드아웃채점 을 한 잡에 묶어 큐 대기 1회로 끝낸다.
#   왜 멀티파티션+gpu:1: 짧은 gpu:1 잡이라 어느 파티션이든 먼저 비는 슬롯에 백필로 들어감.
#     (debug-1gpu 만 노리면 2일째 정체 재현 → 전 GPU 파티션 나열해 first-free 선점.)
#   산출물: (1) sft_mixed_merged  (Stage-2 GRPO 의 INIT_MODEL)
#           (2) logs/eval_v3_results.jsonl  — accuracy / format_think(STRICT) / per-stratum
#              v2 대조: sft_rft_coldstart 홀드아웃 acc 0.22, RL 형식천장(데이터 format_think 0.473)
# =============================================================================
#SBATCH --job-name=eval-v3
#SBATCH --partition=debug-1gpu,1gpu,2gpu,4gpu,8gpu   # 먼저 비는 파티션에 백필
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/eval_v3_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/eval_v3_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
SB="$CONTAINER_IMG"
export NPROC_PER_NODE=1
PORT="${EVAL_PORT:-8196}"
EVAL_N="${EVAL_N:-972}"           # 전체 홀드아웃(math+vl)
RESULT=logs/eval_v3_results.jsonl

# ---- 1) 병합 (없을 때만) ----------------------------------------------------
MERGED="$CKPT_DIR/sft_mixed_merged"
ADAPTER=$(ls -td "$CKPT_DIR"/sft_mixed_lora/*/checkpoint-* 2>/dev/null | head -1)
[ -z "$ADAPTER" ] && { echo "❌ sft_mixed_lora 어댑터 없음"; exit 1; }
if [ -f "$MERGED/config.json" ]; then
  echo "[merge] 이미 존재: $MERGED (재사용)"
else
  echo "[merge] $ADAPTER -> $MERGED"
  run_py swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$MERGED" \
    || { echo "❌ merge 실패"; exit 1; }
fi

# ---- 2) vLLM serve (40GB 안전 설정: 어느 파티션이든 뜨게) ---------------------
echo "[serve] $MERGED (port $PORT)"
singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False \
  "$SB" vllm serve "$MERGED" --served-model-name evalmodel \
    --max-model-len 6144 --max-num-seqs 8 --gpu-memory-utilization 0.92 --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
    > logs/eval_v3_serve_${SLURM_JOB_ID}.log 2>&1 &
srv=$! ready=0
for i in $(seq 1 120); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[serve] READY (~$((i*10))s)"; break; }
  kill -0 $srv 2>/dev/null || { echo "❌ vLLM 서버 종료됨"; tail -25 logs/eval_v3_serve_${SLURM_JOB_ID}.log; break; }
  sleep 10
done

# ---- 3) 홀드아웃 채점 -------------------------------------------------------
if [ "$ready" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env EVAL_BASE_URL=http://localhost:$PORT/v1 --env EVAL_MODEL=evalmodel \
    --env EVAL_N=$EVAL_N --env EVAL_CONC="${EVAL_CONC:-8}" --env EVAL_TAG="${EVAL_TAG:-sft_mixed_v3}" \
    --env EVAL_MAXTOK="${EVAL_MAXTOK:-2048}" --env SYSTEM_PROMPT="$SYSTEM_PROMPT" \
    --env EVAL_RESULT="$RESULT" --env EVAL_PRED_DUMP="logs/eval_v3_preds_${SLURM_JOB_ID}.jsonl" \
    "$SB" python scripts/eval_v3_holdout.py
fi
kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null || true

echo ""; echo "=====  v3(sft_mixed) 홀드아웃 (N=$EVAL_N)  ====="
tail -1 "$RESULT" 2>/dev/null
echo "(v2 대조: sft_rft_coldstart 홀드아웃 acc≈0.22 / 학습데이터 format_think 0.473)"
echo "[eval-v3] done -> $MERGED , $RESULT"
