#!/bin/bash
# =============================================================================
# eval_pass_at_k.slurm — init 모델 pass@k 진단 잡 (K회 재샘플링, temperature=1.0).
#   pmcvqa 학습로그의 frac_reward_zero_std≈49%(deepvision 0.4%)가 "지식 부재"인지
#   "reward 신호 부족"인지 가르기 위한 잡. run_eval 구조는 eval_midtrain.slurm 과 동일,
#   대상이 init 모델 하나뿐이라 스냅샷/병합 단계가 없다. 상세 = eval_pass_at_k.py
# =============================================================================
#SBATCH --job-name=eval-passk
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/eval_passk_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/eval_passk_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
PORT="${EVAL_PORT:-8168}"
EVAL_DATA="${EVAL_DATA:-work/data/pmcvqa_only_holdout.jsonl}"
export EVAL_DATA
[ -f "$EVAL_DATA" ] || { echo "[passk] ❌ 평가셋 없음: $EVAL_DATA"; exit 1; }
TAG="${EVAL_TAG:-init}"
RESULT="${EVAL_RESULT_FILE:-logs/eval_pass_at_k_results_${SLURM_JOB_ID:-manual}.jsonl}"
: > "$RESULT"

MODEL_PATH="${PASSK_MODEL:-$CKPT_DIR/sft_mixed_merged}"

echo "[passk] EVAL_DATA=$EVAL_DATA ($(wc -l < "$EVAL_DATA") 건)  K=${EVAL_K:-8}  MODEL=$MODEL_PATH"
run_serve "$MODEL_PATH" --served-model-name evalmodel \
    --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
    > logs/eval_passk_serve_${SLURM_JOB_ID:-manual}.log 2>&1 &
srv=$! ready=0
for i in $(seq 1 60); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[passk] READY (~$((i*10))s)"; break; }
  kill -0 $srv 2>/dev/null || { echo "[passk] ❌ server died"; tail -15 logs/eval_passk_serve_${SLURM_JOB_ID:-manual}.log; break; }
  sleep 10
done

if [ "$ready" = "1" ]; then
  EVAL_BASE_URL="http://localhost:$PORT/v1" EVAL_MODEL=evalmodel \
  EVAL_N="${EVAL_N:-all}" EVAL_K="${EVAL_K:-8}" EVAL_CONC="${EVAL_CONC:-8}" EVAL_TAG="$TAG" \
  EVAL_MAXTOK="${EVAL_MAXTOK:-8192}" EVAL_RESULT="$RESULT" \
  EVAL_ITEMS="${EVAL_ITEMS:-logs/eval_passk_items_${SLURM_JOB_ID:-manual}.jsonl}" \
    run_py python scripts/eval_pass_at_k.py
fi

# ⚠️ run_serve 는 셸 함수라 $! 는 서브셸 PID 다(runc.sh 가 exec 로 python 이 됨) — eval_midtrain.slurm 과 동일 정리 절차.
pkill -TERM -P "$srv" 2>/dev/null || true
kill -TERM "$srv" 2>/dev/null || true
sleep 5
pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true

echo ""; echo "=====================  pass@k 결과  ====================="
cat "$RESULT"
echo "[passk] done"
