#!/bin/bash
# =============================================================================
# eval_global_bench.slurm — 원 과제계획서(양식1) 약속 이행: MedXpertQA-MM + MMMU-H&M
#   으로 base(Qwen3.5-9B) vs SFT(sft_mixed_merged) vs RL(pmcvqa 도메인 전문가,
#   checkpoint-4896, 1 epoch 완주) 3개를 동일조건 비교한다.
#   eval_midtrain.slurm 과 동일 패턴(vLLM 서버 기동 → eval 스크립트 → 정리)이지만
#   문항당 이미지가 여러 장이라 --limit-mm-per-prompt 를 올리고 eval_global_bench.py 를 쓴다.
#   데이터 = scripts/build_global_bench.py(로컬) 산출물, work/data/global_bench/ 에 scp 로 반입.
# =============================================================================
#SBATCH --job-name=eval-gbench
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/eval_gbench_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/eval_gbench_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
PORT="${EVAL_PORT:-8170}"

EVAL_DATA="${EVAL_DATA:-work/data/global_bench/global_bench.jsonl}"
export EVAL_DATA
[ -f "$EVAL_DATA" ] || { echo "[gbench] ❌ 평가셋 없음: $EVAL_DATA"; exit 1; }
echo "[gbench] EVAL_DATA=$EVAL_DATA ($(wc -l < "$EVAL_DATA") 건)  EVAL_N=${EVAL_N:-all}"

EVAL_STAGES="${EVAL_STAGES:-base,sft,rl}"
RESULT="${EVAL_RESULT_FILE:-logs/eval_gbench_results_${SLURM_JOB_ID:-manual}.jsonl}"
: > "$RESULT"

# RL 모델 = pmcvqa 도메인 전문가(유일하게 1 epoch 완주, 4,896/4,896). deepvision 은 B200 소재라
# 이 잡에서 바로 못 쓴다 — 필요하면 RL_CKPT 로 override.
RL_CKPT="${RL_CKPT:-$CKPT_DIR/expert_pmcvqa_0821-2028/v2-20260901-191946/checkpoint-4896}"
SNAP="$CKPT_DIR/_gbench_snapshot_${SLURM_JOB_ID:-manual}"
MERGED="$CKPT_DIR/_gbench_merged_${SLURM_JOB_ID:-manual}"
export NPROC_PER_NODE=1
trap 'rm -rf "$SNAP" "$MERGED"' EXIT

has_stage() { case ",$EVAL_STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

if has_stage rl; then
  [ -f "$RL_CKPT/adapter_config.json" ] || { echo "[gbench] ❌ RL_CKPT 아님: $RL_CKPT"; exit 1; }
  rm -rf "$SNAP" "$MERGED"
  echo "[gbench] snapshot $RL_CKPT -> $SNAP"
  cp -a "$RL_CKPT" "$SNAP" || { echo "[gbench] ❌ 스냅샷 실패"; exit 1; }
  echo "[gbench] merge $SNAP -> $MERGED"
  run_py swift export --adapters "$SNAP" --merge_lora true --output_dir "$MERGED" \
    || { echo "[gbench] ❌ merge 실패"; exit 1; }
fi

run_eval() {  # $1=tag  $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  run_serve "$model" --served-model-name evalmodel \
      --max-model-len 12288 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 8}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_gbench_serve_${tag}_${SLURM_JOB_ID:-manual}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY (~$((i*10))s)"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_gbench_serve_${tag}_${SLURM_JOB_ID:-manual}.log; break; }
    sleep 10
  done
  if [ "$ready" = "1" ]; then
    EVAL_BASE_URL="http://localhost:$PORT/v1" EVAL_MODEL=evalmodel \
    EVAL_N="${EVAL_N:-all}" EVAL_CONC="${EVAL_CONC:-16}" EVAL_TAG="$tag" \
    EVAL_MAXTOK="${EVAL_MAXTOK:-4096}" EVAL_RESULT="$RESULT" \
    EVAL_ITEMS="${EVAL_ITEMS_DIR:-logs}/eval_items_gbench_${tag}_${SLURM_JOB_ID:-manual}.jsonl" \
      run_py python scripts/eval_global_bench.py
  fi
  pkill -TERM -P "$srv" 2>/dev/null || true
  kill -TERM "$srv" 2>/dev/null || true
  sleep 5
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  sleep 5
  for i in $(seq 1 12); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || break
    echo "[$tag] 포트 $PORT 해제 대기…"; sleep 5
  done
}

BASE_PATH=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)

echo "[gbench] EVAL_STAGES=$EVAL_STAGES  RL_CKPT=$RL_CKPT"
has_stage base && run_eval base "$BASE_PATH"                            || true
has_stage sft  && run_eval sft  "${SFT_MODEL:-$CKPT_DIR/sft_mixed_merged}" || true
has_stage rl   && run_eval rl   "$MERGED"                                || true

echo ""; echo "=====================  글로벌 벤치마크 결과 (MedXpertQA-MM + MMMU-H&M)  ====================="
cat "$RESULT"
echo "[gbench] done"
