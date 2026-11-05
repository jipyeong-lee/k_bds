#!/bin/bash
# =============================================================================
# diag_visual_dependence.slurm — 학생(기본 merged_ta_w07)이 답변 토큰을 만들 때 이미지를 얼마나 쓰는가.
#   VPPO(arXiv 2510.09285)의 시각 의존도 S_t 를 holdout 에서 도메인별·정오답별로 잰다 → scripts/diag_visual_dependence.py
#   목적: VPPO·VA-OPD 를 구현할 가치가 있는지(이미지를 정말 덜 쓰는지) 코드 쓰기 전에 판단. 학습은 하지 않는다.
# 사용:  sbatch scripts/diag_visual_dependence.slurm
#        sbatch --export=ALL,DIAG_MODEL=$PWD/work/checkpoints/sft_mixed_merged,DIAG_TAG=sft scripts/diag_visual_dependence.slurm
# env(선택): DIAG_MODEL DIAG_TAG(ta_w07) DIAG_N(100, 소스당) DIAG_MAXTOK(4096) DIAG_CONC(16) EVAL_PORT(8185) EVAL_DATA
# ⚠ 27B 교사와 비교하려면 8gpu 노드에서 DIAG_MODEL 을 27B 로 주고 --max-model-len 이 맞는지 확인할 것(미시험).
# =============================================================================
#SBATCH --job-name=diag-vis
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/diag_vis_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/diag_vis_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
PORT="${EVAL_PORT:-8185}"
MODEL="${DIAG_MODEL:-$CKPT_DIR/merged_ta_w07}"
TAG="${DIAG_TAG:-ta_w07}"
export EVAL_DATA="${EVAL_DATA:-work/data/stage2_expanded_holdout.jsonl}"
[ -f "$EVAL_DATA" ] && [ -f "$MODEL/config.json" ] || { echo "[diag] ❌ 평가셋 또는 모델 없음: $EVAL_DATA / $MODEL"; exit 1; }
export NPROC_PER_NODE=1

run_serve "$MODEL" --served-model-name evalmodel --max-model-len 10240 --gpu-memory-utilization 0.85 \
    --enforce-eager --limit-mm-per-prompt '{"image": 1}' --chat-template-content-format openai --trust-request-chat-template --host 0.0.0.0 --port "$PORT" \
    > logs/diag_vis_serve_${TAG}_${SLURM_JOB_ID:-manual}.log 2>&1 &
srv=$!; ready=0
for i in $(seq 1 60); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[diag] READY (~$((i*10))s)"; break; }
  kill -0 $srv 2>/dev/null || { echo "[diag] ❌ server died"; tail -15 logs/diag_vis_serve_${TAG}_${SLURM_JOB_ID:-manual}.log; break; }
  sleep 10
done
if [ "$ready" = "1" ]; then
  EVAL_BASE_URL="http://localhost:$PORT/v1" EVAL_MODEL=evalmodel TOKENIZER="$MODEL" \
  DIAG_N="${DIAG_N:-100}" DIAG_MAXTOK="${DIAG_MAXTOK:-4096}" DIAG_CONC="${DIAG_CONC:-16}" \
  DIAG_OUT="logs/diag_visual_dependence_${TAG}_${SLURM_JOB_ID:-manual}.json" \
    run_py python scripts/diag_visual_dependence.py
fi
pkill -TERM -P "$srv" 2>/dev/null || true; kill -TERM "$srv" 2>/dev/null || true
sleep 5; pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
echo "[diag] done"
