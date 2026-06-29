#!/bin/bash
# =============================================================================
# 34_rubric_compare.slurm — 정적 통일 루브릭 vs 인스턴스(논문식) 루브릭 비교.
#   judge(Qwen3.6-27B-FP8) 서버 기동 → judge_compare_rubric.py 로 둘 다 채점·비교 → teardown.
#   Stage-2 학습(8gpu)과 별개 1gpu 라 동시 실행 가능.
# =============================================================================
#SBATCH --job-name=rub-cmp
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a01/kbds_project/logs/rubric_compare_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/rubric_compare_%j.log

set -u
cd /home01/k252a01/kbds_project
SB=work/images/ms-swift-413-sandbox
PORT="${JUDGE_PORT:-8143}"

JUDGE_PORT=$PORT JUDGE_TP=1 bash scripts/judge_server.sh > logs/rubric_cmp_server_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!
READY=0
for i in $(seq 1 60); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen36-judge && { READY=1; echo "[cmp] server READY"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[cmp] server died"; break; }
  sleep 10
done

if [ "$READY" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env JUDGE_BASE_URL=http://localhost:$PORT/v1 --env JUDGE_MODEL=qwen36-judge --env JUDGE_API_KEY=EMPTY \
    --env CMP_N="${CMP_N:-40}" --env CMP_CONC="${CMP_CONC:-8}" \
    "$SB" python scripts/judge_compare_rubric.py
fi
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null || true
echo "[cmp] done"
