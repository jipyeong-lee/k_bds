#!/bin/bash
# =============================================================================
# 33_judge_probe.slurm — Stage-3 judge 분포 프로브(spec §8) on 컴퓨트노드(1gpu).
#   judge 서버 기동 → medix N건 ×3변형(good/wrong/halluc) 채점 → 분포·단조성·c2 변별 점검 → teardown.
# =============================================================================
#SBATCH --job-name=judge-probe
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/judge_probe_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/judge_probe_%j.log

set -u
cd /home01/k266a01/kbds_project
SB=work/images/ms-swift-413-sandbox
PORT="${JUDGE_PORT:-8141}"
PROBE_N="${PROBE_N:-100}"

echo "[probe] node=$(hostname) N=$PROBE_N"
JUDGE_PORT=$PORT JUDGE_TP=1 bash scripts/judge_server.sh > logs/judge_probe_server_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!

READY=0
for i in $(seq 1 60); do
  curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen36-judge && { READY=1; echo "[probe] server READY (~$((i*10))s)"; break; }
  kill -0 $SRV 2>/dev/null || { echo "[probe] server died"; break; }
  sleep 10
done

RC=1
if [ "$READY" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env JUDGE_BASE_URL=http://localhost:$PORT/v1 --env JUDGE_MODEL=qwen36-judge --env JUDGE_API_KEY=EMPTY \
    --env PROBE_N=$PROBE_N --env PROBE_CONC="${PROBE_CONC:-8}" \
    "$SB" python scripts/judge_probe.py
  RC=$?
else
  tail -25 logs/judge_probe_server_${SLURM_JOB_ID}.log
fi

kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null || true
echo "[probe] done (rc=$RC)"
