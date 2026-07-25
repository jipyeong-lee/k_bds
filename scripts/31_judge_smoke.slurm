#!/bin/bash
# =============================================================================
# 31_judge_smoke.slurm — Stage-3 judge(Qwen3.6-27B-FP8) 컴퓨트노드 스모크 테스트
#   1gpu(40GB, 드라이버550) 에서 vLLM judge 서버 기동 → 실제 medical_reward 경로로
#   멀티모달 채점 1건(좋은답 vs 나쁜답) → 40GB 적합성·응답·파싱·단조성 확인 후 teardown.
#   2gpu 로 바꾸려면: --partition=2gpu --gres=gpu:2 + JUDGE_TP=2.
# =============================================================================
#SBATCH --job-name=judge-smoke
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/judge_smoke_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/judge_smoke_%j.log

set -u
cd /home01/k252a02/kbds_project
SB=work/images/ms-swift-413-sandbox
PORT="${JUDGE_PORT:-8137}"
TP="${JUDGE_TP:-1}"

echo "[smoke] node=$(hostname) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

# 1) judge 서버 백그라운드 기동
JUDGE_PORT=$PORT JUDGE_TP=$TP bash scripts/judge_server.sh > logs/judge_smoke_server_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!
echo "[smoke] judge server PID $SRV (port $PORT, TP=$TP) — 로딩 대기..."

# 2) ready 폴링 (최대 ~10분: 30GB 로딩 + init)
READY=0
for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen36-judge; then
    READY=1; echo "[smoke] ✅ server READY (~$((i*10))s)"; break
  fi
  kill -0 $SRV 2>/dev/null || { echo "[smoke] ❌ server 프로세스 종료됨 (로딩 실패 — judge_smoke_server 로그 확인)"; break; }
  sleep 10
done
echo "[smoke] GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

# 3) 실제 medical_reward 경로 end-to-end 스모크
RC=1
if [ "$READY" = "1" ]; then
  singularity exec --nv --env HF_HUB_OFFLINE=1 \
    --env JUDGE_BASE_URL=http://localhost:$PORT/v1 \
    --env JUDGE_MODEL=qwen36-judge --env JUDGE_API_KEY=EMPTY \
    "$SB" python scripts/judge_smoke_client.py
  RC=$?
else
  echo "[smoke] 서버 미기동 → server 로그 tail:"; tail -25 logs/judge_smoke_server_${SLURM_JOB_ID}.log
fi

# 4) teardown
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null || true
echo "[smoke] done (client rc=$RC)"
