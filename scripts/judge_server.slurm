#!/bin/bash
# =============================================================================
# judge_server.slurm — Stage-3 RaR judge 상주 서버 (Qwen3.6-27B-FP8 멀티모달, vLLM).
#   컴퓨트노드(1gpu 40GB, 드라이버550)에서 가동 → 학습 잡이 내부망으로 호출.
#   health 통과 시 logs/judge_endpoint.txt(http://<node>:PORT/v1) + logs/judge_ready.txt 기록
#   → 오케스트레이터(launch_stage3.sh)가 ready 파일 보고 학습 잡 제출.
#   ⚠️ 학습 종료 후 이 잡은 수동 scancel (동시 가동 노드시간 차감).
# =============================================================================
#SBATCH --job-name=judge-srv
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a01/kbds_project/logs/judge_srv_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/judge_srv_%j.log

set -u
cd /home01/k252a01/kbds_project
PORT="${JUDGE_PORT:-8100}"
TP="${JUDGE_TP:-1}"
rm -f logs/judge_ready.txt logs/judge_endpoint.txt

echo "[judge-srv] node=$(hostname) port=$PORT TP=$TP"
JUDGE_PORT=$PORT JUDGE_TP=$TP bash scripts/judge_server.sh > logs/judge_srv_inner_${SLURM_JOB_ID}.log 2>&1 &
SRV=$!

# health 폴링 → ready/endpoint 기록 (학습 잡은 이 파일을 봄)
for i in $(seq 1 90); do
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen36-judge; then
    echo "http://$(hostname):$PORT/v1" > logs/judge_endpoint.txt
    echo "ready $(date '+%F %T')" > logs/judge_ready.txt
    echo "[judge-srv] ✅ READY @ $(cat logs/judge_endpoint.txt)"
    break
  fi
  kill -0 $SRV 2>/dev/null || { echo "[judge-srv] ❌ server 종료됨"; tail -20 logs/judge_srv_inner_${SLURM_JOB_ID}.log; exit 1; }
  sleep 10
done
[ -f logs/judge_ready.txt ] || { echo "[judge-srv] ❌ ready 타임아웃"; exit 1; }

wait $SRV   # 학습 동안 서버 상주
