#!/bin/bash
# =============================================================================
# launch_stage3.sh — Stage-3 오케스트레이터 (judge 잡 + 학습 잡 순차 기동).
#   1) dr_grpo_merged(init) 없으면 merge 잡 제출·대기
#   2) judge 서버 잡 제출 → logs/judge_ready.txt(health 통과) 대기
#   3) 학습 잡 제출 (judge endpoint 주입)
#   ※ 학습 종료 후 judge 잡은 수동 scancel (동시 가동 노드시간 차감).
# 사용:  bash scripts/launch_stage3.sh
# =============================================================================
set -euo pipefail
cd /home01/k252a01/kbds_project
MERGED=work/checkpoints/dr_grpo_merged

wait_job() {  # $1=jobid : 큐에서 사라질 때까지 대기
  while [ -n "$(squeue -j "$1" -h -o '%T' 2>/dev/null)" ]; do sleep 20; done
}

# 1) init 병합본 보장
if [ ! -d "$MERGED" ]; then
  echo "[launch] dr_grpo_merged 없음 → merge 잡 제출"
  MJID=$(sbatch --parsable scripts/merge_drgrpo.slurm)
  echo "[launch] merge job=$MJID 대기..."; wait_job "$MJID"
  [ -d "$MERGED" ] || { echo "[launch] ❌ merge 실패 — logs/merge_drgrpo_*.log 확인"; exit 1; }
  echo "[launch] ✅ merged init 생성"
fi

# 2) judge 서버 잡 제출 → ready 대기 (file 기반, login→compute 네트워크 불필요)
rm -f logs/judge_ready.txt logs/judge_endpoint.txt
JJID=$(sbatch --parsable scripts/judge_server.slurm)
echo "[launch] judge job=$JJID 제출 — ready 대기(서버 로딩 ~5분)..."
for i in $(seq 1 120); do
  [ -f logs/judge_ready.txt ] && break
  [ -n "$(squeue -j "$JJID" -h -o '%T' 2>/dev/null)" ] || { echo "[launch] ❌ judge 잡 조기종료 — logs/judge_srv_${JJID}.log 확인"; exit 1; }
  sleep 15
done
[ -f logs/judge_ready.txt ] || { echo "[launch] ❌ judge ready 타임아웃"; scancel "$JJID"; exit 1; }
URL=$(cat logs/judge_endpoint.txt)
echo "[launch] ✅ judge READY @ $URL (job=$JJID)"

# 3) 학습 잡 제출 (judge endpoint 주입)
TJID=$(sbatch --parsable --export=ALL,JUDGE_BASE_URL="$URL" scripts/30_medical_rl.slurm)
echo "[launch] ✅ Stage-3 학습 job=$TJID 제출 (judge=$URL)"
echo "[launch] 모니터: squeue -u \$USER ; tail -f logs/medrl_${TJID}.log"
echo "[launch] ⚠️ 학습 종료 후 judge 잡 정리: scancel $JJID"
