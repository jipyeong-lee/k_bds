#!/bin/bash
# =============================================================================
# 11_build_coldstart.slurm — 콜드스타트 v3 혼합 데이터 빌드 (CPU 전용)
#   로그인노드에서 돌리면 안 된다: motd 의 "login-only node ... analysis workloads
#   restricted ... processes may be restricted without prior notice" 대로
#   2026-07-16 실제로 워치독에 죽었다(~15분, omr 3,927장에서 중단, 출력도 버퍼째 유실).
#   parquet 16GB(OMR 7 + VWI 8) 스캔 + 이미지 ~10.5K 추출이라 CPU 잡이 맞다.
#   GPU 불필요 → 노드시간 소모 거의 없음(cpu32 = 계수 1).
# =============================================================================
#SBATCH --job-name=build-cs
#SBATCH --partition=cpu32-only
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/build_coldstart_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/build_coldstart_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh

N_OMR="${N_OMR:-5000}"      # 의료 — 정답문자 균형(A 편향 제거)
N_VWI="${N_VWI:-3000}"      # 일반 — difficulty 층화
N_VLAA="${N_VLAA:-2500}"    # 일반 — 이미 목표 형식

echo "[build] OMR=$N_OMR VWI=$N_VWI VLAA=$N_VLAA  node=$(hostname)"

# python -u : 버퍼링 끄기(죽어도 로그가 남게)
singularity exec \
  --bind "$WORK_DIR:$WORK_DIR" --bind "$PROJ_DIR:$PROJ_DIR" \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  "$CONTAINER_IMG" python -u scripts/build_mixed_coldstart.py \
    --n-omr "$N_OMR" --n-vwi "$N_VWI" --n-vlaa "$N_VLAA" \
    --out-train "$DATA_DIR/sft_mixed_train.jsonl" \
    --out-val "$DATA_DIR/sft_mixed_val.jsonl" \
    --images-dir "$DATA_DIR/images/coldstart_mixed"

echo "[build] rc=$?"
wc -l "$DATA_DIR"/sft_mixed_*.jsonl 2>/dev/null
echo "[build] images: $(find "$DATA_DIR/images/coldstart_mixed" -type f | wc -l)"
