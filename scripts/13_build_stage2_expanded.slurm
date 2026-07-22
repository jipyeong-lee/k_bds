#!/bin/bash
# =============================================================================
# 13_build_stage2_expanded.slurm — Stage-2 풀확장 RLVR 데이터 변환 (CPU 잡)
#   DeepVision(기존) 에 STEM RLVR 데이터 2종을 추가해 Stage-2 다양성↑ (특히 v3 약점=math).
#     · FanqingM/MMK12            (K-12 STEM, question/answer/image, 검증가능 수치)
#     · russwang/ThinkLite-VL-hard-11k (hard 멀티모달, problem/ground_truth)
#   변환은 컨테이너 불필요(pyarrow+PIL 만) → conda `swift` env 로 CPU 노드서 실행
#   (계산노드는 userns=0 이라 컨테이너 불가·로그인노드는 워치독. → CPU 잡이 정답. [[cluster-environment]])
#   산출: work/data/{mmk12,thinklite}_train.jsonl  (+ 이미지 work/data/images/*)
#   다음 단계: scripts/build_stage2_mix.py 로 홀드아웃 분리 + DeepVision 과 혼합.
# =============================================================================
#SBATCH --job-name=s2-build
#SBATCH --partition=cpu64-only     # idle 다수·계수 1·컨테이너 불필요
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home01/k252a01/kbds_project/logs/s2_build_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/s2_build_%j.log

set -uo pipefail
cd /home01/k252a01/kbds_project
PY=/home01/k252a01/.conda/envs/swift/bin/python     # pyarrow17+PIL (ms-swift import 는 깨졌지만 변환엔 무관)
DATA=work/data

MMDIR=$(ls -d work/hf_cache/hub/datasets--FanqingM--MMK12/snapshots/*/data 2>/dev/null | head -1)
TLP=$(ls work/hf_cache/hub/datasets--russwang--ThinkLite-VL-hard-11k/snapshots/*/ThinkLite-VL-Hard-11k.parquet 2>/dev/null | head -1)
[ -z "$MMDIR" ] && { echo "❌ MMK12 미다운로드"; exit 1; }
[ -z "$TLP" ]  && { echo "❌ ThinkLite 미다운로드"; exit 1; }

echo "[build] MMK12 변환 (train 3parquet)"
$PY -u scripts/convert_to_swift.py FanqingM/MMK12 \
  --parquet "$MMDIR"/train-*.parquet \
  --out "$DATA/mmk12_train.jsonl" --images-dir "$DATA/images/mmk12" || exit 1

echo "[build] ThinkLite-VL 변환"
$PY -u scripts/convert_to_swift.py russwang/ThinkLite-VL-hard-11k \
  --parquet "$TLP" \
  --out "$DATA/thinklite_train.jsonl" --images-dir "$DATA/images/thinklite" || exit 1

echo "[build] 완료:"
wc -l "$DATA/mmk12_train.jsonl" "$DATA/thinklite_train.jsonl"
