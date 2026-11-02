#!/bin/bash
# =============================================================================
# 37_merge_opd.slurm — OPD 어댑터(학생 LoRA)를 학생 베이스 merged_ta_w07 에 병합해 평가용 모델을 만든다.
#   학습 베이스가 merged_ta_w07 이었으므로 병합 베이스도 merged_ta_w07 이다(sft_mixed_merged 에 얹으면 틀린 모델이 된다).
#   가중치 산술이라 학습·모델 로딩이 없다(scripts/merge_experts.py, 전문가 통합과 같은 도구).
# 사용:  OPD_CKPT=<체크포인트 디렉터리> OPD_OUT=<출력 디렉터리> sbatch scripts/37_merge_opd.slurm
#        보통은 scripts/37_eval_opd.sh 가 호출한다.
# =============================================================================
#SBATCH --job-name=merge-opd
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/merge_opd_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/merge_opd_%j.log

set -uo pipefail
cd /home01/k266a01/kbds_project
source scripts/00_common.sh

: "${OPD_CKPT:?OPD_CKPT 필요}" "${OPD_OUT:?OPD_OUT 필요}"
STUDENT=${STUDENT:-$CKPT_DIR/merged_ta_w07}
[ -f "$OPD_CKPT/adapter_model.safetensors" ] && [ -f "$OPD_CKPT/adapter_config.json" ] || { echo "❌ 어댑터 파일 없음: $OPD_CKPT"; exit 1; }
[ -f "$STUDENT/model.safetensors.index.json" ] || { echo "❌ 학생 베이스 불완전: $STUDENT"; exit 1; }
[ -e "$OPD_OUT" ] && { echo "❌ 출력이 이미 있다(덮어쓰기 방지): $OPD_OUT"; exit 1; }

run_py python scripts/merge_experts.py --base "$STUDENT" --adapters "$OPD_CKPT" --weights 1.0 --method linear --out "$OPD_OUT" \
  || { echo "❌ merge 실패"; rm -rf "$OPD_OUT"; exit 1; }
for f in config.json model.safetensors.index.json; do [ -f "$OPD_OUT/$f" ] || { echo "❌ 병합본에 $f 없음"; exit 1; }; done
echo "MERGE_OK $OPD_OUT"
