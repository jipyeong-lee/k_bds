#!/bin/bash
# =============================================================================
# launch_stage2_expanded.sh — Stage-2 풀확장 GRPO 제출 (다른 계정용 표준 진입점)
#   전제(먼저 완료): ① 데이터 빌드 → work/data/stage2_expanded_train.jsonl (128,349)
#                   ② v3 콜드스타트 병합본 → work/checkpoints/sft_mixed_merged
#   상세 재현: docs/stage2_expansion_runbook.md
#
# 사용:
#   bash scripts/launch_stage2_expanded.sh              # 본실행(dr_grpo, ~70h)
#   SMOKE=1 bash scripts/launch_stage2_expanded.sh      # 스모크(max_steps 3, 배선확인)
#   RECIPE=gdpo bash scripts/launch_stage2_expanded.sh  # GDPO(Stage-3용 권고 레시피)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/00_common.sh

INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/stage2_expanded_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/grpo_expanded}"

# ── 전제 조건 점검 ──────────────────────────────────────────────────────────
[ -f "$INIT_MODEL/config.json" ] || { echo "❌ INIT 병합본 없음: $INIT_MODEL — 먼저 v3 SFT+병합"; exit 1; }
[ -f "$DATASET_FILE" ] || { echo "❌ 확장셋 없음: $DATASET_FILE — 먼저 13_build_stage2_expanded + build_stage2_mix"; exit 1; }
echo "[launch] INIT=$INIT_MODEL"
echo "[launch] DATA=$DATASET_FILE ($(wc -l <"$DATASET_FILE") 건)"
echo "[launch] OUT =$OUTPUT_DIR"

# ── 레시피(Stage-2 A/B 결론: dr_grpo 코어, GDPO 는 Stage-3용) ────────────────
EXTRA=""
case "${RECIPE:-dr_grpo}" in
  dr_grpo) EXTRA="--loss_type dr_grpo" ;;
  gdpo)    EXTRA="--loss_type dr_grpo --scale_rewards gdpo" ;;
  *) echo "알 수 없는 RECIPE=${RECIPE}"; exit 1 ;;
esac
[ "${SMOKE:-0}" = 1 ] && EXTRA="$EXTRA --max_steps 3"

export INIT_MODEL DATASET_FILE OUTPUT_DIR
EXTRA_ARGS="$EXTRA" sbatch \
  --export=ALL,INIT_MODEL="$INIT_MODEL",DATASET_FILE="$DATASET_FILE",OUTPUT_DIR="$OUTPUT_DIR",EXTRA_ARGS="$EXTRA" \
  scripts/20_rlvr_grpo.slurm
echo "[launch] 제출됨. 진행: squeue -u \$USER / logs/grpo_*.log"
