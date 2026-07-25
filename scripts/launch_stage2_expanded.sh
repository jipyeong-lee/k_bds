#!/bin/bash
# =============================================================================
# launch_stage2_expanded.sh — Stage-2 풀확장 GRPO 제출 (다른 계정용 표준 진입점)
#   ⚠️ 검증된 21_rlvr_grpo_adv.slurm 를 제출한다(20 아님). dr_grpo/GDPO 의 plateau 돌파
#      핵심 = dynamic_sample + overlong_filter + beta 0.04 는 21 에만 있음. 20(단순 GRPO)로
#      돌리면 loss_type/scale_rewards 만 바뀌고 정작 검증된 그 레시피가 재현 안 됨.
#   전제: ① work/data/stage2_expanded_train.jsonl (128,349)  ② work/checkpoints/sft_mixed_merged
#   상세: docs/stage2_expansion_runbook.md
#
# 사용:
#   bash scripts/launch_stage2_expanded.sh              # 기본 = GDPO 본실행 (~70h)
#   RECIPE=dr_grpo bash scripts/launch_stage2_expanded.sh
#   SMOKE=1 bash scripts/launch_stage2_expanded.sh      # 배선 스모크(max_steps 5)
#
#   ── 외부 관행 대조 A/B (기본값=검증값, override 시에만 변경) ──────────────
#   NUM_GEN=8     bash scripts/launch_stage2_expanded.sh   # 그룹 4→8 (관행 8~16)
#   TEMPERATURE=1.0 bash scripts/launch_stage2_expanded.sh # 롤아웃 0.9→1.0 (관행 ~1.0)
#   BETA=0.01     bash scripts/launch_stage2_expanded.sh   # KL 0.04→0.01 (관행 0~0.01)
#   근거·판정: docs/rlvr_hparams_external.md  ⚠️ 기본값 변경 시 과거 A/B clean 비교 깨짐
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/00_common.sh

INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/stage2_expanded_train.jsonl}"

# ── 전제 조건 점검 ──────────────────────────────────────────────────────────
[ -f "$INIT_MODEL/config.json" ] || { echo "❌ INIT 병합본 없음: $INIT_MODEL — 먼저 v3 SFT+병합"; exit 1; }
[ -f "$DATASET_FILE" ] || { echo "❌ 확장셋 없음: $DATASET_FILE — 먼저 13_build_stage2_expanded + build_stage2_mix"; exit 1; }

# ── 레시피 → 21_rlvr_grpo_adv 의 dr_grpo 코어 위에서 scale_rewards 만 토글 ────
#   gdpo    = dr_grpo loss + scale_rewards=gdpo (보상별 advantage 개별정규화). ← 기본(사용자 선택)
#   dr_grpo = dr_grpo loss + scale_rewards=none
#   둘 다 21 의 CORE(dynamic_sample true·max_resample 3·overlong_filter true) + beta 0.04 자동 포함.
RECIPE="${RECIPE:-gdpo}"
case "$RECIPE" in
  gdpo)    ADV_RECIPE=dr_grpo; SCALE=gdpo ;;
  dr_grpo) ADV_RECIPE=dr_grpo; SCALE=none ;;
  *) echo "지원: gdpo | dr_grpo  (dapo/gspo 는 scripts/21_rlvr_grpo_adv.slurm 직접 호출)"; exit 1 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/grpo_expanded_$RECIPE}"

SMOKE_ARG=""
[ "${SMOKE:-0}" = 1 ] && SMOKE_ARG="--max_steps 5"

echo "[launch] RECIPE=$RECIPE → 21_rlvr_grpo_adv (loss=dr_grpo · scale_rewards=$SCALE · dynamic_sample+overlong_filter)"
echo "[launch] INIT=$INIT_MODEL"
echo "[launch] DATA=$DATASET_FILE ($(wc -l <"$DATASET_FILE") 건)"
echo "[launch] OUT =$OUTPUT_DIR"
echo "[launch] knobs: NUM_GEN=${NUM_GEN:-4(기본)}  TEMPERATURE=${TEMPERATURE:-0.9(기본)}  BETA=${BETA:-0.04(기본)}  (관행 대조: docs/rlvr_hparams_external.md)"

# 관행 대조 A/B knob 을 명시적으로 전달(--export=ALL 이 놓치는 경우 대비).
#   설정 안 하면 21 스크립트의 검증된 기본값(4 / 0.9 / 0.04)이 그대로 쓰임.
KNOB_EXPORT=""
[ -n "${NUM_GEN:-}" ]     && KNOB_EXPORT="$KNOB_EXPORT,NUM_GEN=$NUM_GEN"
[ -n "${TEMPERATURE:-}" ] && KNOB_EXPORT="$KNOB_EXPORT,TEMPERATURE=$TEMPERATURE"
[ -n "${BETA:-}" ]        && KNOB_EXPORT="$KNOB_EXPORT,BETA=$BETA"

sbatch --export=ALL,RECIPE="$ADV_RECIPE",SCALE_REWARDS="$SCALE",INIT_MODEL="$INIT_MODEL",DATASET_FILE="$DATASET_FILE",OUTPUT_DIR="$OUTPUT_DIR",EXTRA_ARGS="$SMOKE_ARG"$KNOB_EXPORT \
  scripts/21_rlvr_grpo_adv.slurm
echo "[launch] 제출됨. 진행: squeue -u \$USER / logs/grpo_adv_*.log"
