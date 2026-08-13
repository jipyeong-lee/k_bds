#!/bin/bash
# =============================================================================
# launch_domain_experts.sh — 도메인 전문가 3종 동시 제출 (DeepSeek-V4 구조)
# -----------------------------------------------------------------------------
#  구조: sft_mixed_merged 를 공통 base 로 두고 도메인별 LoRA 전문가를 따로 키운 뒤
#        나중에 통합한다. → docs/deepseek_v4_pipeline_adoption.md
#
#        sft_mixed_merged ─┬→ [E1] deepvision 40,000
#                          ├→ [E2] mmk12      15,204
#                          └→ [E3] pmcvqa     19,583  →(뒤에) medix RaR
#
#  전제: python3 scripts/split_stage2_by_source.py 로 도메인 분할이 끝나 있어야 한다.
#
# -----------------------------------------------------------------------------
#  🚨 **num_generations 는 계산이 아니라 데이터 노출을 깎는다.**
#     배치는 32 completion 고정 → 프롬프트/step = 32 ÷ num_gen.
#         n=4 → 8/step      n=8 → 4/step      n=16 → 2/step
#     STEPS=1200 · n=8 기준 노출(= 4,800 프롬프트):
#         deepvision 0.12 epoch (혼합 0.09 의 1.3배)   ← 이득이 가장 작다
#         mmk12      0.32 epoch (3.5배)
#         pmcvqa     0.25 epoch (2.7배)
#     n=16 으로 올리면 위 값이 전부 반토막 나 **도메인 분리의 이득이 사라진다.** 8 이 상한이다.
#
#  ⚠️ E1(deepvision)은 이득이 작다. ck-850 이 이미 deepvision 3,638 프롬프트를 봤으므로
#     n=8 에서 그걸 넘으려면 910 step 이 필요하다. STEPS=1200 이면 4,800 = ck-850 의 1.32배.
#     예산이 빠듯하면 **E1 을 빼고 ck-850 을 일반 교사로 쓰는 선택**이 여전히 유효하다
#     (ARMS="mmk12 pmcvqa" 로 제출).
#
# -----------------------------------------------------------------------------
#  비용 (330 s/it 실측 · 1 벽시계시 = 8 GPU-h · 잔여 예산 4,126)
#     STEPS=1200 → 전문가당 110 벽시계h = 880 GPU-h · 3개 = **2,640 (64%)**
#     ⚠️ 파티션 walltime 상한 5일(120h) → **1,200 step 이 단일 잡의 안전 상한**이다.
#        그 이상은 잡을 연쇄해야 하고 큐 대기(실측 5일 13시간)가 한 번 더 붙는다.
#     ⚠️ step time 은 도메인마다 다를 것이다. pmcvqa 는 max_completion 이 3072 이라 더 빠르다.
#        **SMOKE=1 로 먼저 재고 STEPS 를 조정할 것.**
#
#  ⚠️ 노드가 정확히 3개다. 3 arm 동시 제출 = 파티션 전체 점유.
#     다른 사용자와 충돌할 수 있으니 필요하면 ARMS 를 나눠 두 번 제출한다.
#
#  사용:
#      SMOKE=1 bash scripts/launch_domain_experts.sh        # 5 step 스모크(권장 선행)
#      bash scripts/launch_domain_experts.sh                # 3 arm 본실행
#      ARMS="pmcvqa" STEPS=1200 bash scripts/launch_domain_experts.sh
#      DRY=1 bash scripts/launch_domain_experts.sh          # 제출 안 함
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}"
ARMS="${ARMS:-deepvision mmk12 pmcvqa}"
STEPS="${STEPS:-1200}"
NUM_GEN="${NUM_GEN:-8}"
WALLTIME="${WALLTIME:-118:00:00}"
STAMP="${STAMP:-$(date +%m%d-%H%M)}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS=5; WALLTIME="02:00:00"; STAMP="smoke-$STAMP"
  echo "[expert] 🔍 SMOKE 모드 — 5 step · walltime 2h · 도메인별 step time 측정이 목적"
fi

[[ -f "$INIT_MODEL/config.json" ]] || { echo "❌ init 모델 없음: $INIT_MODEL"; exit 1; }
[[ -d "$DATA_DIR/domains" ]] || {
  echo "❌ 도메인 분할 없음: $DATA_DIR/domains"
  echo "   먼저: python3 scripts/split_stage2_by_source.py"; exit 1; }

PPS=$((32 / NUM_GEN))
echo "[expert] init   = $INIT_MODEL"
echo "[expert] recipe = stable · num_gen=$NUM_GEN (프롬프트/step=$PPS) · lora_dropout=0"
echo "[expert] steps  = $STEPS · walltime $WALLTIME · arms: $ARMS"
echo

for arm in $ARMS; do
  DS="$DATA_DIR/domains/stage2_${arm}.jsonl"
  [[ -f "$DS" ]] || { echo "❌ 데이터 없음: $DS"; exit 1; }
  N=$(wc -l <"$DS")
  SEEN=$((STEPS * PPS))
  OUT="$CKPT_DIR/expert_${arm}_${STAMP}"

  #  --export 값에 **쉼표도 공백도 넣지 않는다.** SLURM 이 쉼표로 쪼개고 공백은 버전에 따라 깨진다.
  #  (이 저장소는 EXTRA_ARGS="--seed 1234" 로 한 번 데인 적이 있다 → 커밋 366a04c)
  SUBMIT=(sbatch --job-name="e-$arm" --time="$WALLTIME"
    --export=ALL,RECIPE=stable,DOMAIN="$arm",NUM_GEN="$NUM_GEN",LORA_DROPOUT=0,MAX_STEPS="$STEPS",INIT_MODEL="$INIT_MODEL",OUTPUT_DIR="$OUT",WATCHDOG=1
    "$PROJ_DIR/scripts/21_rlvr_grpo_adv.slurm")

  printf '[expert] %-11s %6s건 · %s step → %s 프롬프트 = %.3f epoch (혼합 0.09 의 %.1f배)\n' \
    "$arm" "$N" "$STEPS" "$SEEN" \
    "$(awk "BEGIN{printf \"%.3f\", $SEEN/$N}")" \
    "$(awk "BEGIN{printf \"%.1f\", ($SEEN/$N)/0.09}")"
  echo "         out= $OUT"
  if [[ "${DRY:-0}" == "1" ]]; then
    echo "         (DRY) ${SUBMIT[*]}"
  else
    JID=$("${SUBMIT[@]}" | awk '{print $NF}')
    echo "         제출: job $JID  ·  로그 logs/grpo_adv_${JID}.log"
  fi
  echo
done

cat <<'EOF'
[expert] 모니터:
    squeue -u $USER
    grep -E "global_step|watch" logs/grpo_adv_<JID>.log | tail
    cat logs/verdict_<JID>.json          # 형식 붕괴 감지 시 기록

[expert] 스모크에서 반드시 확인할 것:
    1) step time  — 도메인별로 다르다. 본실행 STEPS 를 여기에 맞춰 조정
    2) rollout_correction/* 지표가 로그에 찍히는가 (학습–추론 불일치 계측)
    3) rewards/FormatThink/mean 이 0.9 근처인가 (단계형 보상 정상 동작)
    4) completions/mean_length 가 도메인 예산 안인가 (pmcvqa 는 3072)
EOF
