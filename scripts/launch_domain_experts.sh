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
#     STEPS=1500 · n=8 기준 노출(= 6,000 프롬프트):
#         deepvision 0.15 epoch (혼합 0.09 의 1.7배)   ← 이득이 가장 작다
#         mmk12      0.40 epoch (4.4배)
#         pmcvqa     0.31 epoch (3.4배)
#     n=16 으로 올리면 위 값이 전부 반토막 나 **도메인 분리의 이득이 사라진다.** 8 이 상한이다.
#
#  ⚠️ E1(deepvision)은 이득이 작다. ck-850 이 이미 deepvision 3,638 프롬프트를 봤으므로
#     n=8 에서 그걸 넘으려면 910 step 이 필요하다. STEPS=1500 이면 6,000 = ck-850 의 1.65배.
#     예산이 빠듯하면 **E1 을 빼고 ck-850 을 일반 교사로 쓰는 선택**이 여전히 유효하다
#     (ARMS="mmk12 pmcvqa" 로 제출).
#
# -----------------------------------------------------------------------------
#  비용 — **2026-08-13 스모크(job 75327, deepvision)로 실측 갱신**
#     step time = **213 s/it** (증분 204/170/240/166/283). 구 가정 330 s/it 은 혼합·num_gen=4 때 값이라
#     35% 과대추정이었다. 배치가 32 completion 고정이므로 num_gen 4→8 은 프롬프트를 8→4 로 줄인다
#     → 이미지 절반 + 같은 프롬프트 8개라 prefix caching 이 듣는다. 계산이 준 게 아니라 중복이 는 것.
#
#     1 벽시계시 = 8 GPU-h · 잔여 예산 4,126
#       STEPS=1200 →  71h/arm = 1,704 (41%)
#       STEPS=1500 →  89h/arm = **2,130 (52%)**  ← 채택
#       STEPS=1800 → 106h/arm = 2,556 (62%) · walltime 여유 11% 라 길이 인플레 시 잘린다
#     ⚠️ 파티션 walltime 상한 5일(120h) → 213 s/it 이 유지되면 **약 1,995 step 이 단일 잡의 상한**.
#        잘려도 save_steps 50 + RESUME=1 로 이어받으므로 STEPS 를 미리 정밀하게 맞출 필요는 없다.
#     ⚠️ step time 은 도메인마다 다르다. 위 값은 deepvision(6144) 기준이고 pmcvqa(3072)는 더 빠르다.
#     ✅ **2026-08-21 mmk12 스모크(job 75762) 실측: 정상구간 중앙값 128.7 s/it**(107~190, n=4).
#        213 은 deepvision 값이었고 mmk12 는 완성 길이가 855~1,296 뿐이라 40% 빠르다.
#        step time 은 `completions/mean_length` 가 지배한다 — 길이가 뛴 step(1,296)만 190 이었다.
#        1 epoch 환산: mmk12 3,801 step = 135.8h·1,087 GPU-h · pmcvqa 4,896 step = (미실측)
#        시작 오버헤드는 **잡당 약 36분**(49m23s − step 합계 13m15s). 체인 2~3잡이면 무시 가능.
#        메모리는 68.82 GiB 에서 완전 평탄(beta=0 으로 참조모델 forward 제거 → deepvision 74.1 보다 낮다).
#     ⚠️ memory 가 5 step 만에 52.7→74.1 GiB(80GB 의 93%)까지 올랐다. 본실행 초반에 확인할 것.
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
STEPS="${STEPS:-1500}"   # 213 s/it 실측 → 89 벽시계h/arm. walltime 118h 대비 33% 여유.
NUM_GEN="${NUM_GEN:-8}"

#  ── B200 2차에서 붕괴를 막은 조합 (README 발견 ⑤~⑫) ────────────────────────────
#  1차는 이 넷 중 ENTQ 없이 돌려 **step 700 에 붕괴**했다(tr/ro 208 · fmt 0.936 · len 1487→502).
#  2차는 ENTQ=0.2 로 step 2,203 까지 무붕괴, 보상 0.65 → 0.81.
ENTQ="${ENTQ:-0.2}"                    # top_entropy_quantile · 1.0 이면 마스크 끔
BETA="${BETA:-0}"                      # 참조 KL. 마스크가 KL 항 **이전**에 걸려 beta>0 과 조합이 깨진다
                                       # (grpo_trainer.py: 정책 그래디언트는 상위 20% 토큰, KL 은 100%)
TEMPERATURE="${TEMPERATURE:-1.0}"      # B200 검증값 (slurm 기본 0.9)
SCALE_REWARDS="${SCALE_REWARDS:-gdpo}" # B200 검증값 (stable 기본 none)
WALLTIME="${WALLTIME:-118:00:00}"

#  ── 1 epoch 목표 (EPOCHS) 와 resume 체인 (N_JOBS) ────────────────────────────
#  walltime 상한이 5일(120h)인데 1 epoch 은 그걸 넘는다 → afterany 체인으로 이어학습한다.
#  각 잡이 RESUME=1 로 같은 OUTPUT_DIR 의 최신 checkpoint 에서 재개하고,
#  MAX_STEPS 에 도달하면 남은 잡은 즉시 no-op 종료한다(과다 제출이 싸다).
#      EPOCHS=1 → STEPS 를 도메인별 ceil(건수 × EPOCHS ÷ 프롬프트당step) 으로 덮어쓴다
#      N_JOBS   → 체인 길이. 213 s/it · walltime 118h 기준 1잡 ≈ 1,995 step
EPOCHS="${EPOCHS:-}"
N_JOBS="${N_JOBS:-1}"
#  GPU-h 추정에 쓰는 s/it. 실측: mmk12 129(job 75762) · deepvision 213(job 75327) · pmcvqa 미실측.
SPI="${SPI:-213}"
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
echo "[expert] B200 검증 노브: top_entropy_quantile=$ENTQ · beta=$BETA · temperature=$TEMPERATURE · scale_rewards=$SCALE_REWARDS"
echo "[expert] steps  = $STEPS · walltime $WALLTIME · arms: $ARMS"
echo

for arm in $ARMS; do
  DS="$DATA_DIR/domains/stage2_${arm}.jsonl"
  [[ -f "$DS" ]] || { echo "❌ 데이터 없음: $DS"; exit 1; }
  N=$(wc -l <"$DS")
  #  EPOCHS 가 있으면 도메인 건수로 step 을 직접 계산한다(올림). 없으면 STEPS 를 그대로 쓴다.
  if [[ -n "$EPOCHS" ]]; then
    STEPS_ARM=$(awk "BEGIN{printf \"%d\", int(($N*$EPOCHS + $PPS - 1)/$PPS)}")
  else
    STEPS_ARM="$STEPS"
  fi
  SEEN=$((STEPS_ARM * PPS))
  OUT="$CKPT_DIR/expert_${arm}_${STAMP}"

  #  --export 값에 **쉼표도 공백도 넣지 않는다.** SLURM 이 쉼표로 쪼개고 공백은 버전에 따라 깨진다.
  #  (이 저장소는 EXTRA_ARGS="--seed 1234" 로 한 번 데인 적이 있다 → 커밋 366a04c)
  SUBMIT=(sbatch --job-name="e-$arm" --time="$WALLTIME"
    --export=ALL,RECIPE=stable,RESUME=1,DOMAIN="$arm",NUM_GEN="$NUM_GEN",LORA_DROPOUT=0,MAX_STEPS="$STEPS_ARM",INIT_MODEL="$INIT_MODEL",OUTPUT_DIR="$OUT",WATCHDOG=1,ENTQ="$ENTQ",BETA="$BETA",TEMPERATURE="$TEMPERATURE",SCALE_REWARDS="$SCALE_REWARDS"
    "$PROJ_DIR/scripts/21_rlvr_grpo_adv.slurm")

  printf '[expert] %-11s %6s건 · %s step → %s 프롬프트 = %.3f epoch · 체인 %s잡 (%.0f GPU-h @%ss/it)\n' \
    "$arm" "$N" "$STEPS_ARM" "$SEEN" \
    "$(awk "BEGIN{printf \"%.3f\", $SEEN/$N}")" "$N_JOBS" \
    "$(awk "BEGIN{printf \"%.0f\", $STEPS_ARM*$SPI/3600*8}")" "$SPI"
  echo "         out= $OUT"
  if [[ "${DRY:-0}" == "1" ]]; then
    echo "         (DRY) ${SUBMIT[*]}"
    [[ "$N_JOBS" -gt 1 ]] && echo "         (DRY) + afterany 체인 $((N_JOBS-1))잡 추가"
  else
    PREV=""
    for k in $(seq 1 "$N_JOBS"); do
      #  bash 4.2 는 set -u 에서 빈 배열 "${DEP[@]}" 확장을 unbound 로 죽인다 → 문자열로 쓴다.
      DEP=""
      [[ -n "$PREV" ]] && DEP="--dependency=afterany:$PREV"
      JID=$(sbatch --parsable $DEP "${SUBMIT[@]:1}")
      echo "         제출 $k/$N_JOBS: job $JID  ·  로그 logs/grpo_adv_${JID}.log"
      PREV="$JID"
    done
  fi
  echo
done

cat <<'EOF'
[expert] 모니터:
    squeue -u $USER
    grep -E "global_step|watch" logs/grpo_adv_<JID>.log | tail
    cat logs/verdict_<JID>.json          # 형식 붕괴 감지 시 기록

[expert] 배선 4항목은 스모크(job 75327)에서 전부 통과했다. 본실행에서 볼 것은 따로 있다:
    1) memory(GiB)  — 스모크 5 step 만에 52.7→74.1 (80GB 의 93%). 평탄화되는지 초반 50 step
    2) rollout_correction/chi2_token — 스모크 1.6~4.7%. IcePop 임계 5% 를 **넘어가는지**
       넘으면 --rollout_importance_sampling_mode token_mask 로 보정을 켠다(지금은 계측만)
    3) frac_reward_zero_std — 스모크 내내 0. 학습이 진행되며 올라오면 plateau 재발 신호
    4) completions/mean_length 추세 — 길이 인플레는 붕괴의 2단계였다. 감시자가 형식은 보지만
       길이는 안 본다
EOF
