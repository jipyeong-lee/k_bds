#!/bin/bash
# =============================================================================
# launch_replay_ckpt800.sh — checkpoint-800 재현 실험 (붕괴 개시 원인 규명)
# -----------------------------------------------------------------------------
#  풀려던 질문: **왜 하필 step 800~900 인가.**
#  사후분석(docs/stage2_run73924_postmortem.md §7-1)에서 유일하게 남은 ❓ 다.
#  세 축(모델·하이퍼·데이터) 어느 것도 개시 시점을 짚지 못했고, 로그만으로는 불가능하다.
#  → checkpoint-800 에서 다시 굴려 본다.
#
#  전제 확인 완료(2026-08-06): checkpoint-800 에 optimizer.pt · scheduler.pt ·
#  rng_state_0..7 · trainer_state.json(global_step=800, max_steps=2337) 이 모두 온전하다.
#  즉 옵티마이저 모멘텀·LR 위치·데이터로더 위치까지 복원되는 **진짜 재개**다.
#
# -----------------------------------------------------------------------------
#  실험 arm (ARMS 환경변수로 선택, 공백 구분)
#
#   replay  원본과 완전 동일. gdpo · MAX_STEPS=2337(=LR 지평 보존)
#           ▸ 묻는 것: 붕괴가 **재현되는가**
#           ▸ vLLM 롤아웃 샘플링은 비트단위 결정론이 아니다(T=0.9). 따라서 이 arm 은
#             "같은 조건 + 새 샘플링 잡음" 이고, 그게 정확히 알고 싶은 것이다 —
#             재현되면 **체계적**, 안 되면 저확률 사건이 취약한 상태를 건드린 것.
#           ▸ 안 터지면 replay 를 한 번 더 돌려 N=2 로 확인할 것(라벨 자동 분리됨).
#
#   lr      MAX_STEPS=1200 만 변경 → LR 지평 정정. §3-4 가설 직격.
#           step 900 에서 6.77e-6 → 1.46e-6 (4.6배 감소). 계산:
#              step   T=2337(원본)   T=1200(arm lr)   비율
#               800     7.377e-06      2.500e-06      2.95x
#               850     7.076e-06      1.956e-06      3.62x
#               900     6.766e-06      1.464e-06      4.62x
#           ⚠️ scheduler.pt 는 last_epoch 만 복원하고 lr_lambda 는 새 max_steps 로 다시
#              만들어진다 → 재개 순간 LR 이 7.38e-6 에서 2.50e-6 으로 점프한다. 의도된 것이다
#              ("처음부터 1200 으로 계획했다면" 의 반사실이 아니라 근사임을 유념).
#
#   seed    SEED=1234 만 변경 → 데이터 순서·샘플링 교란.
#           ▸ 묻는 것: 특정 데이터 구간이 방아쇠인가
#
#   none    scale_rewards=none 만 변경.
#           ▸ GDPO 는 §4-2(a) 에서 이미 대체로 무죄로 판정됐다(재가중 ±15%).
#             이 arm 은 그 판정의 **반증 시도**다. 여기서도 터지면 판정이 굳는다.
#
#  기본은 replay 하나만 — 나머지는 replay 결과를 보고 정하는 게 싸다.
#  판단 순서: replay 재현됨 → lr 로 간다 / 재현 안 됨 → replay 재실행(N=2)부터.
#
# -----------------------------------------------------------------------------
#  비용: arm 당 160 step × ~160 s/it ≈ 7.1h + 기동 0.5h ≈ **7.6 노드시간 (≈61 GPU-h)**
#        감시자가 붕괴 즉시 세우므로 터지는 arm 은 이보다 짧다(붕괴가 ~step 900 이면 ≈4.5h).
#
#  사용:
#      bash scripts/launch_replay_ckpt800.sh                    # replay 만
#      ARMS="replay lr" bash scripts/launch_replay_ckpt800.sh   # 두 arm
#      DRY=1 ARMS="replay lr seed none" bash scripts/launch_replay_ckpt800.sh   # 제출 안 함
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

SRC_CKPT="${SRC_CKPT:-$CKPT_DIR/grpo_expanded_gdpo/v1-20260803-074645/checkpoint-800}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/stage2_expanded_train.jsonl}"
INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}"
STOP_STEP="${STOP_STEP:-960}"          # 800 + 160 — 붕괴(899~904)를 60 step 넘겨 확인
WALLTIME="${WALLTIME:-12:00:00}"
ARMS="${ARMS:-replay}"
STAMP="${STAMP:-$(date +%m%d-%H%M)}"

[[ -f "$SRC_CKPT/trainer_state.json" ]] || { echo "❌ 체크포인트 없음: $SRC_CKPT"; exit 1; }
[[ -f "$DATASET_FILE" ]]               || { echo "❌ 데이터셋 없음: $DATASET_FILE"; exit 1; }
[[ -f "$INIT_MODEL/config.json" ]]     || { echo "❌ INIT 병합본 없음: $INIT_MODEL"; exit 1; }

echo "[replay] SRC   = $SRC_CKPT"
echo "[replay] DATA  = $DATASET_FILE ($(wc -l <"$DATASET_FILE") 건)"
echo "[replay] STOP  = step $STOP_STEP · walltime $WALLTIME · arms: $ARMS"
echo

for arm in $ARMS; do
  # arm 별 변경점만 정의 — 나머지는 21 의 기본값(=원본과 동일)을 그대로 탄다.
  case "$arm" in
    replay) SR=gdpo; MS=2337; SD="" ;;
    lr)     SR=gdpo; MS=1200; SD="" ;;
    seed)   SR=gdpo; MS=2337; SD=1234 ;;
    none)   SR=none; MS=2337; SD="" ;;
    *) echo "❌ 알 수 없는 arm: $arm (replay|lr|seed|none)"; exit 1 ;;
  esac

  OUT="$CKPT_DIR/replay_ck800_${arm}_${STAMP}"      # ← 원본과 분리. 21 이 동일경로면 거부한다.

  #  --export 값에 **쉼표도 공백도 넣지 않는다.** SLURM 은 쉼표로 쪼개고, 공백은 버전에 따라
  #  깨진다. 그래서 seed 를 EXTRA_ARGS="--seed 1234" 로 넘기지 않고 SEED=1234 전용 변수로 뺐다
  #  (21 쪽에서 ${SEED:+--seed $SEED} 로 조립). 빈 값이면 인자 자체가 안 붙는다.
  #  KL_BASELINE_VALUE: 재개라 앞 150 step 이력이 없다 → 원본의 step 700~849 평탄값을 준다.
  SUBMIT=(sbatch --job-name="rp-$arm" --time="$WALLTIME"
    --export=ALL,RECIPE=dr_grpo,SCALE_REWARDS="$SR",MAX_STEPS="$MS",RESUME_CKPT="$SRC_CKPT",OUTPUT_DIR="$OUT",INIT_MODEL="$INIT_MODEL",DATASET_FILE="$DATASET_FILE",WATCHDOG=1,STOP_STEP="$STOP_STEP",KL_BASELINE_VALUE=0.031,SEED="$SD"
    "$PROJ_DIR/scripts/21_rlvr_grpo_adv.slurm")

  printf '[replay] %-7s scale_rewards=%-5s max_steps=%-5s seed=%s\n' \
         "$arm" "$SR" "$MS" "${SD:-기본(42)}"
  echo   "         out= $OUT"
  if [[ "${DRY:-0}" == "1" ]]; then
    echo "         (DRY) ${SUBMIT[*]}"
  else
    JID=$("${SUBMIT[@]}" | awk '{print $NF}')
    echo "         제출: job $JID  ·  판정 = logs/verdict_${JID}.json"
  fi
  echo
done

cat <<'EOF'
[replay] 모니터:
    squeue -u $USER
    tail -f logs/grpo_adv_<JID>.log | grep -E "watch|global_step"
    cat logs/verdict_<JID>.json        # COLLAPSE 면 step 이 개시 시점

[replay] 판독:
    COLLAPSE @ ~900  → 재현됨. 체계적 원인 → arm lr 로 진행
    COLLAPSE @ 딴 곳 → 상태 의존. step 번호 자체는 우연
    DONE (붕괴 없음) → 저확률 사건. replay 재실행으로 N=2 확인
EOF
