#!/bin/bash
# =============================================================================
# 21_rlvr_grpo_adv.slurm — [2단계-실험] GRPO 파생기법 A/B (plateau 돌파용)
# -----------------------------------------------------------------------------
#  목적: baseline(job 57249, step1000)에서 진단된 Acc plateau의 원인
#        — frac_reward_zero_std 0.24→0.33 상승(그룹 내 분산=0 → 정확도 gradient 소실) —
#        을 GRPO 파생기법으로 직접 겨냥해 돌파 여부를 검증.
#
#  설계원칙: baseline 과 "기법 외 모든 조건 동일"(clean A/B).
#    동일하게 유지 → init(sft_rft_coldstart_merged), 보상(accuracy_mix/format_think/soft_overlong
#    =1.0/0.2/0.2), num_gen 4, max_completion 6144, soft_max/cache 6144/2048, lr 1e-5, beta 0.04,
#    LoRA rank16/alpha32 all-linear, flash_attn, vllm colocate.
#    변경 → 아래 RECIPE 가 토글하는 GRPO 파생기법만.
#
#  RECIPE (RECIPE 환경변수로 선택):
#   - dapo (기본): 토큰레벨 일관 레시피
#       · dynamic_sample(zero_std 그룹 폐기·재샘플) ← plateau 직격 핵심
#       · clip-higher (epsilon 0.2 / epsilon_high 0.28) — 탐색 여지 확대(엔트로피 붕괴 억제)
#       · overlong_filter — 잘린(=무답) 롤아웃을 loss 에서 제외(신호 오염 차단)
#       · loss_type=dapo (토큰레벨, 길이정규화 편향 제거)
#   - gspo: 시퀀스레벨 일관 레시피 (GSPO, 2507.18071)
#       · importance_sampling_level=sequence + 시퀀스레벨에 맞는 작은 clip(3e-4/4e-4)
#       · dynamic_sample + overlong_filter (직교 기법이라 공통 적용)
#       · LoRA+멀티모달에서 토큰레벨보다 안정적이라는 보고 → 안정성 비교용
#   - dr_grpo: Dr.GRPO (2503.20783) — DAPO 진단(길이 폭주→clip↑→0점) 직격
#       · loss_type=dr_grpo (길이정규화 편향 제거) + scale_rewards=none (std 난이도 편향 제거)
#       · dynamic_sample + overlong_filter 공통 적용. clip-higher 미적용(대칭 ε 0.2)
#
#  주의: dynamic_sample 은 재샘플(최대 max_resample_times)로 롤아웃 비용↑ → step time 증가 가능.
#        overlong_filter 는 잘린 샘플을 loss 에서 빼므로 유효 배치가 작아짐(의도된 DAPO 동작).
#
#  ▶ 빠른 검증(스모크):  EXTRA_ARGS="--max_steps 5"
#  ▶ **배선 검증(1 GPU, 40GB)**: 8gpu 파티션이 며칠씩 막힐 때 debug-1gpu(유휴 2노드, 8h)로
#     "돌아가는지"만 본다. step time 은 못 재지만 보상 플러그인·recipe 인자·도메인 프리셋은 검증된다.
#       scripts/probe_1gpu.sh
#     A100-40GB 라 9B + vLLM colocate 가 그냥은 안 들어간다(학습 18GB + vLLM 18GB = 36GB).
#     → OFFLOAD=1 (롤아웃 중 학습가중치/옵티마 CPU 로) + SLEEP_LEVEL=2 (학습 중 vLLM 가중치까지 해제)
#       로 두 사본이 GPU 에 동시에 있지 않게 만든다. 매 step 18GB 재동기라 느리지만 3 step 이면 족하다.
#  ▶ plateau 직접 돌파(이어학습): INIT_MODEL=<...checkpoint-1000> 로 override (싼 검증, A/B 아님)
# =============================================================================
#SBATCH --job-name=grpo-adv
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00
#SBATCH --output=/home01/k266a01/kbds_project/logs/grpo_adv_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/grpo_adv_%j.log

set -euo pipefail
source /home01/k266a01/kbds_project/scripts/00_common.sh
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# ---- baseline 과 동일한 init/데이터 (clean A/B) ----------------------------
#  기본 = RFT 콜드스타트 병합본(= 57249 의 init). 이어학습 비교를 원하면 INIT_MODEL 로 ckpt 지정.
INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_rft_coldstart_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/deepvision103k_train.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- 도메인 전문가 프리셋 (DOMAIN=deepvision|mmk12|pmcvqa) ------------------
#  DeepSeek-V4 구조: 도메인별 전문가를 따로 키우고 나중에 통합한다.
#  → docs/deepseek_v4_pipeline_adoption.md
#  데이터는 scripts/split_stage2_by_source.py 산출물.
#
#  ⚠️ **도메인별로 다른 값을 주는 게 이 구조의 핵심 이득이다.** 혼합 학습에서는 불가능했다.
#     SOFT_MAX  = 길이 예산. 의료는 객관식 letter 1자라 6144 가 과하다. 수학은 긴 CoT 가 필요.
#     NUM_GEN   = 그룹 크기. 붕괴 전 정확도 균일률 실측 = deepvision 13.5 / mmk12 25.0 / pmcvqa 3.7%.
#                 → 원래는 mmk12 만 올릴 근거였으나 전 도메인 8 로 통일(운영 단순화).
#
#  🚨 **num_generations 는 계산이 아니라 데이터 노출을 깎는다.** 배치는 32 completion 고정이므로
#     프롬프트/step = 32 ÷ num_generations 다. n=4 → 8, n=8 → 4, n=16 → 2.
#     1,000 step 기준 pmcvqa 노출: n=4 0.41 epoch(혼합의 4.5배) / n=8 0.20(2.2배) / n=16 0.10(1.1배).
#     **n=16 이면 도메인 분리의 이득이 통째로 사라진다.** 같은 노출을 유지하려면 계산이 2·4배 들고
#     잔여 예산(4,126 GPU-h)에 안 들어간다. → **8 을 상한으로 본다.**
if [[ -n "${DOMAIN:-}" ]]; then
#  길이 예산은 붕괴 전(751-898) 소스별 실측에 맞췄다. 문자수(≈2.54 자/토큰):
#      소스          평균    중앙    p95     최대    → 추정 토큰 p95 / 최대
#      deepvision   3,919  2,183  14,161  21,964      5,575 / 8,647 (6144 에서 잘림)
#      mmk12        3,458  2,105  12,980  19,749      5,110 / 7,775 (잘림)
#      pmcvqa       1,749  1,722   2,224   3,662        875 / 1,441 ← **6144 근처도 안 간다**
#  → pmcvqa 만 3072 로 내린다. 관측 최대의 2배라 안전하고, vLLM KV 예산이 절반이라 롤아웃이 빨라진다.
#  soft_cache 는 1/3 비율 유지 → 벌점 시작점 = soft_max − soft_cache.
#     deepvision·mmk12 4096 (p95 5,575 아래라 정상 구간엔 거의 안 걸림)
#     pmcvqa           2048 (p95 875 의 2.3배)
  case "$DOMAIN" in
    deepvision) SOFT_MAX_D=6144; SOFT_CACHE_D=2048 ;;   # 시각논리 — 현행 유지(6144 에서 잘림 발생)
    mmk12)      SOFT_MAX_D=6144; SOFT_CACHE_D=2048 ;;   # 수학 — 긴 CoT 필요, 줄이지 않는다
    pmcvqa)     SOFT_MAX_D=3072; SOFT_CACHE_D=1024 ;;   # 의료 객관식 — 실측 최대 1,441 토큰
    *) echo "[grpo-adv] ❌ 알 수 없는 DOMAIN: $DOMAIN (deepvision|mmk12|pmcvqa)"; exit 1 ;;
  esac
  DATASET_FILE="${DATASET_FILE_OVERRIDE:-$DATA_DIR/domains/stage2_${DOMAIN}.jsonl}"
  [[ -f "$DATASET_FILE" ]] || {
    echo "[grpo-adv] ❌ 도메인 데이터 없음: $DATASET_FILE"
    echo "           먼저: python3 scripts/split_stage2_by_source.py"; exit 1; }
  MAX_COMPLETION="${MAX_COMPLETION:-$SOFT_MAX_D}"
  SOFT_MAX="${SOFT_MAX:-$SOFT_MAX_D}"
  SOFT_CACHE="${SOFT_CACHE:-$SOFT_CACHE_D}"
  NUM_GEN="${NUM_GEN:-8}"
  NDATA=$(wc -l <"$DATASET_FILE")
  #  생성배치를 32 로 박아두면 안 된다 — PDTBS/ACCUM/world 를 바꾸면 거짓말이 된다.
  #  (1gpu 검증에서 실제 1 프롬프트/step 인데 "4" 로 찍혔다.)
  GEN_BATCH=$(( ${PDTBS:-1} * ${ACCUM:-4} * ${NPROC_PER_NODE:-8} ))
  PPS=$(( GEN_BATCH / NUM_GEN ))
  (( GEN_BATCH % NUM_GEN == 0 )) || {
    echo "[grpo-adv] ❌ 생성배치 $GEN_BATCH 가 num_generations $NUM_GEN 로 나누어떨어지지 않는다."
    echo "           PDTBS×ACCUM×world = ${PDTBS:-1}×${ACCUM:-4}×${NPROC_PER_NODE:-8}"; exit 1; }
  echo "[grpo-adv] DOMAIN=$DOMAIN  data=$(basename "$DATASET_FILE") ($NDATA 건)"
  echo "[grpo-adv]   soft_max=$SOFT_MAX  soft_cache=$SOFT_CACHE (벌점 시작 $((SOFT_MAX - SOFT_CACHE)))"
  echo "[grpo-adv]   num_gen=$NUM_GEN → 프롬프트/step=$PPS · 1 epoch=$((NDATA / PPS)) step"
fi
SOFT_MAX="${SOFT_MAX:-${MAX_COMPLETION:-6144}}"
SOFT_CACHE="${SOFT_CACHE:-2048}"

# ---- LoRA (baseline 과 동일: no-NVLink → DDP 통신최소) ----------------------
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-1e-5}"

# ---- 롤아웃 temperature (검증된 A/B 기본값 = ms-swift 기본 0.9) -------------
#   외부 관행 다수설은 ~1.0(ORZ "T=1 best"). 기본은 재현성 위해 0.9 유지하고
#   TEMPERATURE=1.0 로 A/B. 상세 = docs/rlvr_hparams_external.md §1,§3.
TEMPERATURE="${TEMPERATURE:-0.9}"
# ── 외부 관행 대조 A/B knob (전부 기본=검증값, override 시에만 변경) ──────────
#   그룹크기:  NUM_GEN=8      (관행 8~16, 기본 4는 메모리 절충)
#   KL 페널티: BETA=0.01      (관행은 0~0.01, 기본 0.04는 보수적)
#   temp:      TEMPERATURE=1.0
#   → 근거·판정표: docs/rlvr_hparams_external.md

# ---- RECIPE 별 GRPO 파생기법 인자 조립 -------------------------------------
RECIPE="${RECIPE:-dapo}"
# 공통 코어: dynamic_sample + overlong_filter (plateau 직격)
DS_RESAMPLE="${MAX_RESAMPLE:-3}"
#  overlong_filter 는 recipe=stable 에서 false 다 — 잘린 샘플을 completion_mask 에서 빼는데
#  그 마스크가 KL 계산에도 쓰여(grpo_trainer.py:1132 → 1140) 회복을 막고 KL 계측까지 오염시킨다.
#  → 사후분석 §4-4
CORE_ARGS="--dynamic_sample true --max_resample_times $DS_RESAMPLE --overlong_filter ${OVERLONG_FILTER:-true}"

case "$RECIPE" in
  dapo)
    # 토큰레벨 DAPO: clip-higher + dapo loss
    RECIPE_ARGS="--loss_type dapo \
      --epsilon ${EPS:-0.2} --epsilon_high ${EPS_HIGH:-0.28} \
      --importance_sampling_level token \
      --scale_rewards ${SCALE_REWARDS:-group}"
    ;;
  gspo)
    # 시퀀스레벨 GSPO: 작은 clip(시퀀스평균 비율이라 token clip 보다 훨씬 작아야 함)
    RECIPE_ARGS="--loss_type grpo \
      --importance_sampling_level sequence \
      --epsilon ${EPS:-3e-4} --epsilon_high ${EPS_HIGH:-4e-4} \
      --scale_rewards ${SCALE_REWARDS:-group}"
    ;;
  dr_grpo)
    # Dr.GRPO (2503.20783) — 두 편향 제거로 "길이 폭주→clip→0점" 직격:
    #   ① loss_type=dr_grpo : 손실을 시퀀스길이가 아닌 상수(batch×max_completion)로 정규화
    #      → 긴 응답에 유리하던 길이정규화 편향 제거 (소스: per_token_loss.sum()/(B*max_len))
    #   ② scale_rewards=none : advantage 를 그룹 std 로 나누지 않음 → 난이도(쉬운 문제 과대가중) 편향 제거
    #   clip-higher 미적용(그건 DAPO 고유) — 대칭 epsilon 0.2 로 dr_grpo 효과만 분리.
    RECIPE_ARGS="--loss_type dr_grpo \
      --importance_sampling_level token \
      --epsilon ${EPS:-0.2} \
      --scale_rewards ${SCALE_REWARDS:-none}"
    ;;
  stable)
    # 2026 재설계 recipe — 붕괴 재발 억제. dr_grpo 를 토대로 세 가지를 바꾼다.
    #   ① overlong_filter=false : 회복 차단 요인 제거 (사후분석 §4-4, 자체 실측)
    #   ② scale_rewards=none    : std 정규화 생략. EXAONE 4.5 가 "안정성 보존" 이유로 명시
    #   ③ log_rollout_offpolicy_metrics : 학습–추론 확률 불일치 **계측만** 켠다.
    #      IcePop 은 확률차 5% 초과 시 학습 실패라고 보고한다(GRPO 는 180~200 step 붕괴).
    #      우리 vLLM colocate 구성도 같은 계열 노출인데 **한 번도 잰 적이 없다.**
    #      ⚠️ 보정(rollout_importance_sampling_mode)은 켜지 않는다 — 먼저 재고, 임계를
    #         넘는 게 확인되면 그때 token_mask 로 켠다. 같이 켜면 무엇 덕분인지 모른다.
    #   loss_type 은 dr_grpo 유지(길이정규화 편향 제거). sapo 는 grpo 계열 시퀀스 정규화를
    #   쓰므로 그 편향이 되돌아와 보류했다 → docs/stage2_redesign_2026.md §3-I
    OVERLONG_FILTER=false
    CORE_ARGS="--dynamic_sample true --max_resample_times $DS_RESAMPLE --overlong_filter false"
    RECIPE_ARGS="--loss_type dr_grpo \
      --importance_sampling_level token \
      --epsilon ${EPS:-0.2} \
      --scale_rewards ${SCALE_REWARDS:-none} \
      --log_rollout_offpolicy_metrics true"
    ;;
  *)
    echo "[grpo-adv] 알 수 없는 RECIPE=$RECIPE (dapo|gspo|dr_grpo|stable)"; exit 1;;
esac

echo "[grpo-adv] RECIPE=$RECIPE  INIT_MODEL=$INIT_MODEL"
echo "[grpo-adv] knobs: NUM_GEN=${NUM_GEN:-4}  BETA=${BETA:-0.04}  TEMPERATURE=$TEMPERATURE  LR=$LR  (기본=검증값, 변경 시 clean A/B 깨짐 — docs/rlvr_hparams_external.md)"
echo "[grpo-adv] CORE=$CORE_ARGS"
echo "[grpo-adv] RECIPE_ARGS=$RECIPE_ARGS"

OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/grpo_general_adv_$RECIPE}"

# ---- resume / max_steps (1-epoch 이어학습 체인용) --------------------------
#  RESUME=1 → OUTPUT_DIR 하위 모든 버전에서 최신 checkpoint-N 자동 resume.
#  MAX_STEPS=<n> → 그 step 에서 정지. 체인 잡들이 max_steps 도달 시 자동 no-op.
#    ⚠️ step 은 epoch 이 아니다: 프롬프트/step = pdtbs×world×accum ÷ num_generations = 8.
#       확장셋(74,787) 기준 1 epoch = 9,348 step. 상세 = docs/stage2_run73924_progress.md §3
#  RESUME_CKPT=<경로> → 그 체크포인트에서 명시적으로 재개(RESUME=1 자동탐색보다 우선).
#    재현 실험용. 반드시 **새 OUTPUT_DIR** 과 함께 쓸 것 — 원본 디렉터리에 그대로 재개하면
#    save_total_limit=5 때문에 새 저장이 원본 checkpoint-800/850 을 밀어낸다.
#    (73924 에서 최고 체크포인트 850 이 step 1100 저장에 삭제될 예정이었다 — §1-4)
# ---- 배치 (PDTBS/ACCUM) ----------------------------------------------------
#  🚨 GRPO 에서 배치 단위는 **프롬프트가 아니라 completion** 이다.
#     생성배치 = PDTBS × ACCUM × world  이고 이 값이 num_generations 로 나누어떨어져야 한다.
#     8gpu 기본:  1 × 4 × 8 = 32 completion → num_gen=8 이면 프롬프트/step = 4.
#     1gpu 축소:  world=1 이라 1 × 4 × 1 = 4 < 8 → **ms-swift 가 거절한다.**
#                 → ACCUM=8 로 8 completion = 1 프롬프트/step 로 맞춘다.
#     이 두 값을 env 로 뺀 이유는 노드 크기를 바꿔가며 같은 스크립트로 검증하기 위해서다.
#     (별도 축소본 스크립트를 만들면 정작 검증하려던 본 스크립트가 검증되지 않는다.)
RESUME_ARGS=""
if [[ -n "${RESUME_CKPT:-}" ]]; then
  [[ -f "$RESUME_CKPT/trainer_state.json" ]] || { echo "❌ RESUME_CKPT 아님: $RESUME_CKPT"; exit 1; }
  SRC_RUN=$(dirname "$(dirname "$RESUME_CKPT")")
  if [[ "$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)" == "$(cd "$SRC_RUN" && pwd)" ]]; then
    echo "❌ OUTPUT_DIR 이 원본과 같다 — 새 저장이 원본 체크포인트를 삭제한다."
    echo "   OUTPUT_DIR 을 별도 경로로 지정할 것: $OUTPUT_DIR"
    exit 1
  fi
  RESUME_ARGS="--resume_from_checkpoint $RESUME_CKPT"
  echo "[grpo-adv] RESUME_CKPT(명시): $RESUME_CKPT  →  새 OUTPUT_DIR=$OUTPUT_DIR"
elif [[ "${RESUME:-0}" == "1" ]]; then
  LATEST=$(ls -d "$OUTPUT_DIR"/*/checkpoint-* 2>/dev/null \
    | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-) || true
  if [[ -n "$LATEST" ]]; then
    RESUME_ARGS="--resume_from_checkpoint $LATEST"; echo "[grpo-adv] RESUME from: $LATEST"
  else
    echo "[grpo-adv] RESUME=1 이나 체크포인트 없음 → 신규 시작"
  fi
fi
[[ -n "${MAX_STEPS:-}" ]] && RESUME_ARGS="$RESUME_ARGS --max_steps $MAX_STEPS" && echo "[grpo-adv] MAX_STEPS=$MAX_STEPS"

# ---- 형식 붕괴 감시자 (WATCHDOG=1) ----------------------------------------
#  73924 는 step ~900 붕괴 후 13h35m 을 더 돌았다(109 GPU-h 낭비). 감시 장치가 없었다.
#  임계값은 73924/73925 로그 재생 스윕으로 고정했다 — scripts/watch_format_collapse.py 주석 참조.
#  STOP_STEP 을 주면 그 step 에서도 정상 종료한다(재현 실험의 상한).
#  ⚠️ 호스트 python3 로 띄운다 — 표준 라이브러리만 쓰므로 컨테이너/로더 진입이 필요 없고,
#     학습이 GPU 를 다 쓰는 동안에도 독립적으로 돈다.
WATCH_PID=""
if [[ "${WATCHDOG:-0}" == "1" ]]; then
  WLOG="$PROJ_DIR/logs/grpo_adv_${SLURM_JOB_ID}.log"
  WVERDICT="$PROJ_DIR/logs/verdict_${SLURM_JOB_ID}.json"
  python3 "$PROJ_DIR/scripts/watch_format_collapse.py" \
    --log "$WLOG" --verdict "$WVERDICT" --job-id "$SLURM_JOB_ID" \
    ${STOP_STEP:+--stop-step "$STOP_STEP"} \
    ${KL_BASELINE_VALUE:+--kl-baseline-value "$KL_BASELINE_VALUE"} \
    --poll "${WATCH_POLL:-120}" &
  WATCH_PID=$!
  echo "[grpo-adv] WATCHDOG pid=$WATCH_PID → $WVERDICT (stop_step=${STOP_STEP:-없음})"
  trap '[[ -n "$WATCH_PID" ]] && kill "$WATCH_PID" 2>/dev/null || true' EXIT
fi

# ---- vLLM 실행 모드 (VLLM_MODE=colocate|server) -----------------------------
#  server: 롤아웃을 전용 GPU 1장으로 분리해 async_generate 로 학습과 오버랩한다.
#          B200 deepvision 에서 검증한 레시피를 그대로 이식(CLAUDE.md §1.5).
#  colocate(기본, 미지정 시): 학습·롤아웃이 같은 GPU 를 순차로 쓴다 — 지금까지 KISTI 기본값.
#  KISTI 2gpu 스모크로 메커니즘 검증 완료(job 76198, 2026-09-01) — 롤아웃 서버 기동·헬스체크·
#  async 접속·학습 step 진행까지 정상. 그 스모크의 OOM 은 2GPU/40GB 로 욱여넣은 탓(world=1,
#  9B 모델) — 실제 8gpu(80GB×6~7) 규모에서는 문제 없을 배치 크기다.
VLLM_MODE="${VLLM_MODE:-colocate}"
if [[ "$VLLM_MODE" == "server" ]]; then
  SERVER_GPU=$((NPROC_PER_NODE - 1))
  TRAIN_GPUS=$(seq -s, 0 $((SERVER_GPU - 1)))
  ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"
  RLOG="$PROJ_DIR/logs/rollout_${SLURM_JOB_ID}.log"
  echo "[grpo-adv] VLLM_MODE=server → GPU $SERVER_GPU 에 롤아웃 서버, GPU $TRAIN_GPUS 로 학습"
  (
    export CUDA_VISIBLE_DEVICES=$SERVER_GPU
    run_py swift rollout \
      --model "$INIT_MODEL" --model_type "$MODEL_TYPE" \
      --vllm_tensor_parallel_size 1 \
      --vllm_max_model_len "${VLLM_MAX_LEN:-10240}" \
      --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.90}" \
      --vllm_enable_prefix_caching true \
      --port "$ROLLOUT_PORT"
  ) > "$RLOG" 2>&1 &
  ROLLOUT_PID=$!
  sleep 2
  echo "[grpo-adv]   롤아웃 서버 기동 시도, pid=$ROLLOUT_PID (2초 후 생존: $(kill -0 "$ROLLOUT_PID" 2>/dev/null && echo yes || echo NO))"
  #  WATCHDOG 트랩과 합친다 — 여기서 새로 trap 을 걸면 위에서 건 WATCH_PID kill 이 덮어써진다.
  trap '[[ -n "${WATCH_PID:-}" ]] && kill "$WATCH_PID" 2>/dev/null; [[ -n "${ROLLOUT_PID:-}" ]] && kill "$ROLLOUT_PID" 2>/dev/null; true' EXIT
  #  포트가 TIME_WAIT 로 밀릴 수 있다 — 요청 포트가 아니라 로그의 실제 바인딩을 따라간다(B200 실측).
  for i in $(seq 1 240); do
    #  set -e + pipefail 아래서는 grep 이 아직 못 찾는 정상 상태(서버 기동 중)조차 파이프 실패로
    #  전파돼 스크립트가 조용히(에러 메시지 없이) 죽는다 — || true 로 그 전파를 끊는다.
    ACTUAL=$(grep -oE "Uvicorn running on http://[0-9.]+:[0-9]+" "$RLOG" 2>/dev/null | tail -1 | sed 's/.*://' || true)
    if [[ -n "$ACTUAL" ]] && curl -s "http://127.0.0.1:$ACTUAL/health" >/dev/null 2>&1; then
      [[ "$ACTUAL" != "$ROLLOUT_PORT" ]] && echo "[grpo-adv]   ⚠️ 포트 밀림: $ROLLOUT_PORT → $ACTUAL"
      ROLLOUT_PORT="$ACTUAL"; echo "[grpo-adv]   롤아웃 서버 health OK (${i}×5s, port=$ROLLOUT_PORT)"; break
    fi
    kill -0 "$ROLLOUT_PID" 2>/dev/null || { echo "[grpo-adv] ❌ 롤아웃 서버 사망 — 로그:"; tail -30 "$RLOG"; exit 1; }
    sleep 5
  done
  curl -s "http://127.0.0.1:$ROLLOUT_PORT/health" >/dev/null 2>&1 || {
    echo "[grpo-adv] ❌ 롤아웃 서버 미기동 — 로그:"; tail -30 "$RLOG"; exit 1; }
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  export NPROC_PER_NODE="$SERVER_GPU"
  VLLM_ARGS="--vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port $ROLLOUT_PORT \
    --async_generate true \
    --rollout_importance_sampling_mode token_truncate --rollout_importance_sampling_threshold 2.0"
else
  VLLM_ARGS="--vllm_mode colocate \
    --vllm_tensor_parallel_size ${VLLM_TP:-1} \
    --vllm_mm_processor_cache_gb 0 \
    --vllm_gpu_memory_utilization ${VLLM_UTIL:-0.4} \
    ${VLLM_MAX_LEN:+--vllm_max_model_len $VLLM_MAX_LEN} \
    --sleep_level ${SLEEP_LEVEL:-1}"
fi

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATASET_FILE" \
  --tuner_type lora \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules all-linear \
  --lora_dropout "${LORA_DROPOUT:-0}" \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong \
  --reward_weights 1.0 0.2 0.2 \
  --soft_max_length "$SOFT_MAX" \
  --soft_cache_length "$SOFT_CACHE" \
  $CORE_ARGS \
  $RECIPE_ARGS \
  --enable_thinking true \
  --num_generations "${NUM_GEN:-4}" \
  --temperature "$TEMPERATURE" \
  --top_entropy_quantile "${ENTQ:-1.0}" \
  --max_completion_length "${MAX_COMPLETION:-6144}" \
  --per_device_train_batch_size "${PDTBS:-1}" \
  --gradient_accumulation_steps "${ACCUM:-4}" \
  --learning_rate "$LR" \
  --beta "${BETA:-0.04}" \
  --max_length "${MAX_LEN:-10240}" \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true \
  $VLLM_ARGS \
  ${OFFLOAD:+--offload_model true --offload_optimizer true} \
  --log_completions true \
  --log_entropy true \
  --save_strategy steps --save_steps 50 --save_total_limit 5 \
  --logging_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard \
  ${SEED:+--seed $SEED} \
  $RESUME_ARGS \
  $EXTRA_ARGS

echo "[grpo-adv] done ($RECIPE) -> $OUTPUT_DIR"
