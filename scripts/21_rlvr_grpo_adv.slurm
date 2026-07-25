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
#  ▶ plateau 직접 돌파(이어학습): INIT_MODEL=<...checkpoint-1000> 로 override (싼 검증, A/B 아님)
# =============================================================================
#SBATCH --job-name=grpo-adv
#SBATCH --partition=8gpu
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00
#SBATCH --output=/home01/k252a02/kbds_project/logs/grpo_adv_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/grpo_adv_%j.log

set -euo pipefail
source /home01/k252a02/kbds_project/scripts/00_common.sh
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# ---- baseline 과 동일한 init/데이터 (clean A/B) ----------------------------
#  기본 = RFT 콜드스타트 병합본(= 57249 의 init). 이어학습 비교를 원하면 INIT_MODEL 로 ckpt 지정.
INIT_MODEL="${INIT_MODEL:-$CKPT_DIR/sft_rft_coldstart_merged}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/deepvision103k_train.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
CORE_ARGS="--dynamic_sample true --max_resample_times $DS_RESAMPLE --overlong_filter true"

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
  *)
    echo "[grpo-adv] 알 수 없는 RECIPE=$RECIPE (dapo|gspo|dr_grpo)"; exit 1;;
esac

echo "[grpo-adv] RECIPE=$RECIPE  INIT_MODEL=$INIT_MODEL"
echo "[grpo-adv] knobs: NUM_GEN=${NUM_GEN:-4}  BETA=${BETA:-0.04}  TEMPERATURE=$TEMPERATURE  LR=$LR  (기본=검증값, 변경 시 clean A/B 깨짐 — docs/rlvr_hparams_external.md)"
echo "[grpo-adv] CORE=$CORE_ARGS"
echo "[grpo-adv] RECIPE_ARGS=$RECIPE_ARGS"

OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_DIR/grpo_general_adv_$RECIPE}"

# ---- resume / max_steps (1-epoch 이어학습 체인용) --------------------------
#  RESUME=1 → OUTPUT_DIR 하위 모든 버전에서 최신 checkpoint-N 자동 resume.
#  MAX_STEPS=<n> → 그 step 에서 정지(예 1 epoch=3235). 체인 잡들이 max_steps 도달 시 자동 no-op.
RESUME_ARGS=""
if [[ "${RESUME:-0}" == "1" ]]; then
  LATEST=$(ls -d "$OUTPUT_DIR"/*/checkpoint-* 2>/dev/null \
    | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)
  if [[ -n "$LATEST" ]]; then
    RESUME_ARGS="--resume_from_checkpoint $LATEST"; echo "[grpo-adv] RESUME from: $LATEST"
  else
    echo "[grpo-adv] RESUME=1 이나 체크포인트 없음 → 신규 시작"
  fi
fi
[[ -n "${MAX_STEPS:-}" ]] && RESUME_ARGS="$RESUME_ARGS --max_steps $MAX_STEPS" && echo "[grpo-adv] MAX_STEPS=$MAX_STEPS"

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATASET_FILE" \
  --tuner_type lora \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --target_modules all-linear \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong \
  --reward_weights 1.0 0.2 0.2 \
  --soft_max_length "${MAX_COMPLETION:-6144}" \
  --soft_cache_length 2048 \
  $CORE_ARGS \
  $RECIPE_ARGS \
  --enable_thinking true \
  --num_generations "${NUM_GEN:-4}" \
  --temperature "$TEMPERATURE" \
  --max_completion_length "${MAX_COMPLETION:-6144}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "$LR" \
  --beta "${BETA:-0.04}" \
  --max_length "${MAX_LEN:-10240}" \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP:-1}" \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.4}" \
  --sleep_level 1 \
  --log_completions true \
  --log_entropy true \
  --save_strategy steps --save_steps 50 --save_total_limit 5 \
  --logging_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard \
  $RESUME_ARGS \
  $EXTRA_ARGS

echo "[grpo-adv] done ($RECIPE) -> $OUTPUT_DIR"
