#!/bin/bash
# =============================================================================
# 20_rlvr_grpo.slurm — [2단계] 범용 멀티모달 RLVR/GRPO 훈련
#   데이터: DeepVision-103K (검증가능 정답 → 규칙기반 보상)
#   계획서 산정: 1노드 × 70h × 8GPU × 6사이클(ablation) = 3,360 노드시간
#   GRPO: 정책/참조 모델 + vLLM rollout 동시 적재 → VRAM 압박 큼.
#         8gpu=A100 80GB라 7B는 server 모드(GPU 분리)로 안정. colocate도 가능.
#   ※ ablation 6사이클을 동시 제출하면 8gpu 동시제출 한도(6개)에 도달.
# =============================================================================
#SBATCH --job-name=kbds-grpo
#SBATCH --partition=8gpu          # A100 80GB ×8 — GRPO에 충분
#SBATCH --comment=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=70:00:00           # 5일(120h) 한도 내. 사이클당 70h 예상
#SBATCH --output=/home01/k252a01/kbds_project/logs/grpo_%j.log
#SBATCH --error=/home01/k252a01/kbds_project/logs/grpo_%j.log

set -euo pipefail
source /home01/k252a01/kbds_project/scripts/00_common.sh
# 호스트 모듈은 conda 폴백 시에만 필요(컨테이너는 자체 cuda/gcc 보유)
if [[ "$ENV_MODE" == conda ]]; then module load compilers/cuda/12.4 compilers/gcc/10.2.0; fi

# 베이스(또는 cold-start 병합 모델)에서 GRPO. cold-start LoRA 병합본을 INIT_MODEL 로 지정 권장.
INIT_MODEL="${INIT_MODEL:-$BASE_MODEL}"
DATASET_FILE="${DATASET_FILE:-$DATA_DIR/deepvision103k_train.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # 스모크 테스트: EXTRA_ARGS="--max_steps 3"

# ⚠️ 이 클러스터는 NVLink 없음(PCIe A100, NCCL→SHM) → full-FT 멀티GPU 통신병목(375~660s/step).
#   LoRA 로 통신량을 줄여 우회(권장 기본값). full 은 INIT 시 TUNER_TYPE=full 로 override(느림).
TUNER_TYPE="${TUNER_TYPE:-lora}"        # lora | full
EXTRA_TUNER=""
DS_ARG="--deepspeed $PROJ_DIR/configs/ds_zero3_offload.json"   # full 용(옵티마이저 CPU offload)
if [[ "$TUNER_TYPE" == lora ]]; then
  EXTRA_TUNER="--lora_rank ${LORA_RANK:-16} --lora_alpha ${LORA_ALPHA:-32} --target_modules all-linear"
  LR="${LR:-1e-5}"                       # LoRA 는 더 높은 LR(GRPO 기준)
  DS_ARG=""                              # LoRA 기본: base 동결 → DDP(통신 최소), DeepSpeed 불필요
  # 길이 확대용 ZeRO-3: 18GB base 를 샤딩해 학습-phase 메모리 확보 → max_completion 8192 가능.
  #  ms-swift 72B LoRA VLM 예시 레시피(zero3 + offload + move_model_batches). no-NVLink 라 param
  #  all-gather 통신비용 있음 → 속도 측정 필수. USE_ZERO3=1 로 활성화.
  if [[ "${USE_ZERO3:-0}" == 1 ]]; then
    DS_ARG="--deepspeed zero3 --offload_optimizer true --offload_model true --move_model_batches ${MOVE_BATCHES:-40}"
  fi
else
  LR="${LR:-1e-6}"                       # full-FT GRPO 는 낮은 LR
fi
# 추론모드 강제: qwen3_5 는 하이브리드 thinking. enable_thinking=true 로 롤아웃 추론 템플릿 보장 +
#   format_think 보상(빈 <think></think> 는 0점)으로 비추론 지름길 reward-hacking 차단.
# 길이 전략 = 수용(accommodate). 진단(probe_coldstart_infer)으로 base Qwen3.5-VL 이 어려운
#   시각/기하 문제에 본래 3.5~4.6K토큰 길게 추론함을 확인(enable_thinking on/off 무관, 콜드스타트
#   clevr_math 760자가 일반화 안 됨). 잘림=무답=0점 순손해라, budget 을 늘려 "마감"되게 함:
#   max_completion 6144 + max_length 10240(prompt~2k+6k). num_generations 8→4 로 메모리/속도 상쇄.
#   soft_overlong(soft_max 6144/cache 2048): 0~4096 무페널티, 4096~6144 완만 nudge("닫을 수 있으면
#   닫아라"). 6144 도 부족(여전히 다량 잘림)하면 8192 로 상향 검토.
# vLLM colocate: vLLM이 학습 GPU 공유, sleep_level로 교대. LoRA 는 full 옵티마이저 없어 메모리 여유.

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$INIT_MODEL" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATASET_FILE" \
  --tuner_type "$TUNER_TYPE" $EXTRA_TUNER \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong \
  --reward_weights 1.0 0.2 0.2 \
  --soft_max_length "${MAX_COMPLETION:-6144}" \
  --soft_cache_length 2048 \
  --enable_thinking true \
  --num_generations "${NUM_GEN:-4}" \
  --max_completion_length "${MAX_COMPLETION:-6144}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "$LR" \
  --max_length "${MAX_LEN:-10240}" \
  --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true \
  $DS_ARG \
  --attn_impl "$ATTN_IMPL" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP:-1}" \
  --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.4}" \
  --sleep_level 1 \
  --log_completions true \
  --save_strategy steps --save_steps 50 --save_total_limit 3 \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR:-$CKPT_DIR/grpo_general}" \
  --report_to tensorboard \
  $EXTRA_ARGS

echo "[grpo] done -> $CKPT_DIR/grpo_general"
