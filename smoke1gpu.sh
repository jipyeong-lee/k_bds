#!/bin/bash
# smoke1gpu.sh — debug-1gpu 에서 Stage-2 GRPO 배선만 검증(1 GPU, max_steps 2).
#   목적: 데이터로딩→vLLM롤아웃→보상(accuracy_mix/format_think/soft_overlong)
#         →GDPO 손실→저장 경로가 loader 모드에서 끝까지 도는지 확인.
#   ⚠️ 성능/수렴 검증 아님. 8GPU 본실행 파라미터와 다름(메모리 맞춤).
cd /home01/k252a02/kbds_project
export NPROC_PER_NODE=1          # 1 GPU
export NUM_GEN=2                 # 그룹크기 축소(A100 40GB 1장)
export MAX_COMPLETION=512        # 롤아웃 길이 대폭 축소(속도)
export MAX_LEN=2048
export VLLM_UTIL=0.52
source scripts/00_common.sh
echo "=== node=$(hostname) ENV_MODE=$ENV_MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2

run_py swift rlhf \
  --rlhf_type grpo \
  --model "$CKPT_DIR/sft_mixed_merged" \
  --model_type "$MODEL_TYPE" \
  --system "$SYSTEM_PROMPT" \
  --dataset "$DATA_DIR/stage2_expanded_train.jsonl" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ_DIR/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong \
  --reward_weights 1.0 0.2 0.2 \
  --soft_max_length "$MAX_COMPLETION" --soft_cache_length 256 \
  --dynamic_sample true --max_resample_times 1 --overlong_filter true \
  --loss_type dr_grpo --importance_sampling_level token --epsilon 0.2 \
  --scale_rewards gdpo \
  --enable_thinking true \
  --num_generations "$NUM_GEN" --temperature 0.9 \
  --max_completion_length "$MAX_COMPLETION" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 --beta 0.04 \
  --max_length "$MAX_LEN" --max_pixels "$MAX_PIXELS" \
  --gradient_checkpointing true --attn_impl "$ATTN_IMPL" \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  --vllm_mm_processor_cache_gb 0 --vllm_gpu_memory_utilization "$VLLM_UTIL" \
  --vllm_max_model_len "$MAX_LEN" \
  --sleep_level 1 --log_completions true \
  --save_strategy no --logging_steps 1 \
  --output_dir "$CKPT_DIR/_smoke1gpu" \
  --report_to none \
  --max_steps 2
echo "SMOKE_EXIT=$?"
