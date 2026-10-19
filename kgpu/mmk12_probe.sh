#!/usr/bin/env bash
# mmk12_probe.sh — kgpu pod 1장에서 mmk12 전문가를 ckpt-350 부터 이어 돌려 붕괴 여부를 본다.
# 왜: KISTI probe_mmk12_beta001(10/14)은 450 붕괴 지점을 넘겼지만 cosine 이 500 에서 0 이 되도록 잡혀
#     450 에서 lr≈2.4e-7 이었다 → beta 효과인지 lr 소멸 효과인지 분리가 안 된다. 여기선 lr 을 고정한다.
# 레시피는 probe_mmk12_beta001 의 args.json 그대로(beta 0.01·ENTQ 0.2·gdpo·temp 1.0·dr_grpo),
# 바뀐 것: lr_scheduler constant · attn sdpa(flash-attn 없음, CLAUDE.md §1.4-5) · 1 GPU 라 PDTBS×ACCUM=2×16(=KISTI 1×4×8, 32 completion/step).
# 전제(~/files = /root/files): kbds/{code.tgz, setup_venv.sh, sft_mixed_merged/, mmk12_data.tar, mmk12_ckpt350.tar}
# env(선택): MAX_STEPS(기본 550) · PDTBS · ACCUM · CKPT(기본 350, kbds/mmk12_ckpt$CKPT.tar) · BETA(기본 0.01)
#           TIS=1 → rollout_importance_sampling_mode token_truncate(2.0) — 학습↔롤아웃 불일치가 붕괴 원인인지 가리는 E1
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv; TAG=mmk12_probe_$(date +%m%d-%H%M)
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) code·base·data·ckpt → pod 로컬"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
[ -d "$W/base" ] || cp -r "$F/sft_mixed_merged" "$W/base" || { echo "❌ base copy"; exit 1; }
mkdir -p "$W/data" "$W/ckpt"
[ -f "$W/data/domains/stage2_mmk12.jsonl" ] || tar xf "$F/mmk12_data.tar" -C "$W/data" --no-same-owner || { echo "❌ data"; exit 1; }
[ -d "$W/ckpt/checkpoint-${CKPT:=350}" ] || tar xf "$F/mmk12_ckpt$CKPT.tar" -C "$W/ckpt" --no-same-owner || { echo "❌ ckpt"; exit 1; }
sed -i "s#/home01/k266a01/kbds_project/work/data/images#$W/data/images#g" "$W/data/domains/stage2_mmk12.jsonl"
head -c 300 "$W/data/domains/stage2_mmk12.jsonl"; echo

stamp "2) venv"
. "$F/setup_venv.sh"
export PYTHONPATH="$W/scripts/pyfix${PYTHONPATH:+:$PYTHONPATH}"   # rng_state·optimizer 재개 로드(weights_only) 우회
export PYTORCH_ALLOC_CONF=expandable_segments:True

stamp "3) swift rlhf — ckpt-$CKPT → ${MAX_STEPS:=550}, lr 고정, beta ${BETA:=0.01}, TIS ${TIS:=0}"
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
OUT=$W/runs/$TAG
"$VENV/bin/swift" rlhf --rlhf_type grpo \
  --model "$W/base" --model_type qwen3_5 --system "$SYS" --dataset "$W/data/domains/stage2_mmk12.jsonl" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
  --external_plugins "$W/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong --reward_weights 1.0 0.2 0.2 \
  --soft_max_length 6144 --soft_cache_length 2048 \
  --loss_type dr_grpo --epsilon 0.2 --importance_sampling_level token --scale_rewards gdpo \
  --dynamic_sample true --max_resample_times 3 --overlong_filter false \
  --beta $BETA --top_entropy_quantile 0.2 --temperature 1.0 \
  --enable_thinking true --num_generations 8 --max_completion_length 6144 \
  --per_device_train_batch_size ${PDTBS:-2} --gradient_accumulation_steps ${ACCUM:-16} \
  --learning_rate 1e-5 --lr_scheduler_type constant \
  --max_length 10240 --max_pixels 1003520 --gradient_checkpointing true --attn_impl sdpa \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization 0.4 --vllm_max_model_len 10240 --sleep_level 1 \
  --log_completions true --log_entropy true --log_rollout_offpolicy_metrics true --logging_steps 1 \
  --save_strategy steps --save_steps 50 --save_total_limit 3 --report_to none --seed 42 --data_seed 42 \
  $([ "$TIS" = 1 ] && echo --rollout_importance_sampling_mode token_truncate --rollout_importance_sampling_threshold 2.0) \
  --output_dir "$OUT" --resume_from_checkpoint "$W/ckpt/checkpoint-$CKPT" --max_steps "$MAX_STEPS"
echo "SWIFT_EXIT=$?"
stamp "4) 결과 → ~/files (tar 1개씩 — 작은 파일 대량 금지)"
cp "$(ls -t "$OUT"/*/logging.jsonl | head -1)" "$F/logs/$TAG.logging.jsonl"
LAST=$(ls -d "$OUT"/*/checkpoint-* | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)
[ -n "$LAST" ] && tar cf "$F/$TAG.$(basename "$LAST").tar" -C "$(dirname "$LAST")" "$(basename "$LAST")"
cp "$W/run.log" "$F/logs/$TAG.log"
stamp "done"
