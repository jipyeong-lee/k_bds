#!/usr/bin/env bash
# dv_loco_smoke.sh — kgpu pod 1장에서 풀이 줄이기 6차 75 step 모델을 init 으로 deepvision GRPO 를 짧게 돌린다.
# 왜: SFT 대신 이 모델로 Stage-2 를 하려는데, 홀드아웃 deepvision 답의 33% 가 6144 토큰을 넘는다(잘리면 보상 0).
#     형식 점수 시작값, 잘림 비율, 길이가 스스로 줄어드는지, step 시간을 본다(2026-11-11).
# 레시피는 KISTI 전문가 학습(launch_domain_experts.sh → 21_rlvr_grpo_adv.slurm RECIPE=stable)과 같다:
#   dr_grpo · gdpo · beta 0 · ENTQ 0.2 · temp 1.0 · num_gen 8 · 6144/soft 2048 · 32 completion/step.
#   바뀐 것: 1 GPU 라 PDTBS×ACCUM=2×16 · attn sdpa · lr constant(mmk12_probe.sh 와 같은 방식).
# 전제(~/files = /root/files): kbds/{code.tgz, setup_venv.sh, merge_experts.py, opd_dv/{stage2_deepvision.jsonl,dv_images.tar}, loco/adapter_v6_75}
# env(선택): MAX_STEPS(기본 40) · SAVE_STEPS(기본 20) · ADAPTER(기본 adapter_v6_75)
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv; TAG=dv_loco_smoke_$(date +%m%d-%H%M)
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) code·data → pod 로컬"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
mkdir -p "$W/data/images"
tar xf "$F/opd_dv/dv_images.tar" -C "$W/data/images" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }
cp "$F/opd_dv/stage2_deepvision.jsonl" "$W/data/dv.jsonl" || { echo "❌ data"; exit 1; }
sed -i "s#${SRC_HOME:?SRC_HOME 필요}/kbds_project/work/data/images/deepvision#$W/data/images#g" "$W/data/dv.jsonl"

stamp "2) venv"
. "$F/setup_venv.sh"
export PYTHONPATH="$W/scripts/pyfix${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_ALLOC_CONF=expandable_segments:True
$PY -c "import json,os;p=json.loads(open('$W/data/dv.jsonl').readline())['images'][0];assert os.path.exists(p),p;print('이미지 OK',p)" || { echo "❌ 이미지 경로"; exit 1; }

stamp "3) init = 공개 Qwen3.5-9B(c202236) + loco/${ADAPTER:=adapter_v6_75} 병합"
$PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a', local_dir='$W/qwen')" || { echo "❌ download base"; exit 1; }
rm -rf "$W/qwen/.cache"   # merge_experts.py 가 base 파일을 복사하다 IsADirectoryError 로 멈춘다(2026-11-10)
cp "$F/merge_experts.py" "$W/merge_experts.py" && cp -r "$F/loco/$ADAPTER" "$W/ad" || { echo "❌ adapter"; exit 1; }   # ~/files 에서 바로 읽으면 Errno 121
"$PY" "$W/merge_experts.py" --base "$W/qwen" --adapters "$W/ad" --weights 1.0 --out "$W/base" || { echo "❌ merge"; exit 1; }
rm -rf "$W/qwen"
# 본학습(KISTI 등)에서 다시 쓰도록 병합본을 남긴다. 이미 있으면 건너뛴다
[ -n "${NO_KEEP_MERGED:-}" ] || [ -f "$F/loco6_75_merged/config.json" ] || { cp -r "$W/base" "$F/loco6_75_merged.tmp" && mv "$F/loco6_75_merged.tmp" "$F/loco6_75_merged"; } &
CP_PID=$!

stamp "4) swift rlhf — deepvision ${MAX_STEPS:=40} step (저장 ${SAVE_STEPS:=20})"
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
SYS="$SYS${SYS_EXTRA:+ $SYS_EXTRA}"   # 형식 점검용: 예) "After </think>, write <answer> immediately; no explanation outside the tags."
OUT=$W/runs/$TAG
# 학습 중 지표를 5분마다 ~/files 로 내보낸다(파일 하나 덮어쓰기) — 끝나기 전에 형식·길이 추세를 본다
( while sleep 300; do L=$(ls -t "$OUT"/*/logging.jsonl 2>/dev/null | head -1); [ -n "$L" ] && cp "$L" "$F/logs/$TAG.logging.jsonl"; done ) & BK=$!
"$VENV/bin/swift" rlhf --rlhf_type grpo \
  --model "$W/base" --model_type qwen3_5 --system "$SYS" --dataset "$W/data/dv.jsonl" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
  --external_plugins "$W/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong --reward_weights 1.0 0.2 0.2 \
  --soft_max_length 6144 --soft_cache_length 2048 \
  --loss_type dr_grpo --epsilon 0.2 --importance_sampling_level token --scale_rewards gdpo \
  --dynamic_sample true --max_resample_times 3 --overlong_filter false \
  --beta 0 --top_entropy_quantile 0.2 --temperature 1.0 \
  --enable_thinking true --num_generations 8 --max_completion_length 6144 \
  --per_device_train_batch_size ${PDTBS:-2} --gradient_accumulation_steps ${ACCUM:-16} \
  --learning_rate 1e-5 --lr_scheduler_type constant \
  --max_length 10240 --max_pixels 1003520 --gradient_checkpointing true --attn_impl sdpa \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization 0.4 --vllm_max_model_len 10240 --sleep_level 1 \
  --log_completions true --log_entropy true --log_rollout_offpolicy_metrics true --logging_steps 1 \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 2 --report_to none --seed 42 --data_seed 42 \
  --output_dir "$OUT" --max_steps "$MAX_STEPS"
echo "SWIFT_EXIT=$?"
kill $BK 2>/dev/null

stamp "5) 결과 → ~/files (tar 1개씩 — 작은 파일 대량 금지)"
L=$(ls -t "$OUT"/*/logging.jsonl 2>/dev/null | head -1); [ -n "$L" ] && cp "$L" "$F/logs/$TAG.logging.jsonl"
C=$(ls -t "$OUT"/*/completions.jsonl 2>/dev/null | head -1); [ -n "$C" ] && cp "$C" "$F/logs/$TAG.completions.jsonl"
LAST=$(ls -d "$OUT"/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)
[ -n "$LAST" ] && tar cf "$F/$TAG.$(basename "$LAST").tar" -C "$(dirname "$LAST")" "$(basename "$LAST")"
wait $CP_PID; ls "$F/loco6_75_merged" | head -3
cp "$W/run.log" "$F/logs/$TAG.log"
stamp "done"
