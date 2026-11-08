#!/usr/bin/env bash
# loco_dpo.sh — 풀이 줄이기 B안 4단계: 원본 Qwen3.5-9B 에 이미지 포함 DPO 본학습(좋은 답 = 큰 Qwen 의 짧은 정답, 나쁜 답 = 원본의 긴 답).
# 전제(~/files/kbds): setup_venv.sh · loco/pairs_inputs.tar(pairs.jsonl 의 이미지 경로는 IMG/<파일>, images/ 포함 — scripts/loco_build_pairs.py 산출)
# env(선택): INPUT_TAR(pairs_inputs.tar) OUT(adapter_new) EPOCHS(2) ACCUM(8) LR(1e-5) MAXLEN(10240) MODEL_HF(Qwen/Qwen3.5-9B)
# 실행: MAX_SEC=7200 setsid nohup bash kgpu/run_pod.sh loco_dpo.sh > logs/loco_dpo.out 2>&1 < /dev/null & disown
# 출력: ~/files/kbds/loco/adapter(학습한 LoRA) · 로그 ~/files/kbds/logs/loco_dpo_*.log. 판정: DPO_EXIT=0 · loss·rewards/accuracies 추이
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs" "$F/loco"; exec > >(tee "$W/dpo.log") 2>&1
trap 'cp "$W/dpo.log" "$F/logs/loco_dpo_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
stamp "1) 입력"
mkdir -p "$W/in" && tar xf "$F/loco/${INPUT_TAR:-pairs_inputs.tar}" -C "$W/in" --no-same-owner || { echo "❌ pairs_inputs.tar"; exit 1; }
sed "s#IMG/#$W/in/images/#g" "$W/in/pairs.jsonl" > "$W/pairs.jsonl"; wc -l "$W/pairs.jsonl"
stamp "2) venv"; . "$F/setup_venv.sh"
stamp "3) 모델 ← HF"
for TRY in 1 2 3; do
  HF_HUB_DISABLE_XET=$([ $TRY -gt 1 ] && echo 1 || echo 0) timeout 480 $PY -c "from huggingface_hub import snapshot_download as s;s('${MODEL_HF:-Qwen/Qwen3.5-9B}',local_dir='$W/model')" && break
  echo "내려받기 ${TRY}번째 실패 — 다시 시도"; [ $TRY = 3 ] && { echo "❌ model"; exit 1; }
done
stamp "4) swift rlhf dpo ${EPOCHS:-2} epoch (LoRA r256, q/k/v/o, beta 0.1, 유효 배치 ${ACCUM:-8})"
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 NPROC_PER_NODE=1 "$VENV/bin/swift" rlhf --rlhf_type dpo \
  --model "$W/model" --model_type qwen3_5 --dataset "$W/pairs.jsonl" --enable_thinking true \
  --tuner_type lora --lora_rank 256 --lora_alpha 512 --lora_dropout 0.05 --target_modules q_proj k_proj v_proj o_proj --torch_dtype bfloat16 \
  --beta 0.1 --learning_rate "${LR:-1e-5}" --lr_scheduler_type cosine --warmup_ratio 0.1 --num_train_epochs "${EPOCHS:-2}" \
  --max_length "${MAXLEN:-10240}" --max_pixels 262144 --gradient_checkpointing true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "${ACCUM:-8}" --attn_impl sdpa \
  --logging_steps 1 --save_strategy epoch --save_total_limit 1 --report_to none --output_dir "$W/run"
echo "DPO_EXIT=$?"
CK=$(ls -d "$W"/run/*/checkpoint-* 2>/dev/null | sort -V | tail -1); echo "체크포인트: $CK"
[ -n "$CK" ] && cp -r "$CK" "$F/loco/${OUT:-adapter_new}" && ls -la "$F/loco/${OUT:-adapter_new}"
stamp "done"
