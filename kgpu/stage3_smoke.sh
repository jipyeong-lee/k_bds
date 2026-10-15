#!/usr/bin/env bash
# stage3_smoke.sh — kgpu pod 에서 Stage-3(의료 RL, KHDP judge) 3 step 스모크. 배선 검증용.
# 전제(~/files = /root/files, 영속):
#   kbds/code.tgz            configs/ + scripts/ (로컬에서 tar 해 PUT)
#   kbds/dr_grpo_merged/     init 모델(KISTI 반출본)
# env(필수): JUDGE_API_KEY=$KHDP_PAT — exec 의 env 로 넘긴다. 파일에 남기지 말 것.
# env(선택): NUM_GEN·MAX_COMP·PDTBS·ACCUM·MAX_STEPS — 기본은 3 step 스모크, 키우면 step 시간 측정용
# 무거운 산출물은 pod 로컬(/root/w)에 둔다 — ~/files 는 네트워크 공유라 작은 파일 대량 생성 금지.
set -uo pipefail
: "${JUDGE_API_KEY:?JUDGE_API_KEY(KHDP PAT) 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv; LOG=$F/logs/stage3_smoke_$(date +%m%d-%H%M).log
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/smoke.log") 2>&1   # 끝에 $LOG 로 1회 복사
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) code + model → pod 로컬 디스크"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
[ -d "$W/model" ] || cp -r "$F/dr_grpo_merged" "$W/model" || { echo "❌ model copy"; exit 1; }

stamp "2) venv (KISTI·B200 검증 핀 — CLAUDE.md §1.4)"
# pro6000 노드엔 기본 이미지(kgpu-pytorch)가 없어 nvidia/cuda 이미지로 빌린다 → curl·uv 부터 깐다
# 컨테이너 root 에 setgroups·DAC override 가 없다 → _apt 강등 끄고, _apt 소유 캐시 대신 $W/apt 사용
mkdir -p "$W/apt/partial"
APT="apt-get -o APT::Sandbox::User=root -o Dir::Cache::Archives=$W/apt"
command -v curl >/dev/null || { $APT update -qq && $APT install -y -qq curl ca-certificates >/dev/null; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH; }
if [ ! -x "$VENV/bin/swift" ]; then
  uv venv "$VENV" --python 3.12 2>&1 | tail -1
  uv pip install --python "$VENV/bin/python" torch==2.10.0+cu129 torchvision==0.25.0+cu129 \
    vllm==0.19.1 ms-swift==4.1.3 transformers==5.6.2 trl==0.29.1 peft==0.19.1 \
    accelerate==1.13.0 datasets==3.6.0 pyarrow==23.0.1 pillow openai huggingface_hub \
    "qwen_vl_utils==0.0.14" "decord==0.6.0" "av==17.0.1" \
    "math_verify==0.9.0" "latex2sympy2_extended==1.11.0" "word2number==1.1" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match 2>&1 | tail -4
  uv pip install --python "$VENV/bin/python" --prerelease=allow "nvidia-cutlass-dsl==4.5.0.dev0" 2>&1 | tail -2
fi
PY=$VENV/bin/python
# flashinfer 가 Blackwell 커널을 JIT 할 때 nvrtc.h 를 venv 휠에서 찾게 한다(B200 함정 ④와 같은 대응)
NVRTC=$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc
export CPATH="$NVRTC/include${CPATH:+:$CPATH}" LIBRARY_PATH="$NVRTC/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
       LD_LIBRARY_PATH="$NVRTC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
$PY -c "import torch;print('  torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"

stamp "3) medix — HF 에서 직접 받아 변환 (스모크는 ${SMOKE_ROWS:=200}행)"
if [ ! -f "$W/data/medix.jsonl" ]; then
  $PY -c "from huggingface_hub import snapshot_download as s;print(s('MBZUAI/medix-rl-data',repo_type='dataset',allow_patterns=['data/train-*']))" > "$W/mx_path"
  $PY "$W/scripts/convert_to_swift.py" MBZUAI/medix-rl-data --parquet "$(tail -1 "$W/mx_path")"/data/train-*.parquet \
    --out "$W/data/medix.jsonl" --images-dir "$W/data/images" --limit "$SMOKE_ROWS" || { echo "❌ convert"; exit 1; }
fi
wc -l "$W/data/medix.jsonl"

stamp "4) swift rlhf — ${MAX_STEPS:=3} step, judge=KHDP ${JUDGE_MODEL:=deepseek-v4-1-flash-ax}, 증거 게이트 ON"
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://api.khdp.net/v1} JUDGE_MODEL RUBRIC_EVIDENCE=1 JUDGE_DEBUG=1
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
"$VENV/bin/swift" rlhf --rlhf_type grpo \
  --model "$W/model" --model_type qwen3_5 --system "$SYS" --dataset "$W/data/medix.jsonl" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear --torch_dtype bfloat16 \
  --external_plugins "$W/configs/accuracy.py" "$W/configs/medical_reward.py" \
  --reward_funcs format_think clinical_judge --reward_weights 0.2 1.0 \
  --enable_thinking true --num_generations ${NUM_GEN:-4} --max_completion_length ${MAX_COMP:-1024} \
  --per_device_train_batch_size ${PDTBS:-1} --gradient_accumulation_steps ${ACCUM:-4} --learning_rate 1e-5 --beta 0.04 \
  --max_length 4096 --max_pixels 262144 --gradient_checkpointing true --attn_impl sdpa \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
  --vllm_gpu_memory_utilization 0.45 --vllm_max_model_len $(( 3072 + ${MAX_COMP:-1024} )) --sleep_level 1 \
  --log_completions true --logging_steps 1 --save_strategy no --report_to none \
  --output_dir "$W/runs/stage3_smoke" --max_steps $MAX_STEPS
echo "SWIFT_EXIT=$?"
stamp "done"
cp "$W/smoke.log" "$LOG"
