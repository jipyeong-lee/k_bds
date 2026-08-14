#!/usr/bin/env bash
# node_setup_and_smoke.sh — PERSISTENT setup on B200 (uses $ORCH_HOME) + pmcvqa 3-step smoke.
# Idempotent: re-runs skip extract/venv if already present. Reads uploaded data from the
# filesystem ($ORCH_HOME/uploads) — no API pull. Everything persists across sessions.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set — storage not mounted in this session}"
[ -w "$ORCH_HOME" ] || { echo "❌ ORCH_HOME not writable: $ORCH_HOME"; exit 1; }
export HOME="$ORCH_HOME"
export XDG_CACHE_HOME="$ORCH_HOME/.cache" UV_CACHE_DIR="$ORCH_HOME/.cache/uv" \
       PIP_CACHE_DIR="$ORCH_HOME/.cache/pip" HF_HOME="$ORCH_HOME/.cache/hf"
# /tmp is tmpfs+noexec here → Triton/Inductor .so can't be mapped. Redirect to exec xfs ($ORCH_HOME).
export TRITON_CACHE_DIR="$ORCH_HOME/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$ORCH_HOME/.cache/inductor" TMPDIR="$ORCH_HOME/.cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
export UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
# flashinfer JIT-compiles Blackwell kernels with system nvcc (CUDA 13) but /usr/local/cuda lacks
# nvrtc.h (and its include dir is read-only). The header+lib ARE in the venv (nvidia-cuda-nvrtc
# wheel) → expose them so the compile/link resolves. flashinfer caches the .so in $ORCH_HOME after.
NVRTC="$ORCH_HOME/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc"
export CUDA_HOME=/usr/local/cuda
export CPATH="$NVRTC/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NVRTC/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVRTC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }

SRC="$ORCH_HOME/uploads"; PROJ="$ORCH_HOME/kbds_project"; VENV="$ORCH_HOME/.venv"
echo "ORCH_HOME=$ORCH_HOME"; df -h "$ORCH_HOME" | tail -1

# There is no `set -e` here (the skip/idempotency logic relies on non-zero returns), so a failed
# extract used to sail straight through: 46G data.tar stopped partway, the script kept going and
# reported SWIFT_EXIT=0 — the smoke passed only because pmcvqa images were already on disk.
# `set -o pipefail` above makes the pipe's rc accurate; this just makes someone look at it.
untar_parts(){  # $1=parts glob (quoted at the call site, expanded here)  $2=dest dir
  cat $1 | tar xf - -C "$2" || { echo "❌ extract failed: $1 → aborting"; exit 1; }
}

stamp "1) extract data from \$ORCH_HOME/uploads (skip if present)"
if [ ! -d "$PROJ/work/checkpoints/sft_mixed_merged" ]; then
  [ -f "$SRC/code.tar.gz" ] || { echo "❌ missing $SRC/code.tar.gz — run upload first"; exit 1; }
  ls "$SRC"/parts/model.tar.*.part        >/dev/null 2>&1 || { echo "❌ missing model parts in $SRC/parts"; exit 1; }
  ls "$SRC"/parts/pmcvqa_data.tar.*.part  >/dev/null 2>&1 || { echo "❌ missing pmcvqa parts in $SRC/parts"; exit 1; }
  mkdir -p "$PROJ/work/checkpoints" "$PROJ/work"
  tar xzf "$SRC/code.tar.gz" -C "$ORCH_HOME" || { echo "❌ code.tar.gz extract failed"; exit 1; }
  untar_parts "$SRC/parts/model.tar.*.part"       "$PROJ/work/checkpoints"
  untar_parts "$SRC/parts/pmcvqa_data.tar.*.part" "$PROJ/work"
  echo "  extracted: model files=$(ls "$PROJ/work/checkpoints/sft_mixed_merged" | wc -l), pmcvqa rows=$(wc -l <"$PROJ/work/data/domains/stage2_pmcvqa.jsonl")"
else
  echo "  already extracted — skip"
fi

stamp "1b) extract full data.tar if uploaded (deepvision+mmk12, for the 3-arm run)"
# Separate guard from 1): pmcvqa-only setups already pass 1), so this must stand on its own.
# data.tar holds all of work/data — extracting it over the pmcvqa subset is a superset, not a clash.
if [ ! -f "$PROJ/work/data/domains/stage2_deepvision.jsonl" ] \
   && ls "$SRC"/parts/data.tar.*.part >/dev/null 2>&1; then
  # NOTE: this overwrites domains/*.jsonl, restoring the KISTI absolute image paths.
  # Step 2) below re-fixes them — never extract data.tar on its own without re-running that sed.
  untar_parts "$SRC/parts/data.tar.*.part" "$PROJ/work"
  for a in deepvision mmk12 pmcvqa; do
    f="$PROJ/work/data/domains/stage2_$a.jsonl"
    echo "  $a rows=$([ -f "$f" ] && wc -l <"$f" || echo MISSING)"
  done
else
  echo "  skip (already extracted, or data.tar not uploaded yet)"
fi

stamp "2) path-fix jsonl (KISTI -> node) + image sanity"
find "$PROJ/work/data" -name '*.jsonl' -print0 | xargs -0 sed -i "s#/home01/k266a01/kbds_project#$PROJ#g"
IMG=$(head -1 "$PROJ/work/data/domains/stage2_pmcvqa.jsonl" | grep -o "$PROJ[^\"]*\.png" | head -1)
[ -f "$IMG" ] && echo "  IMG_OK $IMG" || { echo "  IMG_MISSING $IMG"; exit 1; }

stamp "3) persistent venv at \$ORCH_HOME/.venv"
[ -d "$VENV/bin" ] || uv venv "$VENV" --python 3.12 2>&1 | tail -1
PYV="$VENV/bin/python"
if [ ! -x "$VENV/bin/swift" ]; then
  uv pip install --python "$PYV" torch==2.10.0+cu129 torchvision==0.25.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -2
  # --index-strategy unsafe-best-match: torch from cu129 index, everything else from pypi
  uv pip install --python "$PYV" torch==2.10.0+cu129 \
    vllm==0.19.1 ms-swift==4.1.3 transformers==5.6.2 trl==0.29.1 peft==0.19.1 \
    accelerate==1.13.0 datasets==3.6.0 pyarrow==23.0.1 pillow \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match 2>&1 | tail -4
else
  echo "  swift present — skip install"
fi
# Qwen-VL multimodal deps ms-swift does not auto-pull (always ensure; fast if present)
uv pip install --python "$PYV" "qwen_vl_utils==0.0.14" "decord==0.6.0" "av==17.0.1" 2>&1 | tail -2
# vllm 0.19.1's bundled quack (Blackwell ViT flash path) needs cutlass-dsl with ThrMma.
# pip grabbed 4.7.0 (ThrMma removed) -> pin to KISTI's 4.5.0.dev0.
uv pip install --python "$PYV" --prerelease=allow "nvidia-cutlass-dsl==4.5.0.dev0" 2>&1 | tail -3
# reward-plugin deps: configs/accuracy.py uses math_verify for answer checking
uv pip install --python "$PYV" "math_verify==0.9.0" "latex2sympy2_extended==1.11.0" "word2number==1.1" 2>&1 | tail -2
"$PYV" -c "import torch;assert torch.cuda.get_device_capability(0)[0]>=10;print('  torch',torch.__version__,torch.cuda.get_device_name(0))"

stamp "4) swift rlhf smoke — pmcvqa, 3 steps, sdpa"
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
"$VENV/bin/swift" rlhf \
  --rlhf_type grpo \
  --model "$PROJ/work/checkpoints/sft_mixed_merged" \
  --model_type qwen3_5 \
  --system "$SYS" \
  --dataset "$PROJ/work/data/domains/stage2_pmcvqa.jsonl" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear --lora_dropout 0 \
  --torch_dtype bfloat16 \
  --external_plugins "$PROJ/configs/accuracy.py" \
  --reward_funcs accuracy_mix format_think soft_overlong --reward_weights 1.0 0.2 0.2 \
  --soft_max_length 3072 --soft_cache_length 1024 \
  --dynamic_sample true --max_resample_times 3 --overlong_filter false \
  --loss_type dr_grpo --importance_sampling_level token --epsilon 0.2 --scale_rewards none \
  --enable_thinking true --num_generations 8 --temperature 0.9 --max_completion_length 1024 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 --beta 0.04 --max_length 4096 --max_pixels 262144 \
  --gradient_checkpointing true --attn_impl sdpa \
  --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  --vllm_mm_processor_cache_gb 0 --vllm_gpu_memory_utilization 0.85 --vllm_max_model_len 4096 \
  --sleep_level 1 --log_completions true --logging_steps 1 \
  --save_strategy no --output_dir "$ORCH_HOME/runs/pmcvqa_smoke" --report_to none \
  --max_steps 3 2>&1 | tail -70
echo "SWIFT_EXIT=${PIPESTATUS[0]}"
stamp "done — venv+data persist in \$ORCH_HOME for reuse"
