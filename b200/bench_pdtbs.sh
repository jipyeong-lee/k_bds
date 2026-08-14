#!/usr/bin/env bash
# bench_pdtbs.sh — gradient_checkpointing off 상태에서 PDTBS 상한과 처리량을 실측한다.
#
# GEN_BATCH = PDTBS × ACCUM × 7 을 112 로 고정(곱이 16)하므로 롤아웃 부하는 모든 조합에서
# 동일하다. 따라서 조합 간 s/step 차이는 전부 학습 쪽 이득이다.
# GC off 는 activation 을 몇 배로 키우므로 PDTBS 와 같은 메모리를 두고 경쟁한다 → OOM 나는
# 지점이 곧 상한. OOM 은 잡아서 기록하고 다음 조합으로 넘어간다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
export HOME="$ORCH_HOME"
export XDG_CACHE_HOME="$ORCH_HOME/.cache" UV_CACHE_DIR="$ORCH_HOME/.cache/uv" \
       PIP_CACHE_DIR="$ORCH_HOME/.cache/pip" HF_HOME="$ORCH_HOME/.cache/hf"
export TRITON_CACHE_DIR="$ORCH_HOME/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$ORCH_HOME/.cache/inductor" TMPDIR="$ORCH_HOME/.cache/tmp"
NVRTC="$ORCH_HOME/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc"
export CUDA_HOME=/usr/local/cuda
export CPATH="$NVRTC/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NVRTC/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVRTC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }

PROJ="$ORCH_HOME/kbds_project"; VENV="$ORCH_HOME/.venv"
ARM="${ARM:-deepvision}"
DS="$PROJ/work/data/domains/stage2_${ARM}.jsonl"
MODEL="$PROJ/work/checkpoints/sft_mixed_merged"
PORT=8000
MAXCOMP=8192; SOFTCACHE=2048; MAXLEN=16384; NUMGEN=16; WORLD=7
# GC off 는 1차 벤치에서 PDTBS 2 조차 OOM 이었다(16K 시퀀스 × 전 레이어 activation).
# drive_node.sh 는 로컬 env 를 노드로 넘기지 않으므로 기본값 자체를 바꾼다.
GC="${GC:-true}"
# "PDTBS:ACCUM" — 곱은 항상 16. 메모리 낮은 순서라 뒤에서 터져도 앞 결과는 남는다.
COMBOS="${COMBOS:-2:8 4:4 8:2}"
STEPS="${STEPS:-3}"

# 직전 벤치가 남긴 OOM 이 정확히 어디서 났는지 — 세션이 반납돼도 $ORCH_HOME 로그는 남는다.
echo "=== 직전 벤치 OOM 지점 ==="
for f in "$ORCH_HOME"/bench_p*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f")"
  grep -oE "CUDA out of memory[^.]*\. [^.]*\. [^.]*\." "$f" | head -1 || echo "  (OOM 라인 없음)"
done

echo "=== 벤치 설정 ==="
echo "  arm=$ARM  rows=$(wc -l <"$DS")  gradient_checkpointing=$GC"
echo "  GEN_BATCH 고정=112 (num_gen=$NUMGEN → 프롬프트/step=7), max_completion=$MAXCOMP"
echo "  조합: $COMBOS  ($STEPS step 씩)"

stamp "1) vLLM 롤아웃 서버 기동 (GPU 7) — 조합 전체에서 재사용"
CUDA_VISIBLE_DEVICES=7 nohup "$VENV/bin/swift" rollout \
  --model "$MODEL" --model_type qwen3_5 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len "$MAXLEN" \
  --vllm_gpu_memory_utilization 0.90 \
  --port "$PORT" > "$ORCH_HOME/rollout.log" 2>&1 &
ROLLOUT_PID=$!
cleanup(){ echo "[cleanup] rollout pid=$ROLLOUT_PID 종료"; kill "$ROLLOUT_PID" 2>/dev/null; wait "$ROLLOUT_PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  health OK ($((i*5))s)"; break; }
  kill -0 "$ROLLOUT_PID" 2>/dev/null || { echo "  ❌ rollout 사망"; tail -30 "$ORCH_HOME/rollout.log"; exit 1; }
  sleep 5
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "  ❌ 서버 미기동"; exit 1; }

SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."

for combo in $COMBOS; do
  PDTBS="${combo%%:*}"; ACCUM="${combo##*:}"
  GEN_BATCH=$(( PDTBS * ACCUM * WORLD ))
  LOG="$ORCH_HOME/bench_p${PDTBS}a${ACCUM}.log"
  stamp "2) PDTBS=$PDTBS ACCUM=$ACCUM (GEN_BATCH=$GEN_BATCH) 시작"
  S=$(date +%s)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC_PER_NODE="$WORLD" \
  "$VENV/bin/swift" rlhf \
    --rlhf_type grpo \
    --model "$MODEL" --model_type qwen3_5 --system "$SYS" \
    --dataset "$DS" \
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear --lora_dropout 0 \
    --torch_dtype bfloat16 \
    --external_plugins "$PROJ/configs/accuracy.py" \
    --reward_funcs accuracy_mix format_think soft_overlong --reward_weights 1.0 0.2 0.2 \
    --soft_max_length "$MAXCOMP" --soft_cache_length "$SOFTCACHE" \
    --dynamic_sample true --max_resample_times 3 --overlong_filter false \
    --loss_type dr_grpo --importance_sampling_level token --epsilon 0.2 --scale_rewards none \
    --enable_thinking true --num_generations "$NUMGEN" --temperature 0.9 \
    --max_completion_length "$MAXCOMP" --max_length "$MAXLEN" --max_pixels 262144 \
    --per_device_train_batch_size "$PDTBS" --gradient_accumulation_steps "$ACCUM" \
    --learning_rate 1e-5 --beta 0.04 \
    --gradient_checkpointing "$GC" --attn_impl sdpa \
    --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "$PORT" \
    --log_completions false --logging_steps 1 \
    --save_strategy no --output_dir "$ORCH_HOME/runs/bench_p${PDTBS}a${ACCUM}" --report_to none \
    --max_steps "$STEPS" > "$LOG" 2>&1
  RC=$?
  WALL=$(( $(date +%s) - S ))

  # stdout 은 서버가 tail 로 자른다 → 파일에 전문, 여기엔 한 줄 요약만.
  if grep -qi "out of memory" "$LOG"; then
    echo "  RESULT PDTBS=$PDTBS ACCUM=$ACCUM → ❌ OOM (wall=${WALL}s) — 여기가 상한"
  elif [ "$RC" -ne 0 ]; then
    echo "  RESULT PDTBS=$PDTBS ACCUM=$ACCUM → ❌ rc=$RC (wall=${WALL}s)"
    grep -iE "error|traceback|assert" "$LOG" | grep -viE "it/s|MiB|ignore_args_error" | head -3
  else
    MEM=$(grep -oE "'memory\(GiB\)': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    EL=$(grep -oE "'elapsed_time': '[^']*'" "$LOG" | tail -1)
    LEN=$(grep -oE "'completions/mean_length': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    CLIP=$(grep -oE "'completions/clipped_ratio': '[0-9.]+'" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    echo "  RESULT PDTBS=$PDTBS ACCUM=$ACCUM → ✅ wall=${WALL}s  s/step=$(( WALL / STEPS ))  mem=${MEM}GiB  meanlen=${LEN}  clip=${CLIP}  $EL"
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
  sleep 5   # 다음 조합 전 메모리 반납 대기
done

stamp "3) 끝 — 로그: \$ORCH_HOME/bench_p*.log"
