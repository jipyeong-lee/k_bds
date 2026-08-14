#!/usr/bin/env bash
# run_epoch.sh — arm 하나를 1 epoch 학습한다. 며칠 걸리는 실행이라 job 이 끊기는 것을 전제로,
# output_dir 의 최신 체크포인트를 찾아 자동 재개한다. 같은 명령을 반복 투입하면 이어서 간다.
#
#   ARM=deepvision PDTBS=4 ACCUM=4 bash run_epoch.sh
#
# 롤아웃 서버(GPU 7)는 매 job 마다 새로 띄운다 — job 이 killed 되면 trap 이 안 돌 수 있어
# 서버를 세션 밖으로 살려둘 수 없다. 기동 55초는 며칠 단위 실행에서 무시할 만하다.
set -uo pipefail
: "${ORCH_HOME:?ORCH_HOME not set}"
export HOME="$ORCH_HOME"
export XDG_CACHE_HOME="$ORCH_HOME/.cache" UV_CACHE_DIR="$ORCH_HOME/.cache/uv" \
       PIP_CACHE_DIR="$ORCH_HOME/.cache/pip" HF_HOME="$ORCH_HOME/.cache/hf"
export TRITON_CACHE_DIR="$ORCH_HOME/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$ORCH_HOME/.cache/inductor" TMPDIR="$ORCH_HOME/.cache/tmp"
# PDTBS 2 는 벤치 3 step 에서 117.5 GiB 였는데 실전 41 step 에서 170.6/180 GiB 까지 올랐다.
# 길이가 들쭉날쭉한 completion 이 쌓이며 생기는 단편화라 expandable_segments 가 정확히 그 대응이다
# (1차 OOM 로그가 권고한 설정이기도 하다).
export PYTORCH_ALLOC_CONF="expandable_segments:True"
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
# SCALE 을 디렉터리명에 넣어 dr_grpo+none 진행분과 섞이지 않게 한다.
# (advantage 정규화가 바뀌면 같은 output_dir 에서 이어받는 게 의미가 없다)
SCALE="${SCALE:-gdpo}"
# 비동기 롤아웃: 롤아웃이 "1 라운드 이전 가중치"를 쓰는 대신 생성과 학습이 겹친다.
# 순차 573s(롤아웃 287 + 학습 286) → max(287,286) ≈ 300s 를 노린다. server 모드 전용이라
# 오늘 colocate→server 로 바꾼 것이 전제 조건을 만들었다. 멀티턴에서는 못 쓰지만 우리는 싱글턴.
ASYNC="${ASYNC:-true}"
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)"
OUT="$ORCH_HOME/runs/$RUNTAG"
LOG="$ORCH_HOME/train_${RUNTAG}.log"
PORT=8000
NUMGEN="${NUMGEN:-16}"; WORLD="${WORLD:-7}"
# 벤치 실측: PDTBS 2 = 171 s/step @117.5 GiB (상한). PDTBS 4 는 235 GiB 가 필요해 thrashing.
PDTBS="${PDTBS:-2}"; ACCUM="${ACCUM:-8}"
# GC off 는 16K 시퀀스에서 PDTBS 1 조차 못 버틴다(GPU 178 GiB 중 357 MiB 잔여로 OOM).
GC="${GC:-true}"
# --max_steps 는 21_rlvr_grpo_adv.slurm 과 같은 의미로 쓴다: 이번 job 의 몫이 아니라 **누적 목표**.
# job 마다 다른 값을 주면 lr 스케줄러가 그 값 기준으로 다시 깔려 매번 0 으로 소멸한다.
# job 은 60분에 killed 되고 다음 job 이 체크포인트에서 이어받는다 → 손실 상한이 save_steps.
SAVE_STEPS="${SAVE_STEPS:-4}"
MAXLEN="${MAXLEN:-16384}"
# 21_rlvr_grpo_adv.slurm 의 도메인별 프리셋 — pmcvqa 만 짧다(실측 최대 1,441 토큰).
case "$ARM" in
  pmcvqa) MAXCOMP="${MAXCOMP:-4096}"; SOFTCACHE="${SOFTCACHE:-1024}" ;;
  *)      MAXCOMP="${MAXCOMP:-8192}"; SOFTCACHE="${SOFTCACHE:-2048}" ;;
esac

GEN_BATCH=$(( PDTBS * ACCUM * WORLD ))
(( GEN_BATCH % NUMGEN == 0 )) || { echo "❌ GEN_BATCH $GEN_BATCH 가 num_gen $NUMGEN 로 안 나눠짐"; exit 1; }
PPS=$(( GEN_BATCH / NUMGEN ))
ROWS=$(wc -l <"$DS")
TOTAL=$(( (ROWS + PPS - 1) / PPS ))

# 최신 체크포인트 자동 탐색 — 없으면 처음부터.
# ms-swift 는 실행마다 $OUT/v<N>-<날짜>/ 를 만들고 그 아래에 checkpoint-N 을 넣는다.
# $OUT/checkpoint-* 로 찾으면 영영 못 찾아 매 job 이 step 0 부터 다시 시작한다(실제로 겪었다).
# 재개할 때마다 새 v 디렉터리가 생기므로 전체 버전을 훑어 step 번호로 최댓값을 고른다.
# (21_rlvr_grpo_adv.slurm:224 와 같은 방식)
LATEST=$(ls -d "$OUT"/*/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)
LAST=$(printf '%s' "$LATEST" | sed 's/.*checkpoint-//')
LAST="${LAST:-0}"
RESUME=()
[ -n "$LATEST" ] && RESUME=(--resume_from_checkpoint "$LATEST") && echo "  재개 지점: $LATEST"
if [ "$LAST" -ge "$TOTAL" ]; then
  echo "✅ $ARM 1 epoch 완료 ($LAST/$TOTAL step) — 더 돌릴 것 없음"
  exit 0
fi
echo "▶ $ARM: $LAST → $TOTAL step 목표 (남은 $(( TOTAL - LAST )))"

echo "=== 설정 ==="
echo "  arm=$ARM  rows=$ROWS  1epoch=$TOTAL step"
echo "  PDTBS=$PDTBS ACCUM=$ACCUM world=$WORLD → GEN_BATCH=$GEN_BATCH, num_gen=$NUMGEN, 프롬프트/step=$PPS"
echo "  max_completion=$MAXCOMP soft_cache=$SOFTCACHE gradient_checkpointing=$GC"

stamp "1) vLLM 롤아웃 서버 기동 (GPU 7)"
CUDA_VISIBLE_DEVICES=7 nohup "$VENV/bin/swift" rollout \
  --model "$MODEL" --model_type qwen3_5 \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len "$MAXLEN" \
  --vllm_gpu_memory_utilization 0.90 \
  --vllm_enable_prefix_caching true \
  --port "$PORT" > "$ORCH_HOME/rollout_${RUNTAG}.log" 2>&1 &
ROLLOUT_PID=$!
cleanup(){ echo "[cleanup] rollout pid=$ROLLOUT_PID 종료"; kill "$ROLLOUT_PID" 2>/dev/null; wait "$ROLLOUT_PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  health OK ($((i*5))s)"; break; }
  kill -0 "$ROLLOUT_PID" 2>/dev/null || { echo "  ❌ rollout 사망"; tail -30 "$ORCH_HOME/rollout_${ARM}.log"; exit 1; }
  sleep 5
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "  ❌ 서버 미기동"; exit 1; }

stamp "2) 형식 붕괴 감시자 기동"
# 73924/73925 는 step ~900 붕괴 후 13h35m 을 더 돌았다(109 GPU-h). 이번 실행은 5,715 step /
# 11일이라 감시 없이 붕괴하면 며칠을 태운다. 표준 라이브러리만 쓰므로 venv 없이 돈다.
# --job-id 는 주지 않는다(scancel 은 Slurm 전용) → 관측·기록만 하고, 판정은 verdict 파일에 남는다.
VERDICT="$ORCH_HOME/verdict_${RUNTAG}.json"
WATCH_PID=""
if [ -f "$PROJ/scripts/watch_format_collapse.py" ]; then
  python3 "$PROJ/scripts/watch_format_collapse.py" \
    --log "$LOG" --verdict "$VERDICT" --poll 120 \
    > "$ORCH_HOME/watchdog_${ARM}.log" 2>&1 &
  WATCH_PID=$!
  echo "  watchdog pid=$WATCH_PID → $VERDICT"
else
  echo "  ⚠️ watch_format_collapse.py 없음 — 감시 없이 진행"
fi
cleanup(){
  echo "[cleanup] rollout pid=$ROLLOUT_PID 종료"; kill "$ROLLOUT_PID" 2>/dev/null; wait "$ROLLOUT_PID" 2>/dev/null
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null
}

stamp "3) 학습 1 epoch (GPU 0-6) — 로그: $LOG"
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
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
  --loss_type dr_grpo --importance_sampling_level token --epsilon 0.2 --scale_rewards "$SCALE" \
  --log_rollout_offpolicy_metrics true \
  --enable_thinking true --num_generations "$NUMGEN" --temperature 0.9 \
  --max_completion_length "$MAXCOMP" --max_length "$MAXLEN" --max_pixels 262144 \
  --per_device_train_batch_size "$PDTBS" --gradient_accumulation_steps "$ACCUM" \
  --learning_rate 1e-5 --beta 0.04 \
  --gradient_checkpointing "$GC" --attn_impl sdpa \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "$PORT" \
  --async_generate "$ASYNC" \
  --log_completions false --logging_steps 1 \
  --max_steps "$TOTAL" \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 3 \
  --output_dir "$OUT" --report_to none \
  "${RESUME[@]}" >> "$LOG" 2>&1
RC=$?

# job 이 timeout 으로 killed 되면 이 아래는 안 돈다 — 그래서 save_steps 가 손실 상한이다.
stamp "4) 결과  RC=$RC"
# 여기도 v<N>-<날짜> 를 거쳐야 한다 — 재개 로직과 같은 경로 규칙.
DONE=$(ls -d "$OUT"/*/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
echo "  진행: ${DONE:-0} / $TOTAL step"
[ -f "$VERDICT" ] && { echo "--- 붕괴 감시 판정 ---"; cat "$VERDICT"; }
echo "--- 최근 step ---"
grep -oE "\{'loss'[^}]*\}" "$LOG" | tail -2
echo "--- 에러 ---"
grep -iE "out of memory|traceback|❌" "$LOG" | tail -3 || echo "  (없음)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
