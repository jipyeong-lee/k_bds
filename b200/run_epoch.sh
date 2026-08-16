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
# 학습·추론 확률 불일치 보정(TIS). async 롤아웃은 "1 라운드 이전 가중치"로 생성하므로 정책 지연이
# 그대로 off-policy 편향으로 쌓인다 — 실측으로 log_ppl_abs_diff 가 78 step 4.4% → 102 step 8.1% 로
# 올라 IcePop 기준(5%)을 넘겼다. loss 에 w=min(π_θ/π_rollout, C) 를 곱해 그 편향을 되돌린다.
# ppl_ratio 실측이 1.086 이라 C=2 는 거의 안 걸린다 → 잘라내기보다 순수 IS 보정으로 동작한다.
# rollout logprob 은 이미 수집되고 있다(off-policy 지표가 찍히는 게 그 증거) → 추가 비용 없음.
# 4.1.3 과 최신 4.5.0 의 mismatch 관련 인자는 동일하다(인자 diff 로 확인) → 업그레이드 이득 없음.
ISMODE="${ISMODE:-token_truncate}"   # '' 면 보정 끔. token_mask/sequence_truncate/sequence_mask 도 가능
ISTHRESH="${ISTHRESH:-2.0}"
# 참조 모델 KL 페널티. 0.04 로 돌다가 step ~280 에서 0 으로 내렸다 — Dr. GRPO·DAPO·GSPO·GLM-5 가
# 모두 beta 0 을 쓴다. 검증 가능한 보상이 있는 RLVR 에서는 참조 모델 KL 이 탐색을 묶는 쪽으로 작용한다.
# loss_type=dr_grpo 를 고르면서 KL 을 남겨둔 건 앞뒤가 안 맞았다. 부수 효과로 ref 모델 forward 가
# 사라져 step 이 빨라지고 메모리도 준다.
BETA="${BETA:-0}"
# GRPO 는 그룹 내 다양성이 advantage 의 분산을 만든다 → 관례는 1.0 이고 ms-swift 예제도 1.0 이다.
# 0.9 로 돌다가 같은 시점에 1.0 으로 올렸다.
TEMP="${TEMP:-1.0}"
# 엔트로피 마스크. 첫 실행(RUNTAG …_tis)이 step ~700 부터 엔트로피 붕괴로 무너졌다 —
# 롤아웃 ppl 이 1.70→1.14 로 단조 하락(생성이 뾰족해짐), 길이 1344→498, FormatThink 0.995→0.936,
# log_ppl_abs_diff 0.067→0.380(IcePop 임계의 7.6배). 그런데 그 실행에는 **엔트로피를 붙드는 장치가
# 하나도 없었다**: beta=0(참조 KL 없음) · entropy bonus 는 ms-swift 에 인자 자체가 없음 ·
# top_entropy_quantile=1.0(마스크 없음) · 그리고 clip 은 1,280 step 전부 미발동이었다
# (num_iterations=1 → π_θ==π_old → ratio≡1. grpo_trainer.py:1150 이 그 최적화를 명시한다).
#   ⚠️ 그래서 DAPO Clip-Higher(epsilon_high)는 이 설정에서 **죽은 인자**다 — 건드리지 않는다.
# 남은 레버는 이것뿐이다: 상위 q 비율의 고엔트로피 토큰에만 손실을 남긴다
# (grpo_trainer.py:1126 threshold → :1223 per_token_loss *= entropy_mask).
# 저엔트로피 토큰은 모델이 이미 확신하는 자리라, 거기 계속 그래디언트를 부으면 분포가 더 뾰족해진다.
# 다만 이건 엔트로피를 **더하는** 장치가 아니라 **빼는 힘을 약화시키는** 장치다(arXiv:2509.26114 의 구분).
ENTQ="${ENTQ:-0.2}"                  # 1.0 이면 마스크 끔
#   ⚠️ 부작용: dr_grpo 의 분모는 batch_size*max_completion_length 라 **상수**다(grpo_trainer.py:1252).
#   마스크는 분자만 줄이므로(:1223) 실효 그래디언트가 그만큼 작아진다. 배수는 토큰별 손실 기여가
#   균등하지 않아 미지수 → 추측으로 lr 을 올리지 말고 grad_norm 을 보고 정한다.
#   기준선: 마스크 없던 첫 실행의 초반 grad_norm = 0.0062 (step 1~199 평균).
LR="${LR:-1e-5}"
IS_ARGS=()
[ -n "$ISMODE" ] && IS_ARGS=(--rollout_importance_sampling_mode "$ISMODE" \
                             --rollout_importance_sampling_threshold "$ISTHRESH")
# 보정 유무로 output_dir 을 가른다 — loss 가 바뀌면 같은 디렉터리에서 이어받는 게 의미가 없다.
RUNTAG="${ARM}_ep1_${SCALE}$([ "$ASYNC" = true ] && echo _async)$([ -n "$ISMODE" ] && echo _tis)$([ "$ENTQ" != "1.0" ] && echo _entmask)"
OUT="$ORCH_HOME/runs/$RUNTAG"
LOG="$ORCH_HOME/train_${RUNTAG}.log"
PORT=8000
NUMGEN="${NUMGEN:-16}"; WORLD="${WORLD:-7}"
# PDTBS 4 를 실제로 돌려봤고 되돌렸다. expandable_segments 덕에 OOM 은 안 났지만(143.7→150 GiB)
# **wall clock 이 전혀 줄지 않았다** — job 하나에서 PDTBS 2 는 119 s/step, PDTBS 4 는 122 s/step 이었고
# 심지어 PDTBS 4 쪽 completion 이 더 짧았다(679 vs 889). step_time 은 48→38 s 로 줄었는데 벽시계가
# 그대로라는 건 병목이 step 밖(롤아웃 대기·통신)에 있다는 뜻이다. 메모리 65 GiB 를 더 쓰고 얻은 게 없다.
# GEN_BATCH=PDTBS×ACCUM×world 는 112 로 고정이어야 한다(num_gen 16) → 바꾸려면 곱이 16 을 유지할 것.
PDTBS="${PDTBS:-2}"; ACCUM="${ACCUM:-8}"
# GC off 는 16K 시퀀스에서 PDTBS 1 조차 못 버틴다(GPU 178 GiB 중 357 MiB 잔여로 OOM).
GC="${GC:-true}"
# --max_steps 는 21_rlvr_grpo_adv.slurm 과 같은 의미로 쓴다: 이번 job 의 몫이 아니라 **누적 목표**.
# job 마다 다른 값을 주면 lr 스케줄러가 그 값 기준으로 다시 깔려 매번 0 으로 소멸한다.
# log_completions=true → $OUT/v<N>-*/completions.jsonl 에 prompt·completion·보상·advantage·entropy.
# 롤아웃 실물을 보는 유일한 경로다(1차 실행은 이게 꺼져 있어 붕괴한 텍스트를 끝내 못 봤다).
# 크기는 step 당 generation_batch_size(=112) 건 → job 당 ~50MB, v 디렉터리가 job 마다 새로 생겨 누적되지 않는다.
# job 은 60분에 killed 되고 다음 job 이 체크포인트에서 이어받는다 → 손실 상한이 save_steps.
# 4 는 너무 비쌌다(2026-08-16 실측): 저장 1회 236s = step 4.4개 분량이라 실효 113 s/step 중 절반이 저장이다.
#   save_steps  4 → 실효 113.1 s/step · 1 epoch 179.5h · kill 시 평균 손실 1.8분
#   save_steps 16 → 실효  68.8 s/step · 1 epoch 109.3h · kill 시 평균 손실 7.2분
# 70시간을 아끼려고 job 당 5분을 더 거는 거래 — job 50회면 보험료 4.5h, 명백히 남는다.
SAVE_STEPS="${SAVE_STEPS:-16}"
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
echo "  off-policy 보정: ${ISMODE:-없음} (threshold=$ISTHRESH) · async_generate=$ASYNC"
echo "  beta(KL)=$BETA · temperature=$TEMP · top_entropy_quantile=$ENTQ (1.0=마스크 끔) · lr=$LR"

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

# 요청한 포트가 아니라 **실제로 바인딩된 포트**를 로그에서 읽는다.
#
# 앞 job 이 timeout 으로 killed 되면 그 rollout 의 소켓이 TIME_WAIT 로 남는다. 이때 vLLM 은 실패하지
# 않고 **조용히 다음 포트(8001)로 올라간다.** 우리가 준 $PORT 만 폴링하면 서버가 53 초 만에 멀쩡히
# 떠 있는데도 타임아웃까지 기다리다 죽는다 — job #4·#6·#8·#10 이 전부 이것이었고 매번 10~30 분을
# 8000 에서 헛기다렸다. 실측: 실패 job #10 은 8001, 성공 job #11 은 8000 이었다.
#
# TIME_WAIT 는 "연결이 되는가"로는 감지할 수 없다(연결은 실패하고 bind 만 실패한다) — 그래서
# 포트를 미리 비우려는 시도는 통하지 않는다. 뜬 포트를 따라가는 쪽이 맞다.
RLOG="$ORCH_HOME/rollout_${RUNTAG}.log"
for i in $(seq 1 240); do   # 20분. 정상이면 53 초에 뜬다.
  ACTUAL=$(grep -oE "Uvicorn running on http://[0-9.]+:[0-9]+" "$RLOG" 2>/dev/null | tail -1 | sed 's/.*://')
  if [ -n "$ACTUAL" ] && curl -s "http://127.0.0.1:$ACTUAL/health" >/dev/null 2>&1; then
    [ "$ACTUAL" != "$PORT" ] && echo "  ⚠️ 포트가 밀렸다: $PORT → $ACTUAL (앞 job 소켓이 TIME_WAIT)"
    PORT="$ACTUAL"
    echo "  health OK ($((i*5))s, port=$PORT)"
    break
  fi
  kill -0 "$ROLLOUT_PID" 2>/dev/null || { echo "  ❌ rollout 사망"; tail -30 "$RLOG"; exit 1; }
  sleep 5
done
curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "  ❌ 서버 미기동 — rollout 로그 꼬리:"; tail -20 "$RLOG"; exit 1; }

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
  --top_entropy_quantile "$ENTQ" --log_entropy true \
  --log_rollout_offpolicy_metrics true \
  "${IS_ARGS[@]}" \
  --enable_thinking true --num_generations "$NUMGEN" --temperature "$TEMP" \
  --max_completion_length "$MAXCOMP" --max_length "$MAXLEN" --max_pixels 262144 \
  --per_device_train_batch_size "$PDTBS" --gradient_accumulation_steps "$ACCUM" \
  --learning_rate "$LR" --beta "$BETA" \
  --gradient_checkpointing "$GC" --attn_impl sdpa \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "$PORT" \
  --async_generate "$ASYNC" \
  --log_completions true --logging_steps 1 \
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
