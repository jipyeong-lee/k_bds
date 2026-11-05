#!/usr/bin/env bash
# opd_server_probe.sh — kgpu pod(rtxpro6000 1장)에서 27B 교사 서버 모드 OPD 의 배선을 판정한다. 학습 목적이 아니다(2 step).
#   같은 GPU 에 Qwen3.8-27B-FP8 교사 vLLM 서버(0.42)와 학생 ta_w07 GKD(--teacher_model_server, vLLM colocate 롤아웃)
#   조건 A = 교사 기본 템플릿(지금 설계 그대로) · 조건 B = 교사 서버를 학생 템플릿(--chat-template student.jinja)으로
#   판정(opd_server_probe.py check): C1 교사가 학생 답을 받는가 · C2 꼬리 정렬 일치 · C3 top-20 포함률
# 전제(~/files/kbds): code.tgz setup_venv.sh opd_server_probe.py student.jinja(sft_mixed_merged 의 chat_template.jinja) opd_dv/{stage2_deepvision.jsonl,dv_images.tar}
# env(필수): HF_TOKEN(학생 저장소 비공개) — run_pod.sh 에 이름만 넘긴다. env(선택): MAX_COMP(1024) TEACHER_GPU(0) TEACHER_UTIL(0.42) STUDENT_VLLM_UTIL(0.22) MAX_STEPS(2) SMOKE_ROWS(16)
# 실행: MAX_SEC=7200 setsid nohup bash kgpu/run_pod.sh opd_server_probe.sh HF_TOKEN > logs/opd_server_probe.out 2>&1 < /dev/null & disown
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv; LOG=$F/logs/opd_server_probe_$(date +%m%d-%H%M).log
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/probe.log") 2>&1
trap 'cp "$W/probe.log" "$LOG"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True
MAX_COMP=${MAX_COMP:-1024}; MAX_STEPS=${MAX_STEPS:-2}; SMOKE_ROWS=${SMOKE_ROWS:-16}; PORT=8001
# kgpu 의 rtxpro6000 은 전체 1장(2026-11-05 /v1/gpu-models) → 교사 서버와 학생을 GPU0 하나에 나눠 싣는다.
# 교사 0.42(≈40GB: FP8 가중치 ≈30GB + KV) · 학생 vLLM colocate 0.22 · 학생 학습(가중치 18GB + 활성값). 2장이면 TEACHER_GPU=1 로 분리
TEACHER_GPU=${TEACHER_GPU:-0}; TEACHER_UTIL=${TEACHER_UTIL:-0.42}; STUDENT_VLLM_UTIL=${STUDENT_VLLM_UTIL:-0.22}
nvidia-smi -L

stamp "1) code·입력"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
cp "$F/opd_server_probe.py" "$F/student.jinja" "$W/" || { echo "❌ probe 입력"; exit 1; }

stamp "2) venv"
. "$F/setup_venv.sh"
$PY "$W/opd_server_probe.py" selftest || exit 1

stamp "3) 모델 — 학생 ta_w07(비공개) · 교사 Qwen3.8-27B-FP8(공개) 병렬"
$PY -c "from huggingface_hub import snapshot_download as s;s('easypyeong/Qwen3.5-9B-MedCrossReason',local_dir='$W/student')" & P1=$!
$PY -c "from huggingface_hub import snapshot_download as s;s('Qwen/Qwen3.8-27B-FP8',local_dir='$W/teacher')" & P2=$!
wait $P1 || { echo "❌ student download"; exit 1; }; wait $P2 || { echo "❌ teacher download"; exit 1; }

stamp "4) deepvision 프롬프트 ${SMOKE_ROWS}행"
DATASET=$W/data/dv.jsonl; IMG=$W/data/images/deepvision; mkdir -p "$IMG"
head -n "$SMOKE_ROWS" "$F/opd_dv/stage2_deepvision.jsonl" | sed "s#/home01/k266a01/kbds_project/work/data/images/deepvision/#$IMG/#g" > "$DATASET"
grep -oE "$IMG/[^\"]+" "$DATASET" | sed "s#^$IMG/##" > "$W/dv_need.txt"
[ -s "$W/dv_need.txt" ] || { echo "❌ 이미지 목록이 비었다"; exit 1; }
tar xf "$F/opd_dv/dv_images.tar" -C "$IMG" -T "$W/dv_need.txt" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }
[ "$(grep -c /home01/ "$DATASET")" = 0 ] || { echo "❌ KISTI 경로가 남았다"; exit 1; }

SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
SRV=
serve(){   # $@ = 추가 vllm 인자. GPU1 에 교사 서버를 띄우고 READY 까지 기다린다
  CUDA_VISIBLE_DEVICES=$TEACHER_GPU "$VENV/bin/vllm" serve "$W/teacher" --served-model-name teacher --max-model-len $(( 4096 + MAX_COMP )) \
    --gpu-memory-utilization $TEACHER_UTIL --max-num-batched-tokens 2048 --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
    --max-logprobs 20 --host 127.0.0.1 --port $PORT "$@" > "$W/serve_$COND.log" 2>&1 & SRV=$!
  for i in $(seq 1 90); do
    curl -s http://127.0.0.1:$PORT/v1/models | grep -q teacher && { stamp "교사 READY ($COND)"; return 0; }
    kill -0 $SRV 2>/dev/null || break; sleep 10
  done
  echo "❌ 교사 서버 기동 실패 ($COND)"; tail -30 "$W/serve_$COND.log"; return 1
}
probe(){   # GPU0 에서 GKD 2 step 을 기록 모드로 돌리고 판정한다
  # torchrun 없이 같은 프로세스에서 돌리므로 단일 rank 분산 env 를 직접 준다(없으면 vLLM colocate 가 KeyError: 'RANK')
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 "$PY" "$W/opd_server_probe.py" run "$W/probe_$COND" -- --rlhf_type gkd \
    --model "$W/student" --model_type qwen3_5 --system "$SYS" --dataset "$DATASET" \
    --teacher_model_server http://127.0.0.1:$PORT --gkd_logits_topk 20 \
    --lmbda 1.0 --beta 1.0 --seq_kd false --sft_alpha 0 --temperature 1.0 \
    --tuner_type lora --lora_rank 32 --lora_alpha 32 --use_rslora true --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
    --enable_thinking true --max_completion_length $MAX_COMP \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --learning_rate 1e-5 \
    --max_length $(( 2048 + MAX_COMP )) --max_pixels 262144 --gradient_checkpointing true --attn_impl sdpa \
    --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 --vllm_mm_processor_cache_gb 0 \
    --vllm_gpu_memory_utilization $STUDENT_VLLM_UTIL --vllm_max_model_len $(( 3072 + MAX_COMP )) --sleep_level 1 \
    --logging_steps 1 --save_strategy no --report_to none --output_dir "$W/runs/$COND" --max_steps $MAX_STEPS
  echo "SWIFT_EXIT_$COND=$?"
  "$PY" "$W/opd_server_probe.py" check "$W/probe_$COND" "$W/student" | tee "$W/check_$COND.txt"
  mkdir -p "$F/logs/opd_probe_$COND"; cp "$W/check_$COND.txt" "$W"/probe_$COND/*.json "$F/logs/opd_probe_$COND/" 2>/dev/null
  grep -a -E "Error|Traceback" "$W/serve_$COND.log" | tail -n 20 > "$F/logs/opd_probe_${COND}_serve_err.log"
}

COND=A_default; stamp "5) 조건 A — 교사 기본 템플릿"
serve && probe; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 10
COND=B_student_tpl; stamp "6) 조건 B — 교사 서버를 학생 템플릿으로"
serve --chat-template "$W/student.jinja" && probe; kill $SRV 2>/dev/null
stamp "요약"; grep -h "판정" "$W"/check_*.txt
stamp "done"
