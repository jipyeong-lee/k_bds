#!/usr/bin/env bash
# opd_server_probe46.sh — ms-swift 4.6.0 의 GKD 교사 서버 경로를 rtxpro6000 1장에서 판정한다(2 step, 학습 목적 아님).
#   4.6.0 은 교사로 swift rollout 서버(/infer/)를 쓰고, 학생이 샘플링한 토큰 id 를 그대로 보내며, 위치를 앞에서부터 맞춘다.
#   → 교사 프롬프트가 학생 인코딩과 토큰 단위로 같아야 한다. Qwen3.8-27B 는 4.6.0 에서 템플릿 qwen3_8 로 등록돼 있다.
#   조건 A = 교사 기본 템플릿(qwen3_8) · 조건 B = 학생과 같은 qwen3_5
#   판정(opd_server_probe.py check46): C1 교사 요청에 학생 답이 있는가 · C3 학생 응답 위치의 다음 토큰이 교사 top-20 에 드는 비율
#   (위치가 맞으면 99% 근처, 밀리면 10% 근처 — 4.1.3 실측 9~11%)
# 전제(~/files/kbds): code.tgz setup_venv.sh opd_server_probe.py opd_dv/{stage2_deepvision.jsonl,dv_images.tar}
# env(필수): HF_TOKEN. env(선택): MAX_COMP(1024) MAX_STEPS(2) SMOKE_ROWS(16) TEACHER_UTIL(0.40) STUDENT_VLLM_UTIL(0.30)
# 실행: GPU_MODEL=rtxpro6000 MAX_SEC=7200 setsid nohup bash kgpu/run_pod.sh opd_server_probe46.sh HF_TOKEN > logs/opd_server_probe46.out 2>&1 < /dev/null & disown
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv; LOG=$F/logs/opd_server_probe46_$(date +%m%d-%H%M).log
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/probe.log") 2>&1
trap 'cp "$W/probe.log" "$LOG"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTORCH_ALLOC_CONF=expandable_segments:True
MAX_COMP=${MAX_COMP:-1024}; MAX_STEPS=${MAX_STEPS:-2}; SMOKE_ROWS=${SMOKE_ROWS:-16}; PORT=8001
TEACHER_UTIL=${TEACHER_UTIL:-0.40}; STUDENT_VLLM_UTIL=${STUDENT_VLLM_UTIL:-0.30}; MAXPIX=262144
nvidia-smi -L

stamp "1) code·입력"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
cp "$F/opd_server_probe.py" "$W/" || { echo "❌ probe"; exit 1; }

stamp "2) venv(4.1.3 핀) → ms-swift 4.6.0 · datasets 4.0.0 으로 올림(4.6.0 요구: datasets>=4.0)"
. "$F/setup_venv.sh"
uv pip install --python "$PY" ms-swift==4.6.0 datasets==4.0.0 2>&1 | tail -3
$PY -c "import swift, datasets; print('swift', swift.__version__, 'datasets', datasets.__version__)" || { echo "❌ 4.6.0 import"; exit 1; }

stamp "3) 모델 병렬"
$PY -c "from huggingface_hub import snapshot_download as s;s('easypyeong/Qwen3.5-9B-MedCrossReason',local_dir='$W/student')" & P1=$!
$PY -c "from huggingface_hub import snapshot_download as s;s('Qwen/Qwen3.8-27B-FP8',local_dir='$W/teacher')" & P2=$!
wait $P1 || { echo "❌ student download"; exit 1; }; wait $P2 || { echo "❌ teacher download"; exit 1; }

stamp "4) deepvision 프롬프트 ${SMOKE_ROWS}행"
DATASET=$W/data/dv.jsonl; IMG=$W/data/images/deepvision; mkdir -p "$IMG"
head -n "$SMOKE_ROWS" "$F/opd_dv/stage2_deepvision.jsonl" | sed "s#/home01/k266a01/kbds_project/work/data/images/deepvision/#$IMG/#g" > "$DATASET"
grep -oE "$IMG/[^\"]+" "$DATASET" | sed "s#^$IMG/##" > "$W/dv_need.txt"
tar xf "$F/opd_dv/dv_images.tar" -C "$IMG" -T "$W/dv_need.txt" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }

SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
SRV=
serve(){   # $1 = 교사 템플릿
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/swift" rollout --model "$W/teacher" --model_type qwen3_5 --template "$1" \
    --vllm_gpu_memory_utilization $TEACHER_UTIL --vllm_max_model_len $(( 4096 + MAX_COMP )) --vllm_enforce_eager true \
    --vllm_limit_mm_per_prompt '{"image": 1}' --max_pixels $MAXPIX --host 127.0.0.1 --port $PORT > "$W/serve_$COND.log" 2>&1 & SRV=$!
  for i in $(seq 1 90); do
    curl -s http://127.0.0.1:$PORT/health/ >/dev/null 2>&1 && { stamp "교사 READY ($COND)"; return 0; }
    kill -0 $SRV 2>/dev/null || break; sleep 10
  done
  echo "❌ 교사 서버 기동 실패 ($COND)"; tail -40 "$W/serve_$COND.log"; return 1
}
probe(){
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 \
  CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 "$PY" "$W/opd_server_probe.py" run46 "$W/probe_$COND" -- --rlhf_type gkd \
    --model "$W/student" --model_type qwen3_5 --system "$SYS" --dataset "$DATASET" \
    --teacher_model_server http://127.0.0.1:$PORT --gkd_logits_topk 20 \
    --lmbda 1.0 --beta 1.0 --seq_kd false --sft_alpha 0 --temperature 1.0 \
    --tuner_type lora --lora_rank 32 --lora_alpha 32 --use_rslora true --lora_dropout 0 --target_modules all-linear --torch_dtype bfloat16 \
    --enable_thinking true --max_completion_length $MAX_COMP \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --learning_rate 1e-5 \
    --max_length $(( 2048 + MAX_COMP )) --max_pixels $MAXPIX --gradient_checkpointing true --attn_impl sdpa \
    --use_vllm true --vllm_mode colocate --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization $STUDENT_VLLM_UTIL --vllm_max_model_len $(( 3072 + MAX_COMP )) --sleep_level 1 \
    --logging_steps 1 --save_strategy no --report_to none --output_dir "$W/runs/$COND" --max_steps $MAX_STEPS
  echo "SWIFT_EXIT_$COND=$?"
  "$PY" "$W/opd_server_probe.py" check46 "$W/probe_$COND" | tee "$W/check_$COND.txt"
  mkdir -p "$F/logs/opd_probe46_$COND"; cp "$W/check_$COND.txt" "$W"/probe_$COND/*.json "$F/logs/opd_probe46_$COND/" 2>/dev/null
  grep -a -E "Error|Traceback|differs from" "$W/serve_$COND.log" | tail -n 20 > "$F/logs/opd_probe46_${COND}_serve_err.log"
}
stop(){ kill $SRV 2>/dev/null; sleep 5; G=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ -n "$G" ] && kill -9 $G 2>/dev/null; sleep 5; }

COND=A_qwen3_8; stamp "5) 조건 A — 교사 템플릿 qwen3_8(기본)"
serve qwen3_8 && probe; stop
COND=B_qwen3_5; stamp "6) 조건 B — 교사 템플릿 qwen3_5(학생과 같게)"
serve qwen3_5 && probe; stop
stamp "요약"; grep -h "판정46" "$W"/check_*.txt
stamp "done"
