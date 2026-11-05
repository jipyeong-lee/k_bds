#!/usr/bin/env bash
# diag_vis.sh — kgpu pod 1장에서 ta_w07 의 시각 의존도(VPPO S_t)를 holdout 에서 잰다. 학습 없음.
# 전제(~/files/kbds): code.tgz setup_venv.sh holdout.tar diag_visual_dependence.py
# env(필수): HF_TOKEN(모델 저장소가 비공개) / env(선택): MODEL_HF DIAG_N(100) DIAG_MAXTOK(4096)
# 먼저 소스당 1문항으로 토큰 정렬을 확인하고(사용 0 이면 중단), 통과하면 본 측정을 돈다.
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
MODEL_HF=${MODEL_HF:-easypyeong/Qwen3.5-9B-MedCrossReason}
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."

stamp "1) code·data"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
cp "$F/diag_visual_dependence.py" "$W/scripts/" || { echo "❌ diag py"; exit 1; }
mkdir -p "$W/hold"; tar xf "$F/holdout.tar" -C "$W/hold" --no-same-owner || { echo "❌ holdout"; exit 1; }
sed -i "s#/home01/k266a01/kbds_project/work/data/images#$W/hold/images#g" "$W/hold/stage2_expanded_holdout.jsonl"

stamp "2) venv"
. "$F/setup_venv.sh"

stamp "3) model ← HF $MODEL_HF"
"$PY" -c "from huggingface_hub import snapshot_download as s;s('$MODEL_HF',local_dir='$W/model')" || { echo "❌ model download"; exit 1; }

stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/model" --served-model-name evalmodel --max-model-len 10240 \
  --gpu-memory-utilization 0.6 --max-num-batched-tokens 4096 --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
  --chat-template-content-format openai --trust-request-chat-template --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q evalmodel && { stamp "READY"; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }
  sleep 10
done

run(){  # $1=소스당 N  $2=출력 json
  SYSTEM_PROMPT="$SYS" EVAL_BASE_URL=http://127.0.0.1:8000/v1 EVAL_MODEL=evalmodel TOKENIZER="$W/model" \
  EVAL_DATA="$W/hold/stage2_expanded_holdout.jsonl" DIAG_N=$1 DIAG_MAXTOK="${DIAG_MAXTOK:-4096}" DIAG_CONC="${DIAG_CONC:-4}" DIAG_OUT="$2" \
    "$PY" "$W/scripts/diag_visual_dependence.py"
}
stamp "5) 정렬 점검 (소스당 1문항)"
run 1 "$W/diag_check.json" | tee "$W/check.out"
grep -Eq '사용 [1-9][0-9]*/' "$W/check.out" || { echo "❌ 정렬 점검 실패(사용 0) — 본 측정 중단"; cp "$W/run.log" "$F/logs/diag_vis_fail.log"; exit 1; }

stamp "6) 본 측정 N=${DIAG_N:-100}/소스"
run "${DIAG_N:-100}" "$F/logs/diag_vis_ta_w07.json"
echo "DIAG_EXIT=$?"
cp "$W/run.log" "$F/logs/diag_vis_run.log"
grep -a -E "Error|OOM|out of memory|Traceback" "$W/serve.log" | tail -n 30 > "$F/logs/diag_vis_serve_err.log"   # 500 의 근본 원인을 남긴다
stamp "done"
