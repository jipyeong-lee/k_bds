#!/usr/bin/env bash
# timing_bench.sh — kgpu pod 1장(같은 GPU)에서 base · SFT · 통합 모델(ta_w07)의 추론 비용을 잰다.
# 왜: 평가 기록의 소요 시간은 GPU 가 서로 달라(KISTI A100 대 kgpu RTX 5090) 비교할 수 없다.
# 측정: holdout 층화 표본에서 동시 1요청(문항당 지연) 48문항 + 동시 16요청(처리량) 198문항, 정확도도 함께 기록.
# 전제(~/files = /root/files): kbds/{code.tgz, setup_venv.sh, merge_experts.py, eval_compare.py, sft_mixed_merged/,
#        holdout.tar, ckpt_{dv,mm,pmc}.tar}
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs"; exec > >(tee -a "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
nvidia-smi -L

stamp "1) code·data → pod 로컬"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
cp "$F/eval_compare.py" "$W/scripts/eval_compare.py" || { echo "❌ eval_compare"; exit 1; }   # 지연·토큰 기록판
cp "$F/merge_experts.py" "$W/merge_experts.py"     # ~/files 에서 바로 실행하면 Errno 121
mkdir -p "$W/hold"; tar xf "$F/holdout.tar" -C "$W/hold" --no-same-owner || { echo "❌ holdout"; exit 1; }
sed -i "s#/home01/k266a01/kbds_project/work/data/images#$W/hold/images#g" "$W/hold/stage2_expanded_holdout.jsonl"

stamp "2) venv"
. "$F/setup_venv.sh"

stamp "3) 모델 3종 준비"
"$PY" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a', local_dir='$W/qwen')" \
  || { echo "❌ download base"; exit 1; }
[ -d "$W/sft" ] || cp -r "$F/sft_mixed_merged" "$W/sft" || { echo "❌ sft copy"; exit 1; }
A=""
for e in dv mm pmc; do
  mkdir -p "$W/ck_$e"; tar xf "$F/ckpt_$e.tar" -C "$W/ck_$e" --no-same-owner || { echo "❌ ckpt_$e"; exit 1; }
  A="$A${A:+,}$(ls -d "$W"/ck_$e/checkpoint-*)"
done
[ -f "$W/ta_w07/merge_config.json" ] || "$PY" "$W/merge_experts.py" --base "$W/sft" --adapters "$A" \
  --weights 0.7,0.7,0.7 --method linear --out "$W/ta_w07" || { echo "❌ merge ta_w07"; exit 1; }

for tag in base sft ta_w07; do
  model=$W/$tag; [ $tag = base ] && model=$W/qwen
  "$VENV/bin/vllm" serve "$model" --served-model-name evalmodel --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --enforce-eager --limit-mm-per-prompt '{"image": 8}' \
    --host 127.0.0.1 --port 8000 > "$W/serve_$tag.log" 2>&1 &
  srv=$!
  for i in $(seq 1 120); do
    curl -s http://127.0.0.1:8000/v1/models | grep -q evalmodel && break
    kill -0 $srv 2>/dev/null && [ "$i" -lt 120 ] || { echo "❌ serve $tag"; tail -20 "$W/serve_$tag.log"; kill $srv; continue 2; }
    sleep 10
  done
  stamp "[$tag] READY"
  for run in "c1 48 1" "c16 198 16"; do
    set -- $run
    SYSTEM_PROMPT="$SYS" EVAL_BASE_URL=http://127.0.0.1:8000/v1 EVAL_MODEL=evalmodel EVAL_TAG="$tag" \
      EVAL_DATA="$W/hold/stage2_expanded_holdout.jsonl" EVAL_N=$2 EVAL_CONC=$3 EVAL_MAXTOK=8192 \
      EVAL_RESULT="$F/logs/timing_${tag}_$1.jsonl" EVAL_ITEMS="$F/logs/timing_items_${tag}_$1.jsonl" \
      "$PY" "$W/scripts/eval_compare.py" || echo "❌ eval $tag $1"
  done
  kill $srv; sleep 15
done
cp "$W/run.log" "$F/logs/timing_run_$(hostname).log"
stamp "done"
