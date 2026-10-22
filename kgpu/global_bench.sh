#!/usr/bin/env bash
# global_bench.sh — kgpu pod 1장에서 전문가 LoRA 를 병합해 MedXpertQA-MM + MMMU-H&M 글로벌 벤치를 돈다.
# 왜: KISTI 는 사용자당 job 제출 한도(QOSMaxSubmitJobPerUserLimit)와 1gpu 대기로 막혔다.
# 설정은 scripts/eval_global_bench.slurm 의 rl 단계 그대로(max-model-len 12288, 이미지 8장, temp 0, max_tokens 4096).
# 전제(~/files = /root/files): kbds/{code.tgz, setup_venv.sh, merge_experts.py, sft_mixed_merged/, global_bench.tar, ckpt_$TAG.tar}
# env: TAG(필수, 예 dv2700) — ckpt_$TAG.tar 안에 checkpoint-* 디렉터리 1개
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv; TAG=${TAG:?TAG}
mkdir -p "$W" "$F/logs"; exec > >(tee "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) code·base·data·ckpt → pod 로컬"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
[ -d "$W/base" ] || cp -r "$F/sft_mixed_merged" "$W/base" || { echo "❌ base copy"; exit 1; }
mkdir -p "$W/gb" "$W/ckpt"
tar xf "$F/global_bench.tar" -C "$W/gb" --no-same-owner || { echo "❌ global_bench"; exit 1; }
sed -i "s#/home01/k266a01/kbds_project/work/data/global_bench/images#$W/gb/images#g" "$W/gb/global_bench.jsonl"
tar xf "$F/ckpt_$TAG.tar" -C "$W/ckpt" --no-same-owner || { echo "❌ ckpt"; exit 1; }
CKPT=$(ls -d "$W"/ckpt/checkpoint-* | head -1)

stamp "2) venv"
. "$F/setup_venv.sh"

stamp "3) merge $CKPT"
# swift export 는 체크포인트 args.json 의 KISTI 경로(external_plugins)를 찾다 죽는다 → 가중치 산술로 직접 병합
# ~/files 에서 바로 실행하면 네트워크 저장소 I/O 오류(Errno 121)가 난다 → 로컬로 복사해서 실행
cp "$F/merge_experts.py" "$W/merge_experts.py" || { echo "❌ cp merge_experts"; exit 1; }
"$VENV/bin/python" "$W/merge_experts.py" --base "$W/base" --adapters "$CKPT" --weights 1.0 \
  --out "$W/merged" || { echo "❌ merge"; exit 1; }

stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/merged" --served-model-name evalmodel \
  --max-model-len 12288 --gpu-memory-utilization 0.85 --enforce-eager \
  --limit-mm-per-prompt '{"image": 8}' --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q evalmodel && { stamp "READY"; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }
  sleep 10
done

stamp "5) eval"
RES="$F/logs/gbench_$TAG.jsonl"; : > "$RES"
SYSTEM_PROMPT="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>." \
EVAL_BASE_URL=http://127.0.0.1:8000/v1 EVAL_MODEL=evalmodel EVAL_DATA="$W/gb/global_bench.jsonl" \
EVAL_N=all EVAL_CONC=16 EVAL_TAG="$TAG" EVAL_MAXTOK=4096 EVAL_RESULT="$RES" \
EVAL_ITEMS="$F/logs/gbench_items_$TAG.jsonl" \
  "$VENV/bin/python" "$W/scripts/eval_global_bench.py"
echo "EVAL_EXIT=$?"; cat "$RES"
cp "$W/run.log" "$F/logs/gbench_$TAG.log"
stamp "done"
