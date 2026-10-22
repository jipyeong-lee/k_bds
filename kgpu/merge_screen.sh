#!/usr/bin/env bash
# merge_screen.sh — kgpu pod 1장에서 전문가 3종 병합 후보를 차례로 만들고 축소 평가셋으로 선별한다.
# 왜: 계단 0(가중치 merging) 기법이 10종이라 전량 평가(5,484문항)는 비싸다 → 축소셋(holdout 600 + 글로벌 600)으로
#     고른 뒤 상위만 전량 평가. 판정 = 도메인별 전문가 대비 90% 보존 + 글로벌 SFT 대비 향상(docs/deepseek_v4_pipeline_adoption.md §8).
# 전제(~/files = /root/files): kbds/{code.tgz, setup_venv.sh, merge_experts.py, sft_mixed_merged/, holdout.tar,
#        global_bench.tar, ckpt_{dv,mm,pmc}.tar, merge_configs.tar}
# env: CANDS="sft exp_dv ta_w1 ties_d05 ..." — sft·exp_* 는 기준선(병합 없음), 나머지는 configs/merge/<이름>.yaml
#      본평가: HOLD_N=all GB_N=all RES_PREFIX=full · GB_N=0 이면 글로벌 생략
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv; MK=/root/mk
mkdir -p "$W" "$F/logs"; exec > >(tee -a "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
nvidia-smi -L

stamp "1) code·base·data·ckpt → pod 로컬"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
[ -d "$W/base" ] || cp -r "$F/sft_mixed_merged" "$W/base" || { echo "❌ base copy"; exit 1; }
mkdir -p "$W/hold" "$W/gb" "$W/ckpt" "$W/mcfg"
for t in holdout:hold global_bench:gb merge_configs:mcfg; do
  tar xf "$F/${t%%:*}.tar" -C "$W/${t##*:}" --no-same-owner || { echo "❌ ${t%%:*}"; exit 1; }
done
sed -i "s#/home01/k266a01/kbds_project/work/data/images#$W/hold/images#g" "$W/hold/stage2_expanded_holdout.jsonl"
sed -i "s#/home01/k266a01/kbds_project/work/data/global_bench/images#$W/gb/images#g" "$W/gb/global_bench.jsonl"
cp "$F/merge_experts.py" "$W/merge_experts.py"     # ~/files 에서 바로 실행하면 Errno 121

stamp "2) venv (+ mergekit 별도 venv — vLLM 핀을 건드리지 않게)"
. "$F/setup_venv.sh"
if [ ! -x "$MK/bin/mergekit-yaml" ]; then
  uv venv "$MK" --python 3.12 2>&1 | tail -1
  uv pip install --python "$MK/bin/python" torch==2.10.0+cu129 transformers==5.6.2 peft==0.19.1 \
    "mergekit @ https://github.com/arcee-ai/mergekit/archive/c91c787ff9293bda5d67bf59942700204f7ac2ba.tar.gz" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match 2>&1 | tail -2
fi

stamp "3) 전문가 3종 → 완전 모델 (merge_experts --weights 1.0)"
for e in dv mm pmc; do
  [ -f "$W/exp_$e/merge_config.json" ] && continue
  rm -rf "$W/ck_$e"; mkdir -p "$W/ck_$e"; tar xf "$F/ckpt_$e.tar" -C "$W/ck_$e" --no-same-owner || { echo "❌ ckpt_$e"; exit 1; }
  "$PY" "$W/merge_experts.py" --base "$W/base" --adapters "$(ls -d "$W"/ck_$e/checkpoint-*)" --weights 1.0 \
    --out "$W/exp_$e" || { echo "❌ expert $e"; exit 1; }
done

serve_eval() {  # $1=tag $2=model dir
  local tag=$1 model=$2
  "$VENV/bin/vllm" serve "$model" --served-model-name evalmodel --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --enforce-eager --limit-mm-per-prompt '{"image": 8}' \
    --host 127.0.0.1 --port 8000 > "$W/serve_$tag.log" 2>&1 &
  local srv=$!
  for i in $(seq 1 90); do
    curl -s http://127.0.0.1:8000/v1/models | grep -q evalmodel && break
    kill -0 $srv 2>/dev/null && [ "$i" -lt 90 ] || { echo "❌ serve $tag"; tail -20 "$W/serve_$tag.log"; kill $srv; return 1; }
    sleep 10
  done
  stamp "[$tag] READY"
  # 축소셋: EVAL_N 숫자 = _source 균등 층화(같은 seed → 후보 간 같은 문항). holdout 600=도메인별 200, 글로벌 600=벤치별 300
  local env=(SYSTEM_PROMPT="$SYS" EVAL_BASE_URL=http://127.0.0.1:8000/v1 EVAL_MODEL=evalmodel EVAL_CONC=16 EVAL_TAG="$tag")
  env "${env[@]}" EVAL_DATA="$W/hold/stage2_expanded_holdout.jsonl" EVAL_N=${HOLD_N:-600} EVAL_MAXTOK=8192 \
    EVAL_RESULT="$F/logs/${RES_PREFIX:-screen}_${tag}_hold.jsonl" EVAL_ITEMS="$F/logs/${RES_PREFIX:-screen}_items_${tag}_hold.jsonl" \
    "$PY" "$W/scripts/eval_compare.py" || echo "❌ hold $tag"
  [ "${GB_N:-600}" = 0 ] || env "${env[@]}" EVAL_DATA="$W/gb/global_bench.jsonl" EVAL_N=${GB_N:-600} EVAL_MAXTOK=4096 \
    EVAL_RESULT="$F/logs/${RES_PREFIX:-screen}_${tag}_gb.jsonl" EVAL_ITEMS="$F/logs/${RES_PREFIX:-screen}_items_${tag}_gb.jsonl" \
    "$PY" "$W/scripts/eval_global_bench.py" || echo "❌ gb $tag"
  kill $srv; sleep 15
}

for c in ${CANDS:?CANDS}; do
  case $c in
    sft)   stamp "4) [$c] 기준선"; serve_eval "$c" "$W/base" ;;
    exp_*) stamp "4) [$c] 기준선"; serve_eval "$c" "$W/$c" ;;
    *)
      stamp "4) [$c] mergekit"
      rm -rf "$W/m"
      # ties·della·breadcrumbs·ram 은 마스크 때문에 32GB GPU 에서 OOM(실측) → CPU 로 재시도(60Gi RAM)
      "$MK/bin/mergekit-yaml" "$W/mcfg/$c.yaml" "$W/m" --cuda --copy-tokenizer \
        || { stamp "[$c] GPU 실패 → CPU 재시도"; rm -rf "$W/m"; "$MK/bin/mergekit-yaml" "$W/mcfg/$c.yaml" "$W/m" --copy-tokenizer; } \
        || { echo "❌ mergekit $c"; continue; }
      cp -n "$W"/base/*.json "$W"/base/*.jinja "$W/m/" 2>/dev/null   # processor·chat_template 이 빠지면 vLLM 이 이미지를 못 읽는다
      cp "$W/mcfg/$c.yaml" "$F/logs/${RES_PREFIX:-screen}_${c}.yaml"
      serve_eval "$c" "$W/m" ;;
  esac
done
cp "$W/run.log" "$F/logs/${RES_PREFIX:-screen}_run_$(hostname).log"
stamp "done"
