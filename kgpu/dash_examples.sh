#!/usr/bin/env bash
# dash_examples.sh — kgpu pod 1장에서 ta_w07 이 대시보드 예시 30문항에 답하게 하고, ta_w07·27B 답변을 둘 다 채점한다.
# 전제(~/files/kbds): code.tgz setup_venv.sh dash_examples.tar(examples.json answers_27b.json assets/examples/*.png)
# env(필수): HF_TOKEN(모델 저장소가 비공개) / env(선택): MODEL_HF
# 산출: ~/files/kbds/logs/dash_examples_out.json = {id: {ta_w07: {text, score}, q27b: {score}}}
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
MODEL_HF=${MODEL_HF:-easypyeong/Qwen3.5-9B-MedCrossReason}
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W/dash" "$F/logs"; exec > >(tee "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L
export SYS="You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."

stamp "1) code·data"
tar xzf "$F/code.tgz" -C "$W" --no-same-owner || { echo "❌ code.tgz"; exit 1; }
tar xf "$F/dash_examples.tar" -C "$W/dash" --no-same-owner || { echo "❌ dash_examples.tar"; exit 1; }

stamp "2) venv"
. "$F/setup_venv.sh"

stamp "3) model ← HF $MODEL_HF"
"$PY" -c "from huggingface_hub import snapshot_download as s;s('$MODEL_HF',local_dir='$W/model')" || { echo "❌ model download"; exit 1; }

stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/model" --served-model-name evalmodel --max-model-len 10240 \
  --gpu-memory-utilization 0.85 --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
  --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q evalmodel && { stamp "READY"; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }
  sleep 10
done

stamp "5) 30문항 생성·채점"
W="$W" "$PY" - <<'EOF'
import base64, json, os, sys
sys.path.insert(0, os.path.join(os.environ['W'], 'configs'))
import accuracy as A
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
W = os.environ['W']; D = f'{W}/dash'
items = json.load(open(f'{D}/examples.json')); q27 = json.load(open(f'{D}/answers_27b.json'))
cli = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='EMPTY'); am = A.AccuracyMix()

def run(it):
    b64 = base64.b64encode(open(f'{D}/{it["image"]}', 'rb').read()).decode()
    r = cli.chat.completions.create(model='evalmodel', temperature=0.0, max_tokens=4096, messages=[
        {'role': 'system', 'content': os.environ['SYS']},
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}},
                                      {'type': 'text', 'text': it['question']}]}])
    return it['id'], r.choices[0].message.content or '', r.choices[0].finish_reason

with ThreadPoolExecutor(4) as ex: res = list(ex.map(run, items))
out = {}
for (i, text, fin), it in zip(res, items):
    out[i] = {'ta_w07': {'text': text, 'finish_reason': fin, 'score': float(am([text], [it['solution']])[0])},
              'q27b': {'score': float(am([q27[i]['text']], [it['solution']])[0])}}
json.dump(out, open(f'{W}/dash_examples_out.json', 'w'), ensure_ascii=False, indent=1)
for m in ('ta_w07', 'q27b'):
    print(m, 'acc', sum(v[m]['score'] for v in out.values()) / len(out))
EOF
echo "GEN_EXIT=$?"
[ -s "$W/dash_examples_out.json" ] && cp "$W/dash_examples_out.json" "$F/logs/dash_examples_out.json" || echo "❌ 산출 없음"
cp "$W/run.log" "$F/logs/dash_examples_run.log"
grep -a -E "Error|Traceback" "$W/serve.log" | tail -n 20 > "$F/logs/dash_examples_serve_err.log"
stamp "done"
