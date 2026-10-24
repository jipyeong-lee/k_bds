#!/usr/bin/env bash
# card_test.sh — HF 모델 카드(easypyeong/Qwen3.5-9B-MedCrossReason)의 사용 예시 2개를 카드 그대로 실행해 본다.
# A) transformers 예시: holdout 의료 VQA 1문항. B) vLLM 서버 + OpenAI 클라이언트 예시: 도메인별 1문항씩 3문항.
# 전제(~/files = /root/files): kbds/{setup_venv.sh, holdout.tar}. pod env HF_TOKEN(비공개 저장소 읽기).
set -uo pipefail
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs"; exec > >(tee -a "$W/run.log") 2>&1
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
NAME=easypyeong/Qwen3.5-9B-MedCrossReason
nvidia-smi -L

stamp "2) venv"
. "$F/setup_venv.sh"

stamp "1) holdout 3문항(도메인별 첫 문항)"
mkdir -p "$W/hold"; tar xf "$F/holdout.tar" -C "$W/hold" --no-same-owner || { echo "❌ holdout"; exit 1; }
"$PY" - "$W" <<'EOF'
import json, sys
W = sys.argv[1]; seen = {}
for l in open(f'{W}/hold/stage2_expanded_holdout.jsonl'):
    d = json.loads(l); s = d['_source']
    if s not in seen:
        seen[s] = {'src': s, 'image': d['images'][0].replace('/home01/k266a01/kbds_project/work/data/images', f'{W}/hold/images'),
                   'question': d['messages'][0]['content'].replace('<image>\n', ''), 'solution': d['solution']}
json.dump(list(seen.values()), open(f'{W}/items.json', 'w'), ensure_ascii=False, indent=1)
print(len(seen), 'items')
EOF
[ -s "$W/items.json" ] || { echo "❌ items"; exit 1; }


cat > "$W/common.py" <<'EOF'
import json
SYSTEM = "You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>."
ITEMS = json.load(open('/root/w/items.json'))
EOF

stamp "3) A — transformers 예시(카드 코드 그대로, 의료 VQA 1문항)"
cat > "$W/test_tf.py" <<'EOF'
import re, time
from common import SYSTEM, ITEMS
from transformers import AutoModelForImageTextToText, AutoProcessor
it = next(i for i in ITEMS if i['src'] == 'pmcvqa')
t0 = time.time()
name = "easypyeong/Qwen3.5-9B-MedCrossReason"
model = AutoModelForImageTextToText.from_pretrained(name, dtype="bfloat16", device_map="auto")
proc = AutoProcessor.from_pretrained(name)
msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "path": it['image']},
                                     {"type": "text", "text": it['question']}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
t1 = time.time()
out = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
text = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
m = re.search(r"<answer>(.*?)</answer>", text, re.S)
print('TF_OUTPUT_TAIL', repr(text[-400:]))
print('TF_RESULT', {'answer': m and m.group(1).strip(), 'solution': it['solution'], 'gen_tokens': out.shape[1] - inputs["input_ids"].shape[1],
                    'load_s': round(t1 - t0), 'gen_s': round(time.time() - t1)})
EOF
(cd "$W" && "$PY" test_tf.py) || echo "❌ TF 예시 실패"

[ "${ONLY_TF:-0}" = 1 ] && { cp "$W/run.log" "$F/logs/card_test.log"; stamp "done (ONLY_TF)"; exit 0; }
stamp "4) B — vLLM 서버(카드 명령 그대로)"
"$VENV/bin/vllm" serve $NAME --max-model-len 12288 --limit-mm-per-prompt '{"image": 8}' > "$W/serve.log" 2>&1 &
srv=$!
for i in $(seq 1 90); do
  curl -s http://localhost:8000/v1/models | grep -q MedCrossReason && break
  kill -0 $srv 2>/dev/null || { echo "❌ serve 종료"; tail -30 "$W/serve.log"; break; }
  sleep 10
done
stamp "vLLM READY"
cat > "$W/test_vllm.py" <<'EOF'
import base64, re
from openai import OpenAI
from common import SYSTEM, ITEMS
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
for it in ITEMS:
    img = base64.b64encode(open(it['image'], "rb").read()).decode()
    r = client.chat.completions.create(
        model="easypyeong/Qwen3.5-9B-MedCrossReason", temperature=0, max_tokens=8192,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                      {"type": "text", "text": it['question']}]}])
    text = r.choices[0].message.content
    answer = re.search(r"<answer>(.*?)</answer>", text, re.S).group(1).strip()
    print('VLLM_RESULT', {'src': it['src'], 'answer': answer, 'solution': it['solution'], 'gen_tokens': r.usage.completion_tokens})
EOF
(cd "$W" && "$PY" test_vllm.py) || echo "❌ vLLM 예시 실패"
kill $srv
cp "$W/run.log" "$F/logs/card_test.log"
stamp "done"
