#!/usr/bin/env bash
# loco_eval.sh — 풀이 줄이기 5단계: 학습 때 쓰지 않은 deepvision 문항(300개 뒤 N_Q개)에서 원본 vs 어댑터의 답 길이·정답률·길이 상한 도달률을 잰다.
# 전제(~/files/kbds): setup_venv.sh · opd_dv/{stage2_deepvision.jsonl,dv_images.tar} · loco/adapter_v5(LoRA)
# env(필수): HF_TOKEN. env(선택): N_Q(150) N_GEN(2) MAX_TOK(8192)
# 실행: MAX_SEC=5400 setsid nohup bash kgpu/run_pod.sh loco_eval.sh HF_TOKEN > logs/loco_eval.out 2>&1 < /dev/null & disown
# 출력: ~/files/kbds/loco/eval_med_v5.jsonl · 마지막 줄에 요약
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs" "$F/loco"; exec > >(tee "$W/eval.log") 2>&1
trap 'cp "$W/eval.log" "$F/logs/loco_eval_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L; ls "$F/loco/adapter_v5" || { echo "❌ adapter_v5 없음"; exit 1; }
stamp "1) venv"
. "$F/setup_venv.sh"
stamp "2) 문항 고르기(의료 글로벌 벤치 MedXpertQA-MM·MMMU-H&M 에서 무작위 ${N_Q:=200}개, 시드 0)"
IMG=$W/in/images; mkdir -p "$IMG" "$W/gb"
tar xf "$F/global_bench.tar" -C "$W/gb" --no-same-owner || { echo "❌ global_bench.tar"; exit 1; }
$PY - "$W/gb/global_bench.jsonl" "$W/in/rows.jsonl" "$N_Q" <<'PY' || { echo "❌ 문항 고르기"; exit 1; }
import json, os, random, re, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for i, l in enumerate(open(src)):
    d = json.loads(l)
    rows.append({'id': f"{d['_source']}-{i}", 'question': re.sub(r'\s*<image>\s*', ' ', d['messages'][0]['content']).strip(), 'solution': str(d['solution']).strip(),
                 'images': [os.path.basename(p) for p in d['images']]})
random.Random(0).shuffle(rows)
with open(out, 'w') as f:
    for r in rows[:n]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'후보 {len(rows)} → 선택 {min(n, len(rows))}')
PY
$PY -c "import json;[print('images/'+x) for l in open('$W/in/rows.jsonl') for x in json.loads(l)['images']]" | sort -u > "$W/need.txt"
tar xf "$F/global_bench.tar" -C "$W/gb" -T "$W/need.txt" --no-same-owner; mv "$W/gb/images/"* "$IMG/"
ls "$IMG" | wc -l

stamp "3) 모델 ← HF"
for TRY in 1 2 3; do
  HF_HUB_DISABLE_XET=$([ $TRY -gt 1 ] && echo 1 || echo 0) timeout 480 $PY -c "from huggingface_hub import snapshot_download as s;s('${MODEL_REPO:-Qwen/Qwen3.5-9B}',local_dir='$W/model')" && break
  echo "내려받기 ${TRY}번째 실패 — 다시 시도"; [ $TRY = 3 ] && { echo "❌ model"; exit 1; }
done
stamp "4) vLLM serve(원본 + LoRA)"
"$VENV/bin/vllm" serve "$W/model" --served-model-name policy --enable-lora --max-lora-rank 256 --lora-modules loco5=$F/loco/adapter_v5 --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' --gpu-memory-utilization 0.9 \
  --enforce-eager --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q policy && { stamp READY; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }; sleep 10
done
stamp "5) 평가 N_GEN=${N_GEN:=2} MAX_TOK=${MAX_TOK:=8192}"
N_GEN=$N_GEN MAX_TOK=$MAX_TOK $PY - "$W/in" "$F/loco/eval_med_v5.jsonl" <<'PY'
import asyncio, base64, json, os, re, sys
from openai import AsyncOpenAI
from math_verify import parse, verify
src, out = sys.argv[1:3]
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")
cli = AsyncOpenAI(base_url='http://127.0.0.1:8000/v1', api_key='x', timeout=1800); sem = asyncio.Semaphore(16)
rows = [json.loads(l) for l in open(f'{src}/rows.jsonl')]
def ok(text, sol):   # <answer> 안의 내용만 비교한다. 없으면 오답
    m = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    if not m: return False
    a = m[-1].strip(); s = sol.strip()
    if a.lower() == s.lower(): return True
    try: return bool(verify(parse(f'${s}$'), parse(f'${a}$')))
    except Exception: return False
async def one(r, name):
    imgs = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(open(f"{src}/images/{p}", 'rb').read()).decode()}} for p in r['images']]
    async with sem:
        c = await cli.chat.completions.create(model=name, n=int(os.environ['N_GEN']), temperature=0.6, top_p=0.95, max_tokens=int(os.environ['MAX_TOK']),
            messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': imgs + [{'type': 'text', 'text': r['question']}]}])
    comp = [x.message.content or '' for x in c.choices]
    return {'id': r['id'], 'model': name, 'completions': comp, 'finish': [x.finish_reason for x in c.choices], 'correct': [ok(t, r['solution']) for t in comp]}
async def main():
    res = await asyncio.gather(*(one(r, n) for n in ('loco5',) for r in rows), return_exceptions=True)
    good = [x for x in res if isinstance(x, dict)]
    with open(out, 'w') as f:
        for x in good: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('오류', [repr(x)[:200] for x in res if not isinstance(x, dict)][:2])
    for n in ('loco5',):
        g = [x for x in good if x['model'] == n]; L = [len(t.split()) for x in g for t in x['completions']]
        k = sum(len(x['correct']) for x in g)
        print(f"SUMMARY {n}: 응답 {k} · 정답률 {sum(sum(x['correct']) for x in g)/max(k,1):.3f} · 평균 단어 {sum(L)/max(len(L),1):.0f} · 중앙값 {sorted(L)[len(L)//2] if L else 0} · 길이 상한 도달 {sum(f=='length' for x in g for f in x['finish'])/max(k,1):.3f} · </think> 포함 {sum('</think>' in t for x in g for t in x['completions'])/max(k,1):.3f}")
asyncio.run(main())
PY
stamp done
