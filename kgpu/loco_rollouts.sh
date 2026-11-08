#!/usr/bin/env bash
# loco_rollouts.sh — 풀이 줄이기 방법(LoCo) B안 2단계: 원본 모델이 deepvision 문항에 3번씩 답한다(temperature 0.6, 최대 8192 토큰).
#   정답 비교(math_verify)까지 해서, 문항마다 "맞은 답 중 가장 짧은 것"을 다음 단계(확신도 계산)에 넘길 수 있게 한다.
# 전제(~/files/kbds): setup_venv.sh · loco_api_answers.py · opd_dv/{stage2_deepvision.jsonl,dv_images.tar}
# env(필수): HF_TOKEN. env(선택): N_Q(300 문항) N_GEN(3) MAX_TOK(8192) MODEL_REPO(Qwen/Qwen3.5-9B)
# 실행: MAX_SEC=7200 setsid nohup bash kgpu/run_pod.sh loco_rollouts.sh HF_TOKEN > logs/loco_rollouts.out 2>&1 < /dev/null & disown
# 출력: ~/files/kbds/loco/api_answers.jsonl(큰 Qwen API 답) · ~/files/kbds/loco/rollouts.jsonl — 행마다 id·question·image·solution·completions[N_GEN]·finish[N_GEN]·correct[N_GEN]
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs" "$F/loco"; exec > >(tee "$W/loco.log") 2>&1
trap 'cp "$W/loco.log" "$F/logs/loco_rollouts_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) venv"
. "$F/setup_venv.sh"
stamp "2) 문항 고르기(deepvision, 정답이 짧은 문항 중 무작위 ${N_Q:=300}개, 시드 0)"
IMG=$W/in/images; mkdir -p "$IMG"
$PY - "$F/opd_dv/stage2_deepvision.jsonl" "$W/in/rows.jsonl" "$N_Q" <<'PY' || { echo "❌ 문항 고르기"; exit 1; }
import json, os, random, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); off = int(os.environ.get('OFFSET', 0))
rows = []
for i, l in enumerate(open(src)):
    d = json.loads(l); sol = str(d.get('solution', '')).strip()
    if len(sol) <= 30 and d.get('images') and len(d['images']) == 1:
        rows.append({'id': f'dv-{i}', 'question': d['messages'][0]['content'].replace('<image>\n', '', 1), 'solution': sol, 'image': os.path.basename(d['images'][0])})
random.Random(0).shuffle(rows)
with open(out, 'w') as f:
    for r in rows[off:off + n]: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'후보 {len(rows)} → 선택 {min(n, len(rows))}')
PY
$PY -c "import json;[print(json.loads(l)['image']) for l in open('$W/in/rows.jsonl')]" > "$W/need.txt"
tar xf "$F/opd_dv/dv_images.tar" -C "$IMG" -T "$W/need.txt" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }
ls "$IMG" | wc -l

stamp "2.5) 큰 Qwen API 답 받기(백그라운드) — ${API_MODEL:-qwen3.8-flash-next-180b-a6b}"
( [ "${SKIP_API:-0}" = 1 ] || $PY "$F/loco_api_answers.py" "$W/in" "$F/loco/api_answers${TAG:-}.jsonl" "${API_MODEL:-qwen3.8-flash-next-180b-a6b}" ${API_N:-3} > "$W/api.log" 2>&1; cp "$W/api.log" "$F/logs/loco_api_$(date +%m%d-%H%M).log" ) &
API_PID=$!
stamp "3) 원본 모델 ← HF"
for TRY in 1 2 3; do
  HF_HUB_DISABLE_XET=$([ $TRY -gt 1 ] && echo 1 || echo 0) timeout 480 $PY -c "from huggingface_hub import snapshot_download as s;s('${MODEL_REPO:-Qwen/Qwen3.5-9B}',local_dir='$W/model')" && break
  echo "내려받기 ${TRY}번째 실패 — 다시 시도"; [ $TRY = 3 ] && { echo "❌ model"; exit 1; }
done
stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/model" --served-model-name policy --max-model-len 12288 --gpu-memory-utilization 0.9 \
  --enforce-eager --limit-mm-per-prompt '{"image": 1}' --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q policy && { stamp READY; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }; sleep 10
done
stamp "5) 롤아웃 N_GEN=${N_GEN:=3} MAX_TOK=${MAX_TOK:=8192}"
N_GEN=$N_GEN MAX_TOK=$MAX_TOK $PY - "$W/in" "$F/loco/rollouts${TAG:-}.jsonl" <<'PY'
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
async def one(r):
    img = 'data:image/png;base64,' + base64.b64encode(open(f"{src}/images/{r['image']}", 'rb').read()).decode()
    async with sem:
        c = await cli.chat.completions.create(model='policy', n=int(os.environ['N_GEN']), temperature=0.6, top_p=0.95, max_tokens=int(os.environ['MAX_TOK']),
            messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}}, {'type': 'text', 'text': r['question']}]}])
    fix = lambda t: t if t.lstrip().startswith('<think>') else '<think>\n' + t
    comp = [fix(x.message.content or '') for x in c.choices]
    return {**r, 'completions': comp, 'finish': [x.finish_reason for x in c.choices], 'correct': [ok(t, r['solution']) for t in comp]}
async def main():
    res = await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
    good = [x for x in res if isinstance(x, dict)]
    with open(out, 'w') as f:
        for x in good: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    errs = [repr(x)[:200] for x in res if not isinstance(x, dict)]
    tot = sum(len(x['correct']) for x in good); right = sum(sum(x['correct']) for x in good)
    cut = sum(f == 'length' for x in good for f in x['finish']); anyok = sum(any(x['correct']) for x in good)
    print(f'문항 {len(good)}/{len(rows)} · 응답 {tot} · 정답 {right} · 한 번이라도 맞은 문항 {anyok} · 길이 상한 도달 {cut} · 오류 {len(errs)} {errs[:2]}')
asyncio.run(main())
PY
echo "ROLLOUT_EXIT=$?"
wait $API_PID; stamp "done"
