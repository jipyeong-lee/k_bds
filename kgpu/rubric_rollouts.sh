#!/usr/bin/env bash
# rubric_rollouts.sh — 루브릭 조합 실험(1단계, 학습 없음)용 고정 응답 세트를 만든다.
#   ta_w07 이 medix 100문항(시드 7)에 temp 1.0 으로 문항당 N_GEN(8)개씩 답한다 = 실제 GRPO 그룹과 같은 모양.
# 전제(~/files/kbds): setup_venv.sh · rubric/rubric_inputs.tar(rows.jsonl + images/, 로컬 work/rubric_ablation 에서 만든 것)
# env(필수): HF_TOKEN(학생 저장소 비공개). env(선택): N_GEN(8) MAX_TOK(2048) TP(1 — rtx3090 24GB 면 GPU_COUNT=2 TP=2)
# 실행: GPU_MODEL=rtx5090 MAX_SEC=5400 setsid nohup bash kgpu/run_pod.sh rubric_rollouts.sh HF_TOKEN > logs/rubric_rollouts.out 2>&1 < /dev/null & disown
# 출력: ~/files/kbds/rubric/rubric_rollouts.jsonl — 행마다 rows.jsonl 필드 + completions[N_GEN] · finish[N_GEN]
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs" "$F/rubric"; exec > >(tee "$W/rollouts.log") 2>&1
trap 'cp "$W/rollouts.log" "$F/logs/rubric_rollouts_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) 입력"
mkdir -p "$W/in" && tar xf "$F/rubric/rubric_inputs.tar" -C "$W/in" --no-same-owner || { echo "❌ rubric_inputs.tar"; exit 1; }
wc -l "$W/in/rows.jsonl"
stamp "2) venv"
. "$F/setup_venv.sh"
stamp "3) ta_w07 ← HF"
$PY -c "from huggingface_hub import snapshot_download as s;s('easypyeong/Qwen3.5-9B-MedCrossReason',local_dir='$W/model')" || { echo "❌ model"; exit 1; }
stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/model" --served-model-name policy --max-model-len 6144 --gpu-memory-utilization 0.9 --tensor-parallel-size "${TP:-1}" \
  --enforce-eager --limit-mm-per-prompt '{"image": 1}' --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q policy && { stamp READY; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }; sleep 10
done
stamp "5) 롤아웃 N_GEN=${N_GEN:=8} MAX_TOK=${MAX_TOK:=2048}"
N_GEN=$N_GEN MAX_TOK=$MAX_TOK $PY - "$W/in" "$F/rubric/rubric_rollouts.jsonl" <<'PY'
import asyncio, base64, json, os, sys
from openai import AsyncOpenAI
src, out = sys.argv[1:3]
SYS = ("You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, "
       "keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.")  # kgpu/stage3_smoke.sh 와 같은 문구
cli = AsyncOpenAI(base_url='http://127.0.0.1:8000/v1', api_key='x', timeout=900); sem = asyncio.Semaphore(8)
rows = [json.loads(l) for l in open(f'{src}/rows.jsonl')]
async def one(r):
    img = 'data:image/png;base64,' + base64.b64encode(open(f"{src}/images/{r['image']}", 'rb').read()).decode()
    async with sem:
        c = await cli.chat.completions.create(model='policy', n=int(os.environ['N_GEN']), temperature=1.0, max_tokens=int(os.environ['MAX_TOK']),
            messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img}}, {'type': 'text', 'text': r['question']}]}])
    # 생성 프롬프트 끝의 '<think>\n' 은 응답에 안 들어온다. ms-swift 도 보상 함수에 넘기기 전에 다시 붙인다(template/base.py:649-650)
    fix = lambda t: t if t.lstrip().startswith('<think>') else '<think>\n' + t
    return {**r, 'completions': [fix(x.message.content or '') for x in c.choices], 'finish': [x.finish_reason for x in c.choices]}
async def main():
    res = await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
    ok = [x for x in res if isinstance(x, dict)]
    with open(out, 'w') as f:
        for x in ok: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    errs = [repr(x)[:200] for x in res if not isinstance(x, dict)]
    n = sum(len(x['completions']) for x in ok); cut = sum(f == 'length' for x in ok for f in x['finish'])
    print(f'문항 {len(ok)}/{len(rows)} · 응답 {n} · 길이 상한 도달 {cut} · 오류 {len(errs)} {errs[:2]}')
asyncio.run(main())
PY
echo "ROLLOUT_EXIT=$?"
stamp "done"
