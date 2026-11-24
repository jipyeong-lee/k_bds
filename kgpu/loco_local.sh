#!/usr/bin/env bash
# loco_local.sh — 풀이 줄이기 지점 단위 짝 시험(탐지기 없는 UniCut, LoCo arXiv 2611.04814 변형): 기존 롤아웃의 가장 짧은 정답을 균등 간격 지점에서 잘라
#   원본 모델에게 이어 쓰게 하고(vLLM), 정답이고 엄격 형식인 가장 짧은 이어쓰기를 좋은 답, 원래 연속을 나쁜 답으로 짝짓는다. 짝 만들기는 scripts/loco_local_pairs.py.
# 전제(~/files/kbds): setup_venv.sh · loco_local_pairs.py · fmt_strict.py(둘 다 scripts/ 에서 올려 둔다) · loco/${ROLLOUTS:-rollouts.jsonl}(kgpu/loco_rollouts.sh 산출) · opd_dv/dv_images.tar
# env(필수): HF_TOKEN. env(선택): TAG(출력 이름 끝에 붙임) ROLLOUTS(rollouts.jsonl) N_Q(20 문항) CUTS(3 지점) N_CONT(4 이어쓰기) GAP(200 단어) MAX_TOK(8192) MAXLEN(20480) MODEL_REPO(Qwen/Qwen3.5-9B)
# 실행: MAX_SEC=3600 setsid nohup bash kgpu/run_pod.sh loco_local.sh HF_TOKEN > logs/loco_local.out 2>&1 < /dev/null & disown
# 출력: ~/files/kbds/loco/localpairs${TAG}.jsonl · loco/pairs_inputs_local${TAG}.tar(loco_dpo.sh 의 INPUT_TAR 로 쓴다: pairs.jsonl + images/) · 로그 ~/files/kbds/logs/loco_local_*.log
#   판정: PAIRS_EXIT=0 · "쌍" 줄의 개수와 error 0 · 로그 끝의 "공유 prefix 끝 / 좋은 답 이어쓰기 앞 / 나쁜 답 이어쓰기 앞" 이 자연스럽게 이어지는지
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN 필요}"
F=/root/files/kbds; W=/root/w; VENV=/root/venv
mkdir -p "$W" "$F/logs" "$F/loco"; exec > >(tee "$W/local.log") 2>&1
trap 'cp "$W/local.log" "$F/logs/loco_local_$(date +%m%d-%H%M).log"' EXIT
T0=$(date +%s); stamp(){ echo "[+$(( $(date +%s)-T0 ))s] $*"; }
export HF_HOME=$W/hf UV_CACHE_DIR=$W/uv UV_HTTP_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
nvidia-smi -L

stamp "1) venv"
. "$F/setup_venv.sh"
stamp "2) 입력 — 롤아웃과 그 문항의 이미지"
cp "$F/loco/${ROLLOUTS:-rollouts.jsonl}" "$W/rollouts.jsonl" && cp "$F/loco_local_pairs.py" "$F/fmt_strict.py" "$W/" || { echo "❌ 입력 복사"; exit 1; }
IMG=$W/in/images; mkdir -p "$IMG"
$PY -c "import json;[print(json.loads(l)['image']) for l in open('$W/rollouts.jsonl')]" > "$W/need.txt"
tar xf "$F/opd_dv/dv_images.tar" -C "$IMG" -T "$W/need.txt" --no-same-owner || { echo "❌ dv_images.tar"; exit 1; }
ls "$IMG" | wc -l
stamp "3) 원본 모델 ← HF"
for TRY in 1 2 3; do
  HF_HUB_DISABLE_XET=$([ $TRY -gt 1 ] && echo 1 || echo 0) timeout 480 $PY -c "from huggingface_hub import snapshot_download as s;s('${MODEL_REPO:-Qwen/Qwen3.5-9B}',local_dir='$W/model')" && break
  echo "내려받기 ${TRY}번째 실패 — 다시 시도"; [ $TRY = 3 ] && { echo "❌ model"; exit 1; }
done
stamp "3.5) 채팅 템플릿 패치 — Qwen3.5 템플릿은 마지막 assistant 글을 '</think>' 로 닫는다. 닫지 않고 그대로 두어야 풀이 중간부터 이어 쓴다"
$PY - "$W/model/chat_template.jinja" "$W/tpl.jinja" <<'PY' || { echo "❌ 템플릿 패치"; exit 1; }
import sys
s = open(sys.argv[1]).read()
a = "        {%- if loop.index0 > ns.last_query_index %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n'"
assert s.count(a) == 1, '템플릿이 달라 패치할 자리를 못 찾았다'
open(sys.argv[2], 'w').write(s.replace(a, "        {%- if loop.last %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n" + a.replace('{%- if', '{%- elif', 1)))
print('패치 완료')
PY
stamp "4) vLLM serve"
"$VENV/bin/vllm" serve "$W/model" --served-model-name policy --max-model-len "${MAXLEN:-20480}" --gpu-memory-utilization 0.9 --chat-template "$W/tpl.jinja" \
  --enforce-eager --limit-mm-per-prompt '{"image": 1}' --host 127.0.0.1 --port 8000 > "$W/serve.log" 2>&1 &
for i in $(seq 1 90); do
  curl -s http://127.0.0.1:8000/v1/models | grep -q policy && { stamp READY; break; }
  [ "$i" = 90 ] && { echo "❌ serve"; tail -30 "$W/serve.log"; exit 1; }; sleep 10
done
stamp "5) 지점 단위 짝 N_Q=${N_Q:-20} CUTS=${CUTS:-3} N_CONT=${N_CONT:-4} GAP=${GAP:-200}"
(cd "$W" && $PY -I loco_local_pairs.py rollouts.jsonl "$IMG" "$W/out" --n-q "${N_Q:-20}" --cuts "${CUTS:-3}" --n "${N_CONT:-4}" --gap "${GAP:-200}" --max-tok "${MAX_TOK:-8192}")
echo "PAIRS_EXIT=$?"
if [ -s "$W/out/localpairs.jsonl" ]; then   # loco_dpo.sh 가 읽는 꼴(pairs.jsonl + images/)로 묶는다
  mkdir -p "$W/pk/images" && cp "$W/out/localpairs.jsonl" "$W/pk/pairs.jsonl" && while read -r f; do cp "$IMG/$f" "$W/pk/images/"; done < "$W/out/images.txt"
  tar cf "$F/loco/pairs_inputs_local${TAG:-}.tar" -C "$W/pk" pairs.jsonl images && cp "$W/out/localpairs.jsonl" "$F/loco/localpairs${TAG:-}.jsonl" && ls -la "$F/loco/pairs_inputs_local${TAG:-}.tar"
else echo "❌ 쌍이 없다 — 위 통계(valid·small_gap·error)를 본다"; fi
stamp "done"
