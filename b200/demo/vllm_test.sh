#!/bin/bash
# vllm_test.sh — 노드에서 vLLM 자체 호스팅 인퍼런스가 되는지 검증한다.
#
# 두 가지를 따로 본다:
#   Ⓐ 오프라인 배치 추론  vllm.LLM(...).generate(...)      — 서버 없이 프로세스 안에서
#   Ⓑ OpenAI 호환 서버     vllm serve + curl localhost:8000 — 같은 job 안에서 호출
#
# ⚠️ 외부에서 이 서버를 직접 부를 수는 없다. 플랫폼 API 에 포트 노출·ingress 기능이 없어서
#    노드의 8000 포트는 job 안에서만 보인다. 외부 서비스로 쓰려면 job 안에서 배치로 돌려
#    결과를 $ORCH_HOME 에 쓰고 파일 API 로 꺼내는 형태가 된다.
set -uo pipefail

export HOME="$ORCH_HOME"
export HF_HOME="$ORCH_HOME/.hf"
export TMPDIR="$ORCH_HOME/.tmp"
export TRITON_CACHE_DIR="$ORCH_HOME/.triton"
export TORCHINDUCTOR_CACHE_DIR="$ORCH_HOME/.inductor"
export VLLM_CACHE_ROOT="$ORCH_HOME/.vllm"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$VLLM_CACHE_ROOT"

W="$ORCH_HOME/vllm_test"; mkdir -p "$W"; cd "$W"
VENV="$ORCH_HOME/.venv-vllm"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
# tensor_parallel_size 는 **어텐션 헤드 수를 나눠야** 한다. Qwen2.5-0.5B 는 14 헤드라
# 1·2·7·14 만 된다(8 은 안 된다). 8장을 다 쓰려면 헤드가 8의 배수인 모델이어야 한다.
TP="${VLLM_TP:-2}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[vllm] venv 생성 (최초 1회, 5~10분)"
  uv venv "$VENV" --python 3.12
  # CLAUDE.md §1.4 의 검증 핀. cutlass-dsl 은 최신(4.7.0)이 cute.core.ThrMma 를 없애
  # vLLM ViT 의 quack 이 깨지므로 4.5.0.dev0 으로 내려야 한다.
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    torch==2.10.0 vllm==0.19.1 transformers==5.6.2 || { echo "[vllm] 설치 실패"; exit 1; }
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" --prerelease=allow \
    nvidia-cutlass-dsl==4.5.0.dev0 || echo "[vllm] cutlass-dsl 설치 실패(계속 진행)"
else
  echo "[vllm] 기존 venv 재사용"
fi
PY="$VENV/bin/python"

# flashinfer 가 Blackwell 커널을 JIT 컴파일할 때 nvrtc.h 를 찾는다. 시스템 CUDA 에는 없고
# venv 의 nvidia/cuda_nvrtc 에 있으므로 노출해 준다(CLAUDE.md §1.4 함정④).
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export CPATH="$SP/nvidia/cuda_nvrtc/include:${CPATH:-}"
export LIBRARY_PATH="$SP/nvidia/cuda_nvrtc/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

echo "[vllm] version: $("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
echo "[vllm] model=$MODEL  tp=$TP  gpu=$(nvidia-smi -L | wc -l)장"

echo "══ Ⓐ 오프라인 배치 추론 ══"
#  ⚠️ 이 코드를 heredoc(stdin)으로 넘기면 안 된다. vLLM 엔진이 워커 서브프로세스를 띄우며
#     메인 모듈을 다시 import 하는데, stdin 은 다시 열 수 없어 FileNotFoundError('<stdin>') 로 죽는다.
#     반드시 **파일로 써서** 실행할 것.
cat > "$W/offline.py" <<'PYCODE'
#  ⚠️ **반드시 __main__ 가드 안에 둘 것.** tp>1 이면 vLLM 이 spawn 으로 워커를 띄우는데,
#     자식이 이 파일을 다시 실행한다(multiprocessing 의 _fixup_main_from_path).
#     LLM(...) 이 모듈 최상단에 있으면 자식이 또 엔진을 만들려다 죽는다.
import sys, time
from vllm import LLM, SamplingParams


def main():
    model, tp = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=0.60,
              max_model_len=2048, enforce_eager=True)
    print(f"[A] 모델 로드 {time.time()-t0:.1f}s", flush=True)
    prompts = ["대한민국의 수도는?", "LoRA 가 전체 미세조정보다 가벼운 이유를 한 문장으로.",
               "3 곱하기 7 은?"]
    chats = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outs = llm.chat(chats, SamplingParams(temperature=0.0, max_tokens=64))
    el = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    for p, o in zip(prompts, outs):
        print(f"[A] Q: {p}\n[A] A: {o.outputs[0].text.strip()[:160]}", flush=True)
    print(f"[A] {len(outs)}건 · {ntok} 토큰 · {el:.2f}s · {ntok/el:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
PYCODE
"$PY" "$W/offline.py" "$MODEL" "$TP"
echo "[A] EXIT=$?"

echo "══ Ⓑ OpenAI 호환 서버 ══"
"$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.60 \
  --max-model-len 2048 --enforce-eager --port 8000 > "$W/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 90); do
  curl -s --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "[B] 서버 사망 — server.log 참고"; tail -15 "$W/server.log"; break; }
  sleep 5
done
if curl -s --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  # 앞 job 이 killed 되면 8000 이 TIME_WAIT 로 남고 vLLM 은 조용히 8001 로 뜬다.
  # 실서비스에서는 포트를 비우려 하지 말고 로그의 "Uvicorn running on ...:<포트>" 를 읽을 것.
  echo "[B] /health OK"
  curl -s --max-time 60 http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"빛의 속도는?\"}],\"max_tokens\":64,\"temperature\":0}" \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("[B] 응답:", d["choices"][0]["message"]["content"].strip()[:200]); print("[B] 토큰:", d["usage"])'
  curl -s --max-time 10 http://127.0.0.1:8000/v1/models | "$PY" -c 'import json,sys; print("[B] /v1/models:", [m["id"] for m in json.load(sys.stdin)["data"]])'
else
  echo "[B] 서버가 뜨지 않았다"
fi
kill $SRV 2>/dev/null || true
echo "[vllm] DONE"
