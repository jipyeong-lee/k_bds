#!/bin/bash
# payload.sh — 노드 안에서 실행되는 부분. venv 를 준비하고 torchrun 으로 8장을 쓴다.
#
# 이 스크립트가 하는 일의 절반은 **B200 노드의 함정을 피하는 것**이다(CLAUDE.md §1.4).
# 아무 설정 없이 torchrun 을 돌리면 아래 네 곳에서 막힌다:
#   ⓞ **업로드된 디렉터리는 job 이 쓸 수 없다** → 코드는 demo/ 에서 읽고, 출력은 job 이 만든
#      demo_run/ 에 쓴다. API 서비스가 만든 파일·디렉터리의 소유자가 job 사용자와 다르다
#      (같은 이유로 uploads/ 는 세션에서 rm 도 안 된다 — CLAUDE.md §1.5)
#   ① HOME=/root 가 쓰기 불가       → HF·pip 캐시가 못 만들어진다
#   ② /tmp 가 tmpfs + noexec        → Triton 이 컴파일한 .so 를 map 못 해 죽는다
#   ③ 세션 venv 는 세션과 함께 사라짐 → $ORCH_HOME 에 만들어야 재실행이 빠르다
#   ④ flash-attn 없음               → attn_implementation='sdpa' (lora_demo.py 에서 처리)
set -euo pipefail

export HOME="$ORCH_HOME"                       # ①
export HF_HOME="$ORCH_HOME/.hf"
export TMPDIR="$ORCH_HOME/.tmp"                # ②
export TRITON_CACHE_DIR="$ORCH_HOME/.triton"
export TORCHINDUCTOR_CACHE_DIR="$ORCH_HOME/.inductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

VENV="$ORCH_HOME/.venv-demo"                   # ③
if [ ! -x "$VENV/bin/python" ]; then
  echo "[payload] venv 생성 — 최초 1회만 (약 2~3분)"
  uv venv "$VENV" --python 3.12
  # torch 는 Blackwell(B200)용 cu129 휠이어야 한다. 일반 pypi 휠은 이 GPU 를 못 쓴다.
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    'transformers>=5.0' 'peft>=0.19' accelerate
else
  echo "[payload] 기존 venv 재사용 → $VENV"
fi

WORK="$ORCH_HOME/demo_run"        # ⓞ job 이 만든 디렉터리 — 여기만 쓸 수 있다
mkdir -p "$WORK"; cd "$WORK"
export DEMO_OUT="$WORK/lora_demo_out"
echo "[payload] GPU $(nvidia-smi -L | wc -l) 장 · torchrun 시작"
"$VENV/bin/python" -m torch.distributed.run --nproc_per_node=8 --standalone \
  "$ORCH_HOME/demo/lora_demo.py"
echo "[payload] EXIT=$?"
