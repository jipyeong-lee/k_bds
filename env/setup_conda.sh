#!/bin/bash
# =============================================================================
# setup_conda.sh — ms-swift 전용 conda 환경 구성 (로그인 노드에서 1회 실행)
#   목적: SFT / RLVR / GRPO(LLM-as-judge) 전 단계를 한 env 에서 수행
#   참고: 로그인 노드는 인터넷 가능. 컴퓨트 노드는 보통 불가하므로 여기서 설치.
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-swift}"
source /apps/application/miniconda3-new/etc/profile.d/conda.sh

# ── Environment Module (가이드 §4) ──────────────────────────────────────────
#  시스템 gcc 4.8.5 로는 cython/C++17 소스 빌드 불가 → gcc 10.2.0 모듈 필수.
#  cuda 모듈은 flash-attn/deepspeed 등 CUDA 확장 컴파일(nvcc)용.
module purge
module load compilers/cuda/12.4 compilers/gcc/10.2.0
export CC=gcc CXX=g++               # module 의 gcc 10.2.0 을 빌드에 사용
echo "[env] gcc=$(gcc -dumpversion)  cuda=$(nvcc --version | grep -oE 'release [0-9.]+' | head -1)"

if ! conda env list | grep -q "/${ENV_NAME}\b"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# --- 코어 스택 (torch는 cu124 휠, manylinux_2_17 → glibc 2.17 OK) -----------
pip install --upgrade pip
pip install "torch==2.6.0" "torchvision" --index-url https://download.pytorch.org/whl/cu124

# --- glibc 2.17 호환 constraints (소스 빌드 회피) ----------------------------
#   최신 pandas/pyarrow/libcst 는 manylinux_2_28 휠만 제공 → glibc 2.17 에선 소스
#   빌드로 빠져 실패(gcc/cmake/Rust 요구). constraints.txt 로 2_17 휠 버전에 캡.
#   pyarrow 캡이 datasets 를 호환 버전으로 자동 다운그레이드시킴.
CONSTRAINTS="$(dirname "$0")/constraints.txt"

# --- ms-swift + 멀티모달/RL 의존성 (모든 설치에 constraints 적용) -----------
pip install -c "$CONSTRAINTS" "ms-swift[llm]" -U
pip install -c "$CONSTRAINTS" "vllm==0.8.5"   # GRPO rollout / LLM-as-judge (manylinux1 휠, OK)
pip install -c "$CONSTRAINTS" deepspeed       # ZeRO-3 (실행노드도 gcc/cuda 모듈 필요: JIT)
pip install -c "$CONSTRAINTS" "transformers>=4.49" accelerate peft trl datasets qwen-vl-utils
# flash-attn: 소스 빌드가 무겁고(20~40분, 대용량 RAM) 로그인노드 부적합.
#   기본은 sdpa 로 동작(스크립트 ATTN_IMPL). 필요 시 계산노드 salloc 에서 별도 빌드 권장:
#     pip install flash-attn --no-build-isolation
echo "[note] flash-attn 미설치(기본 sdpa). 필요 시 계산노드에서 별도 빌드."

# --- 검증 -------------------------------------------------------------------
python - <<'PY'
import torch, swift, vllm, deepspeed, transformers
print("torch     :", torch.__version__, "cuda", torch.version.cuda)
print("ms-swift  :", swift.__version__)
print("vllm      :", vllm.__version__)
print("deepspeed :", deepspeed.__version__)
print("transformers:", transformers.__version__)
PY

echo "[done] conda env '$ENV_NAME' 준비 완료.  scripts/00_common.sh 의 CONDA_ENV 와 일치시킬 것."
