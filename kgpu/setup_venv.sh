# setup_venv.sh — kgpu pod 공통: curl·uv 설치 후 /root/venv 에 검증 핀 스택(CLAUDE.md §1.4). source 해서 쓴다.
# 전제: $W(pod 로컬 작업 dir), $VENV, stamp() 가 정의돼 있을 것. 끝나면 $PY 를 쓸 수 있다.
# pro6000 노드엔 기본 이미지(kgpu-pytorch)가 없어 nvidia/cuda 이미지로 빌린다 → curl·uv 부터 깐다
# 컨테이너 root 에 setgroups·DAC override 가 없다 → _apt 강등 끄고, _apt 소유 캐시 대신 $W/apt 사용
mkdir -p "$W/apt/partial"
APT="apt-get -o APT::Sandbox::User=root -o Dir::Cache::Archives=$W/apt"
command -v curl >/dev/null || { $APT update -qq && $APT install -y -qq curl ca-certificates >/dev/null; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=$HOME/.local/bin:$PATH; }
if [ ! -x "$VENV/bin/swift" ]; then
  uv venv "$VENV" --python 3.12 2>&1 | tail -1
  uv pip install --python "$VENV/bin/python" torch==2.10.0+cu129 torchvision==0.25.0+cu129 \
    vllm==0.19.1 ms-swift==4.1.3 transformers==5.6.2 trl==0.29.1 peft==0.19.1 \
    accelerate==1.13.0 datasets==3.6.0 pyarrow==23.0.1 pillow openai huggingface_hub \
    "qwen_vl_utils==0.0.14" "decord==0.6.0" "av==17.0.1" \
    "math_verify==0.9.0" "latex2sympy2_extended==1.11.0" "word2number==1.1" \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match 2>&1 | tail -4
  uv pip install --python "$VENV/bin/python" --prerelease=allow "nvidia-cutlass-dsl==4.5.0.dev0" 2>&1 | tail -2
fi
PY=$VENV/bin/python
# fastapi 0.159 는 vLLM 의 prometheus_fastapi_instrumentator 7.1.0 과 안 맞아 모든 요청이 500(_IncludedRouter.path) → KISTI 핀으로
$PY -c "import fastapi,sys;sys.exit(fastapi.__version__!='0.136.1')" || uv pip install --python "$PY" fastapi==0.136.1 2>&1 | tail -1
# flashinfer 가 Blackwell 커널을 JIT 할 때 nvrtc.h 를 venv 휠에서 찾게 한다(B200 함정 ④와 같은 대응)
NVRTC=$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc
export CPATH="$NVRTC/include${CPATH:+:$CPATH}" LIBRARY_PATH="$NVRTC/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
       LD_LIBRARY_PATH="$NVRTC/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
$PY -c "import torch;print('  torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))"

