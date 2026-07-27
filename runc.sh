#!/bin/bash
# =============================================================================
# runc.sh — apptainer 없이 컨테이너 sandbox 런타임을 직접 구동하는 래퍼 (우회책)
# -----------------------------------------------------------------------------
#  배경: 클러스터 공용 apptainer 1.4.5 가 파손(libsubid.so.3 부재 + GLIBC_2.28 요구,
#        호스트는 CentOS7/glibc 2.17) → `singularity exec` 불가. 이미지 자체는 정상.
#  원리: sandbox 안에 Ubuntu22.04 의 완전한 glibc 2.35 런타임이 들어있다.
#        그 로더(ld-linux)로 sandbox 안 python 을 직접 실행하면 호스트 glibc 를
#        우회하므로 컨테이너 없이도 동일 스택(torch/vllm/swift)이 그대로 돈다.
#        네임스페이스 격리는 없지만 학습엔 파일시스템 가시성만 필요하고,
#        홈은 어차피 bind 대상이었으므로 실질 차이 없음.
#  사용:  ./runc.sh -c "import torch; print(torch.__version__)"
#         ./runc.sh -m torch.distributed.run --nproc_per_node 8 ...   (run_py 대체)
#  ⚠️ apptainer 복구되면 00_common.sh 의 ENV_MODE=container 정식 경로로 복귀할 것.
# =============================================================================
# 계정 이식성: 스크립트 위치에서 PROJ_DIR 을 유도(하드코딩 금지).
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJ_DIR="${PROJ_DIR:-$_HERE}"
S="${CONTAINER_IMG:-$PROJ_DIR/work/images/ms-swift-413-sandbox}"

# sandbox 안 pip nvidia-* 라이브러리 전부 수집(cudart/cublas/cudnn/nccl 등)
NV_LIBS="$(ls -d "$S"/usr/local/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"

LP="$S/usr/local/lib"
LP="$LP:$S/lib/x86_64-linux-gnu:$S/usr/lib/x86_64-linux-gnu"
LP="$LP:$S/usr/local/cuda/lib64:$S/usr/local/cuda/targets/x86_64-linux/lib"
LP="$LP:$S/usr/local/cuda-12.9/targets/x86_64-linux/lib"
LP="$LP:$S/usr/local/lib/python3.12/site-packages/torch/lib"
LP="$LP:${NV_LIBS}"
# libcuda.so(드라이버)는 컨테이너에 없음 → 호스트 것을 사용해야 GPU 가 잡힌다
LP="$LP:/usr/lib64:/lib64"

export PYTHONPATH="$S/usr/local/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
# ⚠️ $S/usr/local/cuda 는 /etc/alternatives/cuda 를 가리키는 심볼릭 링크라
#    컨테이너 밖(chroot 없음)에서는 깨진다 → deepspeed 가 nvcc 를 못 찾아 죽음.
#    실경로(cuda-12.9)를 직접 지정한다.
_CUDA_REAL="$(ls -d "$S"/usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1)"
export CUDA_HOME="${CUDA_HOME:-${_CUDA_REAL:-$S/usr/local/cuda}}"
export PATH="$CUDA_HOME/bin:$PATH"

# Triton 이 런타임에 C 커널(cuda_utils.c)을 컴파일하는데, 호스트 기본 gcc 4.8.5 는
# C89 라 "'for' loop initial declarations are only allowed in C99 mode" 로 실패한다.
# (컨테이너 안 gcc-11 은 glibc 2.32 요구라 로더 없이 못 뜸) → 모듈 gcc 10.2.0 사용.
if [[ -x /apps/compiler/gcc/10.2.0/bin/gcc ]]; then
  export CC="${CC:-/apps/compiler/gcc/10.2.0/bin/gcc}"
  export CXX="${CXX:-/apps/compiler/gcc/10.2.0/bin/g++}"
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib64}"
  export PATH="/apps/compiler/gcc/10.2.0/bin:$PATH"
  export LD_LIBRARY_PATH="/apps/compiler/gcc/10.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# torchrun/DDP 가 자식 프로세스를 sys.executable 로 띄울 때도 로더를 경유하도록,
# sys.executable 을 shim(bin/python) 으로 고정한다. 이게 없으면 자식이 호스트
# glibc 2.17 로 직접 뜨면서 GLIBC_2.34 not found 로 죽는다.
export PYTHONEXECUTABLE="${PYTHONEXECUTABLE:-$PROJ_DIR/bin/python}"
# PYTHONEXECUTABLE 를 바꾸면 stdlib 탐색 기준(prefix)이 호스트 /usr/local 로 흘러
# 'No module named encodings' 로 죽는다 → PYTHONHOME 으로 sandbox prefix 를 명시.
export PYTHONHOME="${PYTHONHOME:-$S/usr/local}"
# transformer_engine 등이 dlopen 으로 찾는 경로 → LD_LIBRARY_PATH 에도 노출.
#  ⚠️ 단, 컨테이너 glibc 디렉터리($S/lib/x86_64-linux-gnu 등)는 절대 넣지 않는다.
#     LD_LIBRARY_PATH 는 자식 프로세스에 상속되는데, swift 가 띄우는 호스트 /bin/bash 가
#     컨테이너 libc.so.6 를 host ld-linux 로 로드하면 즉사한다:
#       "relocation error: ... symbol _dl_audit_preinit, version GLIBC_PRIVATE not defined"
#     glibc 해석은 아래 exec 의 --library-path(=$LP)가 python 프로세스에만 적용한다.
export LD_LIBRARY_PATH="$S/usr/local/cuda/lib64:$S/usr/local/cuda/targets/x86_64-linux/lib:$S/usr/local/cuda-12.9/targets/x86_64-linux/lib:$S/usr/local/lib/python3.12/site-packages/torch/lib:${NV_LIBS}/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$S/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" --library-path "$LP" \
     "$S/usr/local/bin/python3" "$@"
