#!/bin/bash
# =============================================================================
# build_image.sh — ms-swift Singularity 환경 구축 (로그인 노드에서 실행, 인터넷 O)
#   glibc 2.17 때문에 conda/pip 로는 vLLM 설치 불가 → 컨테이너가 정식 경로.
#   ① 공식 이미지 pull(SIF) → ② sandbox(디렉토리) 변환 → ③ (검증은 계산노드에서)
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/../scripts/00_common.sh"
SIF_IMAGE="$WORK_DIR/images/ms-swift-413.sif"
mkdir -p "$(dirname "$SIF_IMAGE")"

# ModelScope 공식 ms-swift 이미지. ★ swift4.1.3 / cuda12.9.1 — 현재 활성 이미지(=ms-swift-413-sandbox 원본).
#   계산노드 드라이버 550.54.14(CUDA 12.4)에서 CUDA 마이너버전 호환으로 동작(검증 완료).
#   대안 폴백 이미지(swift3.8.3/cuda12.6.3, swift3.6.4/cuda12.4.0)는 디스크 정리로 삭제됨 — 필요 시 재pull.
#   us-west-1 이 막히면 cn-hangzhou / cn-beijing 으로 교체.
SRC_DOCKER="${SRC_DOCKER:-docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3}"

# 캐시는 작업영역에(홈 quota 무제한). pull 완료 후 수동 삭제 가능: rm -rf "$SINGULARITY_CACHEDIR"
export SINGULARITY_CACHEDIR="$WORK_DIR/.singularity_cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_CACHEDIR"

# ── ① SIF pull ──────────────────────────────────────────────────────────────
if [[ ! -f "$SIF_IMAGE" ]]; then
  echo "[pull] $SRC_DOCKER"
  singularity pull "$SIF_IMAGE" "$SRC_DOCKER"
else
  echo "[skip] SIF 이미 존재: $SIF_IMAGE"
fi

# ── ② sandbox 변환 ───────────────────────────────────────────────────────────
#   setuid=no + squashfuse 부재 환경에선 SIF 직접 실행 시 매번 추출(느림).
#   sandbox(디렉토리)로 풀어두면 추출 없이 빠르게 exec.  (= $CONTAINER_IMG)
if [[ ! -d "$CONTAINER_IMG" ]]; then
  echo "[sandbox] $SIF_IMAGE -> $CONTAINER_IMG"
  singularity build --sandbox "$CONTAINER_IMG" "$SIF_IMAGE"
else
  echo "[skip] sandbox 이미 존재: $CONTAINER_IMG"
fi

# ── ③ 검증 안내 ──────────────────────────────────────────────────────────────
#   GPU/vLLM 검증은 반드시 계산노드에서(로그인노드 드라이버 470=CUDA11.4 라 vLLM import 실패).
echo "[done] 빌드 완료. 계산노드 검증 예:"
echo "  srun -p debug-1gpu --nodes=1 --tasks-per-node=1 --time=00:05:00 --comment=etc \\"
echo "    singularity exec --nv $CONTAINER_IMG python -c \\"
echo "    \"import torch,swift,vllm; print(swift.__version__, torch.__version__, vllm.__version__, torch.cuda.is_available())\""
