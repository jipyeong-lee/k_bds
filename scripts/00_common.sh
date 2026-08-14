#!/bin/bash
# =============================================================================
# 00_common.sh — 모든 단계 Slurm 스크립트가 source 하는 공통 설정
#   각 *.slurm 스크립트 상단에서:  source "$(dirname "$0")/00_common.sh"
# =============================================================================

# ---- 프로젝트 경로 ----------------------------------------------------------
# 다른 계정 이식: PROJ_DIR 만 바꾸면 파생경로(WORK_DIR 등) 자동 추종. 단 *.slurm 의
#  #SBATCH --output 은 SBATCH 지시어라 env 치환 불가 → 인수인계(HANDOFF.md)의 sed 로 일괄교체.
export PROJ_DIR="${PROJ_DIR:-/home01/k266a01/kbds_project}"
# 가이드: 모든 분석/저장은 작업(홈) 디렉토리에서 수행. 홈 quota는 사실상 무제한(Lustre).
# (대용량 임시 IO를 scratch에 두려면 WORK_DIR=/scratch/$USER/kbds 로 override)
export WORK_DIR="${WORK_DIR:-/home01/k266a01/kbds_project/work}"  # 산출물 루트
export DATA_DIR="${DATA_DIR:-$WORK_DIR/data}"            # DeepVision-103K, medix-rl-data
export CKPT_DIR="${CKPT_DIR:-$WORK_DIR/checkpoints}"     # 단계별 가중치
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"          # HF/모델 캐시 (gemma-4-12B-it 사전 다운로드됨)
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$WORK_DIR/ms_cache}"
export USE_HF="${USE_HF:-1}"                             # ms-swift: HF 허브/캐시 사용(모델을 HF서 받음, 기본 ModelScope 아님)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"             # 컴퓨트노드 오프라인 → 캐시만 사용(산출물 사전 다운로드 가정)
mkdir -p "$WORK_DIR" "$DATA_DIR" "$CKPT_DIR" "$HF_HOME" "$MODELSCOPE_CACHE" "$PROJ_DIR/logs"

# ---- 베이스 모델 ------------------------------------------------------------
# Qwen3.5-9B (멀티모달, MLLMModelType.qwen3_5). 컨테이너 풀스택 지원 확인:
#   swift4.1.3 등록 / transformers5.6.2 qwen3_5 / vllm0.19.1 Qwen3_5ForConditionalGeneration.
#   (Gemma4 12B 는 gemma4_unified=transformers5.10.dev 필요 → CUDA13 툴체인이라 이 드라이버서 불가)
#   9B 라 8×A100-80GB GRPO 여유. 게이트 없음. -Base 대신 instruct 기본.
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
export MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"

# 출력 형식 시스템 프롬프트 (전 단계 통일: <think>추론</think><answer>답</answer>)
#  - 내장 format 보상이 이 구조를 검사, accuracy(math_verify) 가 <answer> 내 평문답을 파싱.
#  - \boxed 미사용(컨벤션 결정): math_verify 가 평문 숫자답 정상 검증. "concise" 로 장황/truncation 억제.
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a multimodal reasoning assistant. Carefully examine the image(s) and reason step by step INSIDE <think> </think>, keeping the reasoning concise. Then give ONLY the final answer INSIDE <answer> </answer>. For multiple-choice, put only the letter, e.g. <answer>A</answer>.}"

# ---- 환경 방식 선택: container(권장) | conda -------------------------------
#  glibc 2.17 때문에 vLLM 은 conda/pip 설치 불가 → 컨테이너가 기본. (자세히: 메모리 env-glibc-container)
# 2026-07-27: 클러스터 apptainer 파손(libsubid/GLIBC_2.28)으로 container 모드 불가 →
#   기본값을 loader 로 전환. apptainer 복구되면 container 로 되돌릴 것.
export ENV_MODE="${ENV_MODE:-loader}"
export CONDA_ENV="${CONDA_ENV:-swift}"                   # conda 폴백용(SFT 한정, vLLM 불가)
# 실행 이미지: SIF 는 squashfuse/ setuid 제약으로 매 실행 추출이 느려 sandbox(디렉토리) 사용.
#  기본 = swift4.1.3/cuda12.9.1(Gemma4 지원, 계산노드 검증: vllm._C+커널 OK). 마이너버전 호환 동작.
#  재빌드 원본: ms-swift-413.sif (env/build_image.sh). 구 폴백 이미지(383/3.6.4)는 디스크 정리로 삭제됨.
export CONTAINER_IMG="${CONTAINER_IMG:-$WORK_DIR/images/ms-swift-413-sandbox}"

# 단계별 실행 명령을 감싸는 래퍼. 사용:  run_py swift sft ...
#  singularity 는 기본적으로 호스트 env 를 전달하므로 NPROC_PER_NODE/MASTER_*/NCCL_* 등
#  앞서 export 한 분산 변수들이 컨테이너 안에서도 보임(--cleanenv 미사용).
#  ENV_MODE=loader : apptainer 파손 대비 우회 경로(2026-07-27~).
#    클러스터 공용 apptainer 1.4.5 가 libsubid.so.3 부재 + GLIBC_2.28 요구(호스트 2.17)로
#    실행 불가. sandbox 안 glibc 2.35 로더로 sandbox python 을 직접 구동해 우회한다.
#    GPU 노드 검증: torch 2.10.0+cu129 cuda_avail True · vllm 0.19.1 (job 72830).
#    apptainer 복구되면 ENV_MODE=container 로 되돌릴 것. 상세 = runc.sh 주석.
run_py() {
  if [[ "$ENV_MODE" == "container" ]]; then
    singularity exec --nv \
      --bind "$WORK_DIR:$WORK_DIR" --bind "$PROJ_DIR:$PROJ_DIR" \
      --env HF_HOME="$HF_HOME",MODELSCOPE_CACHE="$MODELSCOPE_CACHE",USE_HF="$USE_HF",HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
      "$CONTAINER_IMG" "$@"
  elif [[ "$ENV_MODE" == "loader" ]]; then
    # "swift ..." / "python ..." 형태 호출을 sandbox python 진입점으로 변환
    local first="$1"; shift
    case "$first" in
      swift)          "$PROJ_DIR/runc.sh" -m swift.cli.main "$@" ;;
      python|python3) "$PROJ_DIR/runc.sh" "$@" ;;
      *)              "$PROJ_DIR/runc.sh" -m "$first" "$@" ;;
    esac
  else
    conda run -n "$CONDA_ENV" --no-capture-output "$@"
  fi
}

# vLLM OpenAI 호환 서버 기동 래퍼.  사용:  run_serve "$MODEL" --port 8165 ... &
#  🚨 평가 스크립트는 반드시 이것을 쓸 것. `singularity exec ... vllm serve` 를 직접 부르면
#     ENV_MODE 분기를 우회해 apptainer 파손 환경(2026-07-27~)에서 100% 실패한다.
#     (2026-08-02 발견: 평가 스크립트 9종이 전부 07-25자 = 파손 이전 작성이라 이 경로로 깨져 있었다.)
#  진입점은 `vllm` 콘솔 스크립트와 동일한 vllm.entrypoints.cli.main:main → `vllm serve <model>` 과
#  인자 처리가 같다. run_py 를 경유하므로 container/loader/conda 분기를 그대로 따른다.
run_serve() {
  local model="$1"; shift
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
  run_py python -m vllm.entrypoints.cli.main serve "$model" "$@"
}

# ---- 분산 학습 공통 (단일 노드 8GPU 기준) ----------------------------------
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | head -n1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# NCCL — IB 환경. 문제 시 NCCL_DEBUG=INFO 로 디버그
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# ms-swift 멀티모달 최대 픽셀 (VRAM 절약 핵심 노브)
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
# attention 구현: 컨테이너에 flash_attn 2.8.3 내장 → flash_attn(forward 가속 + 메모리 절감).
#   문제 시 sdpa 로 폴백(ATTN_IMPL=sdpa). vLLM 생성은 자체 커널이라 무관, HF forward 에만 효과.
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

echo "[common] ENV_MODE=$ENV_MODE  BASE_MODEL=$BASE_MODEL  WORK_DIR=$WORK_DIR"
echo "[common] node=$(hostname)  GPUs=$(nvidia-smi -L 2>/dev/null | wc -l)"
