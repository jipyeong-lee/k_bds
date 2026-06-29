#!/bin/bash
# =============================================================================
# judge_server.sh — Stage-3 RaR judge 서버 (Qwen3.6-27B-FP8 멀티모달, vLLM OpenAI)
#   로그인 노드 A100 40GB 에 상주 → 컴퓨트 노드(학습)가 내부 IP:PORT 로 호출.
#   같은 qwen3_5 아키텍처라 컨테이너 vLLM 0.19.1 로 그대로 서빙(검증된 스택).
#
# 40GB 보수 설정: max-model-len 8K(native 262K 축소) + enforce-eager(메모리↓) + util 0.92.
# 사용:  bash scripts/judge_server.sh            # 포그라운드
#        JUDGE_PORT=8100 nohup bash scripts/judge_server.sh >logs/judge_server.log 2>&1 &
# 클라이언트: JUDGE_BASE_URL=http://<이 노드 IP>:8100/v1  JUDGE_MODEL=qwen36-judge
# =============================================================================
set -eu
cd /home01/k252a01/kbds_project
SB="${CONTAINER_IMG:-work/images/ms-swift-413-sandbox}"
MODEL=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "모델 미다운로드"; exit 1; }
PORT="${JUDGE_PORT:-8100}"
MAXLEN="${JUDGE_MAX_LEN:-8192}"
UTIL="${JUDGE_GPU_UTIL:-0.92}"
TP="${JUDGE_TP:-1}"           # tensor parallel (1gpu=1, 2gpu=2)

echo "[judge] MODEL=$MODEL PORT=$PORT MAXLEN=$MAXLEN UTIL=$UTIL TP=$TP"
exec singularity exec --nv \
  --env HF_HUB_OFFLINE=1 \
  "$SB" vllm serve "$MODEL" \
    --served-model-name qwen36-judge \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --tensor-parallel-size "$TP" \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 1}' \
    --host 0.0.0.0 --port "$PORT"
