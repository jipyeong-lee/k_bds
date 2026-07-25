#!/bin/bash
# =============================================================================
# 45_healthbench_smoke.slurm — base(Qwen3.5-9B) HealthBench 스모크 (Hard, limit 20)
#   GPU0 = judge(Qwen3.6-27B-FP8) vLLM · GPU1 = base(Qwen3.5-9B) vLLM · evalscope 채점.
#   채점자=자체 judge(오프라인, GPT-4.1 아님) → 내부 상대비교용 절대점수.
#   데이터는 로그인노드서 미리 받아둔 work/ms_cache 사용(계산노드 오프라인).
# =============================================================================
#SBATCH --job-name=hb-smoke
#SBATCH --partition=4gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/healthbench_smoke_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/healthbench_smoke_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
SB="${CONTAINER_IMG:-work/images/ms-swift-413-sandbox}"

BASE=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
JUDGE=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ 2>/dev/null | head -1)
# 평가대상 모델: 기본 base, HB_TARGET_MODEL 로 오버라이드(콜드스타트/학습모델 등). TID=served name.
TARGET="${HB_TARGET_MODEL:-$BASE}"
TID="${HB_MODEL_ID:-base9b}"
[ -z "$TARGET" ] && { echo "❌ 평가대상 모델 없음"; exit 1; }
[ -z "$JUDGE" ]  && { echo "❌ judge 모델 없음"; exit 1; }
MSCACHE=/home01/k252a02/kbds_project/work/ms_cache
PORT_BASE=8710; PORT_JUDGE=8720

serve() {  # $1=gpu $2=model $3=name $4=port $5=maxlen $6=extra
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$1" \
  singularity exec --nv --env HF_HUB_OFFLINE=1 --env VLLM_USE_MODELSCOPE=False "$SB" \
    vllm serve "$2" --served-model-name "$3" --max-model-len "$5" \
      --gpu-memory-utilization 0.90 --enforce-eager $6 \
      --host 0.0.0.0 --port "$4" > "logs/hb_serve_${3}_${SLURM_JOB_ID}.log" 2>&1 &
  echo $!
}

wait_ready() {  # $1=port $2=name
  for i in $(seq 1 90); do
    curl -s "http://localhost:$1/v1/models" 2>/dev/null | grep -q "$2" && { echo "[$2] READY (~$((i*10))s)"; return 0; }
    sleep 10
  done
  echo "[$2] ❌ not ready"; return 1
}

echo "[hb] target($TID)=$TARGET"; echo "[hb] judge=$JUDGE"
JPID=$(serve 0 "$JUDGE" qwen36-judge $PORT_JUDGE 8192 "")
BPID=$(serve 1 "$TARGET" "$TID"      $PORT_BASE 10240 "--limit-mm-per-prompt {\"image\":1}")

wait_ready $PORT_JUDGE qwen36-judge || { tail -20 logs/hb_serve_qwen36-judge_${SLURM_JOB_ID}.log; kill $JPID $BPID 2>/dev/null; exit 1; }
wait_ready $PORT_BASE  "$TID"        || { tail -20 logs/hb_serve_${TID}_${SLURM_JOB_ID}.log; kill $JPID $BPID 2>/dev/null; exit 1; }

echo "[hb] ===== evalscope HealthBench (Hard, limit ${HB_LIMIT:-20}) ====="
singularity exec --nv \
  --env HF_HUB_OFFLINE=1 --env MODELSCOPE_CACHE=$MSCACHE --env MODELSCOPE_OFFLINE=1 \
  --env HB_MODEL_ID="$TID" --env HB_API_URL=http://localhost:$PORT_BASE/v1/ \
  --env HB_JUDGE_ID=qwen36-judge --env HB_JUDGE_URL=http://localhost:$PORT_JUDGE/v1/ \
  --env HB_LOCAL_PATH=$MSCACHE/datasets/openai-mirror/healthbench \
  --env HB_CONC="${HB_CONC:-8}" --env HB_JUDGE_CONC="${HB_JUDGE_CONC:-8}" \
  --env HB_VERSION="${HB_VERSION:-Hard}" --env HB_LIMIT="${HB_LIMIT:-20}" \
  --env HB_MAXTOK="${HB_MAXTOK:-2048}" --env HB_WORKDIR=work/eval_healthbench \
  "$SB" python scripts/run_healthbench.py
RC=$?

echo "[hb] evalscope rc=$RC — 서버 정리"
kill $JPID $BPID 2>/dev/null; sleep 3; kill -9 $JPID $BPID 2>/dev/null || true
echo "[hb] 결과: work/eval_healthbench (reports/*/*.json)"
exit $RC
