#!/bin/bash
# =============================================================================
# 60_probe_fragility.slurm — 형식 취약성 temperature 스윕
#
#   §8-i ⑧ 의 미해결 질문: "절벽까지의 거리"가 붕괴 이전에 측정 가능한 양인가?
#   로그로 잴 수 있는 양은 step 800 까지 전부 정상이었다(12개 가설 기각). 남은 수단은
#   체크포인트를 직접 흔들어 보는 것 — temperature 를 올려 깨진 형식 basin 근처의
#   확률질량을 드러낸다.
#
#   사용:  sbatch --export=ALL,STEPS="700 850 400" scripts/60_probe_fragility.slurm
#   knobs: PROBE_TEMPS / PROBE_NS / PROBE_MAXTOK / PROBE_CONC / EVAL_PORT
#
#   ⚠️ ENV_MODE=loader (apptainer 파손, 00_common.sh 참고) → 서버 기동은 반드시 run_serve.
#      `singularity exec ... vllm serve` 직접 호출은 100% 실패한다.
# =============================================================================
#SBATCH --job-name=frag
#SBATCH --partition=debug-1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:30:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k252a02/kbds_project/logs/probe_frag_%j.log
#SBATCH --error=/home01/k252a02/kbds_project/logs/probe_frag_%j.log

set -u
cd /home01/k252a02/kbds_project
source scripts/00_common.sh
export NPROC_PER_NODE=1

STEPS="${STEPS:-700 850}"
PORT="${EVAL_PORT:-8231}"
RESULT="${PROBE_RESULT:-logs/probe_fragility_results.jsonl}"
GM="$CKPT_DIR/_frag_merged_${SLURM_JOB_ID}"

export PROBE_RESULT="$RESULT"
export PROBE_TEMPS="${PROBE_TEMPS:-0.0,0.9,1.2,1.5}"
export PROBE_NS="${PROBE_NS:-200,350,350,350}"
export PROBE_MAXTOK="${PROBE_MAXTOK:-6144}"     # 학습 max_completion_length 와 동일
export PROBE_CONC="${PROBE_CONC:-32}"
export PROBE_DATA="${PROBE_DATA:-work/data/stage2_expanded_train.jsonl}"

echo "[frag] job=$SLURM_JOB_ID node=$(hostname) steps='$STEPS'"
echo "[frag] temps=$PROBE_TEMPS  ns=$PROBE_NS  maxtok=$PROBE_MAXTOK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 학습 체크포인트 경로 해석. 라이브 디렉토리 우선, 없으면 롤오프 전에 떠 둔 스냅샷.
resolve_ckpt() {
  local s="$1" c
  c=$(ls -d "$CKPT_DIR"/grpo_expanded_gdpo/v*/checkpoint-"$s" 2>/dev/null | tail -1)
  [ -n "$c" ] && { echo "$c"; return; }
  [ -d "$CKPT_DIR/_mideval_snap_step$s" ] && { echo "$CKPT_DIR/_mideval_snap_step$s"; return; }
  echo ""
}

for STEP in $STEPS; do
  CKPT=$(resolve_ckpt "$STEP")
  if [ -z "$CKPT" ]; then echo "[frag] ❌ step $STEP 체크포인트 없음 — 건너뜀"; continue; fi
  echo ""; echo "=================  step $STEP  ($CKPT)  ================="

  rm -rf "$GM"
  run_py swift export --adapters "$CKPT" --merge_lora true --output_dir "$GM" \
    > "logs/probe_frag_merge_${SLURM_JOB_ID}_${STEP}.log" 2>&1
  if [ ! -f "$GM/config.json" ]; then
    echo "[frag] ❌ step $STEP merge 실패"; tail -20 "logs/probe_frag_merge_${SLURM_JOB_ID}_${STEP}.log"; continue
  fi
  echo "[frag] merged → $GM"

  run_serve "$GM" --served-model-name probemodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > "logs/probe_frag_serve_${SLURM_JOB_ID}_${STEP}.log" 2>&1 &
  srv=$!
  ready=0
  for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q probemodel && { ready=1; break; }
    kill -0 $srv 2>/dev/null || { echo "[frag] ❌ step $STEP server died"; tail -25 "logs/probe_frag_serve_${SLURM_JOB_ID}_${STEP}.log"; break; }
    sleep 10
  done

  if [ "$ready" = "1" ]; then
    echo "[frag] step $STEP READY"
    # run_py 는 쉘 함수다. `VAR=x run_py ...` 형태의 앞자리 대입은 bash 에서 자식
    # 프로세스로 export 된다는 보장이 없다 → 반드시 export 로 넘긴다.
    export PROBE_BASE_URL="http://localhost:$PORT/v1"
    export PROBE_MODEL=probemodel
    export PROBE_TAG="step$STEP"
    export PROBE_OUT="logs/probe_frag_samples_step${STEP}.jsonl"
    run_py python scripts/probe_format_fragility.py
  fi

  kill $srv 2>/dev/null; sleep 5; kill -9 $srv 2>/dev/null || true
  rm -rf "$GM"
done

echo ""; echo "=====  형식 취약성 스윕 결과  ====="
cat "$RESULT" 2>/dev/null
echo "[frag] done"
