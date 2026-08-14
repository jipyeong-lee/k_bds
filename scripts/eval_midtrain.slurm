#!/bin/bash
# =============================================================================
# eval_midtrain.slurm — Stage-2 중간 홀드아웃 벤치마크 (조기 판단용)
#   본실행이 길어(~9일) → 중간 체크포인트에서 홀드아웃 정확도가 오르는지 조기 확인.
#   base / init(RL 0%) / trained(중간 체크포인트) 3개를 층화 홀드아웃에서 동일조건
#   채점(accuracy_mix, math/vl 층별 분리).
#   핵심 질문: init 대비 trained 가 홀드아웃 Acc 를 올렸나?(=Stage-2 RL 효과 격리)
#
#   2026-08-02 갱신:
#     - init 을 현재 런 기준 v3 `sft_mixed_merged` 로 교체(구 `sft_rft_coldstart_merged` 아님)
#     - MID_CKPT 미지정 시 grpo_expanded_gdpo 의 최신 checkpoint 자동 선택 + 스냅샷
#     - `singularity exec` 직접호출 → `run_serve`/`run_py` 로 교체. apptainer 파손(07-27~)
#       환경에서 종전 코드는 반드시 실패했다. 상세 = docs/stage2_run73924_progress.md
#     - EVAL_STAGES / TRAINED_TAG 추가 + 임시경로·결과파일을 잡별로 분리 → 병렬 제출 가능.
#       추세 평가 예: 이미 측정한 base·init 은 건너뛰고 체크포인트만 여러 잡으로 동시에.
#         for S in 400 500; do
#           sbatch --partition=debug-1gpu \
#             --export=ALL,EVAL_STAGES=trained,TRAINED_TAG=step$S,EVAL_N=300,\
#         MID_CKPT=$PWD/work/checkpoints/_mideval_snap_step$S,\
#         EVAL_DATA=work/data/stage2_expanded_holdout.jsonl,EVAL_PORT=$((8165+S/100)) \
#             scripts/eval_midtrain.slurm
#         done
#       ⚠️ EVAL_N·EVAL_SEED 가 같아야 과거 실행과 동일 슬라이스 → 직접 비교 가능
# =============================================================================
#SBATCH --job-name=eval-mid
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/home01/k266a01/kbds_project/logs/eval_midtrain_%j.log
#SBATCH --error=/home01/k266a01/kbds_project/logs/eval_midtrain_%j.log

set -u
cd /home01/k266a01/kbds_project
source scripts/00_common.sh
PORT="${EVAL_PORT:-8165}"
#  EVAL_N=all → 홀드아웃 전량(1,772 = deepvision 972 / mmk12 400 / pmcvqa 400).
#    숫자를 주면 _source 균등 층화 추출이고, 소스별 상한(400) 때문에 1,600 이 최대다.
#    ⚠️ 전량과 층화 표본은 소스 가중치가 다르므로 전체 accuracy 를 서로 비교하면 안 된다.
EVAL_N="${EVAL_N:-all}"
#  ⚠️ 기본값을 반드시 박아둘 것. 종전에는 미지정 시 eval_compare.py 가 구 홀드아웃
#     (deepvision_holdout.jsonl, 972건, _source 없음)으로 조용히 떨어졌다. 에러 없이
#     그럴듯한 수치가 나와서 알아채기 어렵다 — 실제로 주석에만 있고 배선은 없었다.
EVAL_DATA="${EVAL_DATA:-work/data/stage2_expanded_holdout.jsonl}"
export EVAL_DATA
[ -f "$EVAL_DATA" ] || { echo "[mid] ❌ 평가셋 없음: $EVAL_DATA"; exit 1; }
echo "[mid] EVAL_DATA=$EVAL_DATA ($(wc -l < "$EVAL_DATA") 건)  EVAL_N=$EVAL_N"
#  평가할 대상 선택 — 이미 측정한 baseline 을 다시 돌리지 않기 위함.
#    EVAL_STAGES=trained  →  trained 만 (init/base 는 기존 측정값 재사용)
#  EVAL_N 과 EVAL_SEED 가 같으면 슬라이스가 동일하므로 과거 실행과 직접 비교된다.
EVAL_STAGES="${EVAL_STAGES:-base,init,trained}"
TRAINED_TAG="${TRAINED_TAG:-trained}"
#  ⚠️ 잡별로 분리해야 병렬 제출이 안전하다. 종전에는 고정 경로 + rm -rf 라
#     두 잡을 동시에 띄우면 서로의 병합본과 결과 파일을 파괴했다.
RESULT="${EVAL_RESULT_FILE:-logs/eval_midtrain_results_${SLURM_JOB_ID:-manual}.jsonl}"
: > "$RESULT"

SNAP="$CKPT_DIR/_mideval_snapshot_${SLURM_JOB_ID:-manual}"
MERGED="$CKPT_DIR/_midtrain_merged_${SLURM_JOB_ID:-manual}"
export NPROC_PER_NODE=1
trap 'rm -rf "$SNAP" "$MERGED"' EXIT      # 중간 실패해도 임시본을 남기지 않는다

has_stage() { case ",$EVAL_STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

# ---- 0) 중간 체크포인트 스냅샷 + 병합 (trained 단계에만 필요) ----------------
#  학습이 save_total_limit=5 로 checkpoint 를 롤링 삭제한다. 평가 도중 원본이 사라질 수
#  있으므로 반드시 스냅샷을 떠서 그 복사본에서 병합한다.
#  ⚠️ base/init 만 평가할 때는 건너뛴다. 종전에는 무조건 실행돼서, MID_CKPT 미지정 시
#     **학습 중인 디렉터리에서 최신 checkpoint 를 자동 선택**해 복사·병합했다.
#     쓰지도 않을 5분짜리 병합인 데다, 기록 중인 체크포인트를 집을 위험이 있었다.
if has_stage trained; then
  CKPT="${MID_CKPT:-$(ls -d "$CKPT_DIR"/grpo_expanded_gdpo/*/checkpoint-* 2>/dev/null \
          | awk -F'checkpoint-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)}"
  [ -n "${CKPT:-}" ] && [ -d "$CKPT" ] \
    || { echo "[mid] ❌ 평가할 체크포인트가 없다 (MID_CKPT 로 지정할 것)"; exit 1; }
  rm -rf "$SNAP" "$MERGED"
  echo "[mid] snapshot $CKPT -> $SNAP"
  cp -a "$CKPT" "$SNAP" || { echo "[mid] ❌ 스냅샷 실패"; exit 1; }
  echo "[mid] merge $SNAP -> $MERGED"
  run_py swift export --adapters "$SNAP" --merge_lora true --output_dir "$MERGED" \
    || { echo "[mid] ❌ merge 실패"; exit 1; }
else
  echo "[mid] EVAL_STAGES=$EVAL_STAGES — trained 없음, 체크포인트 스냅샷·병합 건너뜀"
fi

# ---- 평가 함수 (40_eval_compare.slurm 과 동일 로직, 결과 파일만 별도) --------
run_eval() {  # $1=tag  $2=model
  local tag="$1" model="$2"
  echo "=================  [$tag] $model  ================="
  run_serve "$model" --served-model-name evalmodel \
      --max-model-len 10240 --gpu-memory-utilization 0.85 --enforce-eager \
      --limit-mm-per-prompt '{"image": 1}' --host 0.0.0.0 --port "$PORT" \
      > logs/eval_mid_serve_${tag}_${SLURM_JOB_ID:-manual}.log 2>&1 &
  local srv=$! ready=0
  for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q evalmodel && { ready=1; echo "[$tag] READY (~$((i*10))s)"; break; }
    kill -0 $srv 2>/dev/null || { echo "[$tag] ❌ server died"; tail -15 logs/eval_mid_serve_${tag}_${SLURM_JOB_ID:-manual}.log; break; }
    sleep 10
  done
  if [ "$ready" = "1" ]; then
    # loader 모드에선 호스트 env 가 그대로 보이므로 --env 플래그 대신 export 로 전달.
    # SYSTEM_PROMPT·HF_HUB_OFFLINE 은 00_common.sh 가 이미 export 했다.
    EVAL_BASE_URL="http://localhost:$PORT/v1" EVAL_MODEL=evalmodel \
    EVAL_N="$EVAL_N" EVAL_CONC="${EVAL_CONC:-16}" EVAL_TAG="$tag" \
    EVAL_MAXTOK="${EVAL_MAXTOK:-2048}" EVAL_RESULT="$RESULT" \
    EVAL_ITEMS="${EVAL_ITEMS_DIR:-logs}/eval_items_${tag}_${SLURM_JOB_ID:-manual}.jsonl" \
      run_py python scripts/eval_compare.py
  fi
  # ⚠️ run_serve 는 셸 함수라 $! 는 서브셸 PID 다(runc.sh 가 exec 로 python 이 됨).
  #    자손까지 정리하지 않으면 vLLM 이 살아남아 다음 tag 에서 포트가 충돌한다.
  pkill -TERM -P "$srv" 2>/dev/null || true
  kill -TERM "$srv" 2>/dev/null || true
  sleep 5
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  sleep 5
  for i in $(seq 1 12); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || break
    echo "[$tag] 포트 $PORT 해제 대기…"; sleep 5
  done
}

BASE_PATH=$(ls -d work/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)

echo "[mid] EVAL_STAGES=$EVAL_STAGES  EVAL_N=$EVAL_N  RESULT=$RESULT"
has_stage base    && run_eval base           "$BASE_PATH"                            || true
has_stage init    && run_eval init           "${INIT_MODEL:-$CKPT_DIR/sft_mixed_merged}" || true
has_stage trained && run_eval "$TRAINED_TAG" "$MERGED"                               || true

echo ""; echo "=====================  중간 벤치마크 결과 (holdout, 층별)  ====================="
cat "$RESULT"
echo "[mid] done"   # 스냅샷·병합본은 EXIT trap 이 정리한다
