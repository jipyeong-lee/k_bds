#!/bin/bash
# =============================================================================
# watch_and_eval.sh — 목표 step 체크포인트를 기다렸다가 스냅샷 + 평가 제출
#
#   학습이 save_total_limit=5 로 체크포인트를 롤링 삭제한다(현재 50 step 간격 →
#   목표 step 저장 후 약 250 step, 즉 ~20h 뒤 사라진다). 그 창 안에 사람이 붙어
#   있어야 하는 문제를 없애기 위해, 로그인 노드에서 폴링하다가 체크포인트가
#   나타나면 즉시 안전한 위치로 복사하고 평가를 제출한다.
#
#   왜 스냅샷을 여기서 한 번 더 뜨나 — eval_midtrain.slurm 도 자체 스냅샷을 뜨지만
#   그건 **잡이 시작된 뒤**다. 큐 대기가 길면 그 사이 원본이 삭제될 수 있다.
#   여기서 뜬 복사본을 MID_CKPT 로 넘기므로 대기 시간과 무관해진다.
#
#   사용:
#     nohup setsid scripts/watch_and_eval.sh > logs/watch_step1200.log 2>&1 &
#   환경변수:
#     TARGET_STEP  (기본 1200)  감시할 step
#     EVAL_PORT    (기본 8176)  다른 평가 잡과 겹치면 안 된다
#     PARTITION    (기본 debug-1gpu)
#     MAX_WAIT_H   (기본 24)    이 시간 안에 안 나타나면 포기
#     DRY_RUN=1                 제출 없이 감지까지만
# =============================================================================
set -u
cd /home01/k266a01/kbds_project || exit 1

TARGET_STEP="${TARGET_STEP:-1200}"
EVAL_PORT="${EVAL_PORT:-8176}"
PARTITION="${PARTITION:-debug-1gpu}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
POLL_S="${POLL_S:-300}"
SNAP="$PWD/work/checkpoints/_mideval_snap_step${TARGET_STEP}"

say() { echo "[watch $(date '+%m-%d %H:%M:%S')] $*"; }

say "감시 시작 — step $TARGET_STEP, 최대 ${MAX_WAIT_H}h, ${POLL_S}s 간격"

# 이미 스냅샷이 있으면 중복 작업하지 않는다(재실행 안전).
if [ -d "$SNAP" ]; then
  say "스냅샷이 이미 있다: $SNAP — 복사 건너뛰고 제출로 간다"
else
  DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  CKPT=""
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # 재개하면 v1 → v2 로 런 디렉터리가 바뀐다. 글롭으로 전부 훑는다.
    CAND=$(ls -d work/checkpoints/grpo_expanded_gdpo/v*/checkpoint-"$TARGET_STEP" 2>/dev/null | tail -1)
    if [ -n "$CAND" ] && [ -f "$CAND/trainer_state.json" ]; then
      # 아직 쓰는 중일 수 있다. 크기가 30초간 고정이면 완료로 본다.
      s1=$(du -sb "$CAND" 2>/dev/null | cut -f1)
      sleep 30
      s2=$(du -sb "$CAND" 2>/dev/null | cut -f1)
      if [ -n "$s1" ] && [ "$s1" = "$s2" ]; then
        CKPT="$PWD/$CAND"; say "감지: $CKPT ($((s2/1024/1024)) MB)"; break
      fi
      say "쓰는 중으로 보인다($s1 → $s2) — 대기"
    fi
    sleep "$POLL_S"
  done
  [ -n "$CKPT" ] || { say "❌ ${MAX_WAIT_H}h 안에 step $TARGET_STEP 이 나타나지 않았다 — 포기"; exit 1; }

  # ---- 스냅샷 -------------------------------------------------------------
  #  ⚠️ 전체 복사(cp -a)로 뜬다. 병합에 실제로 필요한 건 어댑터 4개 파일뿐인 것으로
  #     보이지만, step500·600 에서 성공한 스냅샷은 전부 전체 복사본이었다. 검증되지
  #     않은 축소본으로 바꿔서 아낄 수 있는 건 333MB 와 몇 초뿐이고(롤오프 여유는
  #     ~20h), 실패하면 그 체크포인트는 영영 사라진다. 남는 쪽에 걸 이유가 없다.
  say "스냅샷 → $SNAP"
  TMP="${SNAP}.partial"
  rm -rf "$TMP"
  cp -a "$CKPT" "$TMP" || { say "❌ 복사 실패"; rm -rf "$TMP"; exit 1; }
  # 원본이 삭제 중이었다면 크기가 안 맞을 수 있다. 어댑터 크기를 대조한다.
  a=$(stat -c%s "$CKPT/adapter_model.safetensors" 2>/dev/null)
  b=$(stat -c%s "$TMP/adapter_model.safetensors" 2>/dev/null)
  [ -n "$a" ] && [ "$a" = "$b" ] || { say "❌ 어댑터 크기 불일치($a vs $b) — 중단"; rm -rf "$TMP"; exit 1; }
  mv "$TMP" "$SNAP"
  say "스냅샷 완료 ($((b/1024/1024)) MB) — 원본이 롤오프돼도 안전하다"
fi

# ---- 평가 제출 ------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then say "DRY_RUN=1 — 제출하지 않고 종료"; exit 0; fi

say "평가 제출 (partition=$PARTITION, port=$EVAL_PORT)"
JOB=$(sbatch --parsable --partition="$PARTITION" \
  --export=ALL,EVAL_STAGES=trained,TRAINED_TAG=step${TARGET_STEP},EVAL_N=all,MID_CKPT=$SNAP,EVAL_DATA=work/data/stage2_expanded_holdout.jsonl,EVAL_PORT=$EVAL_PORT \
  scripts/eval_midtrain.slurm 2>&1)
if [[ "$JOB" =~ ^[0-9]+$ ]]; then
  say "✅ 제출됨: job $JOB"
  say "결과: logs/eval_midtrain_results_${JOB}.jsonl"
  say "문항별: logs/eval_items_step${TARGET_STEP}_${JOB}.jsonl"
  say "짝지음: python3 scripts/eval_paired.py logs/eval_items_step400_74137.jsonl logs/eval_items_step${TARGET_STEP}_${JOB}.jsonl --by source"
else
  say "❌ 제출 실패: $JOB"; exit 1
fi
