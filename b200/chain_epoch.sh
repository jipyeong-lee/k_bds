#!/usr/bin/env bash
# chain_epoch.sh — KISTI 에서 돌리는 재투입 루프. 노드 job 은 60분(또는 timeout_sec)에 killed 되므로
# 1 epoch 을 채우려면 수백 번 다시 넣어야 한다. run_epoch.sh 가 체크포인트에서 이어받으므로
# 여기서는 "죽으면 다시 넣는다"만 한다.
#
#   nohup bash chain_epoch.sh deepvision 7200 > chain_deepvision.log 2>&1 &
#
# run_epoch.sh 가 1 epoch 을 채우면 "1 epoch 완료" 를 찍고 exit 0 → 루프 종료.
set -u
cd "$(dirname "$0")" || exit 1
ARM="${1:-deepvision}"
TMO="${2:-7200}"
MAXJOB="${3:-2000}"          # 폭주 방지 상한. 21일치라도 600 회면 충분하다.
LOG="ep_${ARM}.log"

# drive_node.sh 는 payload 만 stdin 으로 보내고 로컬 env 를 노드에 넘기지 않는다 →
# ARM 을 환경변수로 줘봐야 노드에서는 기본값이 된다. arm 을 박아넣은 payload 를 따로 만든다.
PAYLOAD="run_epoch_${ARM}.sh"
sed "s|^ARM=\"\${ARM:-deepvision}\"|ARM=\"$ARM\"|" run_epoch.sh > "$PAYLOAD"
grep -q "^ARM=\"$ARM\"" "$PAYLOAD" || { echo "❌ $PAYLOAD 에 ARM 치환 실패 — run_epoch.sh 형식 확인"; exit 1; }
# 학습 job 이 8 GPU 를 전부 잡으면 추가 세션이 거부된다 → 진행 확인은 job 사이 틈에서만 가능하다.
CHECK="check_progress_${ARM}.sh"
sed "s|^ARM=\"\${ARM:-deepvision}\"|ARM=\"$ARM\"|" check_progress.sh > "$CHECK"
echo "[chain] payload=$PAYLOAD, check=$CHECK (ARM=$ARM 고정)"

i=0
while [ "$i" -lt "$MAXJOB" ]; do
  i=$(( i + 1 ))
  echo "=== [chain] job #$i  $(date +%F_%T)  arm=$ARM tmo=$TMO ==="
  T=$(mktemp)
  bash run_node_extract.sh "$PAYLOAD" "$TMO" 8 > "$T" 2>&1
  cat "$T" >> "$LOG"

  # 노드가 아직 이전 job 에 물려 있으면 세션이 안 열린다. 즉시 재시도하면 실패만 빠르게 쌓인다.
  if grep -q "session create failed" "$T"; then
    echo "[chain] 세션 점유 중 → 5분 후 재시도"
    rm -f "$T"; i=$(( i - 1 )); sleep 300; continue
  fi
  rm -f "$T"

  # 완료 판정은 run_epoch.sh 의 출력에만 의존한다. 단, job 이 killed 되면 stdout 이 통째로
  # 비므로 "완료" 문자열이 안 보이는 게 정상이다 → 로그 전체에서 한 번이라도 나왔는지 본다.
  if grep -q "1 epoch 완료" "$LOG"; then
    echo "=== [chain] $ARM 1 epoch 완료 — 루프 종료 (job $i 회) ==="
    break
  fi

  # job 이 끝난 이 틈이 노드를 볼 수 있는 유일한 창이다. 진행을 로그에 남겨두면
  # 다음 job 이 도는 2시간 동안에도 KISTI 에서 상태를 읽을 수 있다.
  sleep 20
  echo "--- [chain] 진행 기록 #$i ---" >> "$LOG"
  bash run_node_extract.sh "$CHECK" 180 1 >> "$LOG" 2>&1 || true
  sleep 10
done
[ "$i" -ge "$MAXJOB" ] && echo "=== [chain] ⚠️ MAXJOB=$MAXJOB 도달 — 수동 확인 필요 ==="
