#!/usr/bin/env bash
# loco_chain_v6f.sh — 풀이 줄이기 6차 다시 하기: 6차와 같은 798쌍·같은 설정, 좋은 답만 엄격 형식으로 고친 짝(loco/pairs_inputs6f.tar).
# 왜: 6차 짝의 좋은 답 100% 가 `</think>` 뒤에 설명을 둬서 6차 모델이 그 버릇을 배웠다(2026-11-13, scripts/fmt_strict.py).
# 학습 1개 → 평가 2개(loco_eval6.sh·loco_eval_med6.sh, V=v6f)를 pod 하나씩 이어 돌린다. rtxpro6000 이 1장뿐이라 자리가 날 때까지 10분마다 다시 빌린다.
# 사용: set -a; . ./.env; set +a; python3 로 start_new_session 분리 실행 > logs/loco_chain_v6f.out
cd "$(dirname "$0")/.." || exit 1
L=${LOG_DIR:-logs}   # OneDrive 폴더 안 로그는 동기화가 파일을 바꿔치기해 기록이 사라진다(2026-11-14 실측) → 밖에 둘 것
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/loco | grep -q "\"$1\""; }
# run <완료 표시 파일> <로그> <명령...>: 자리 없음(gpu_unavailable)은 최대 WAIT_TRIES 번 10분 간격으로, 그 밖의 실패는 3번까지 다시 한다
run() {
  local done=$1 log=$2; shift 2; local fail=0
  for i in $(seq "${WAIT_TRIES:-144}"); do
    have "$done" && return 0
    echo "[$(date +%H:%M)] $* (시도 $i)"; "$@" > "$log" 2>&1
    have "$done" && return 0
    if grep -q gpu_unavailable "$log"; then sleep 600; else fail=$((fail+1)); [ $fail -ge 3 ] && return 1; sleep 700; fi
  done; return 1
}
V=${V:-v6f}   # 같은 짝으로 다른 손실을 시험할 때 이름만 바꾼다(v6r = RPO=1.0)
INPUT_TAR=pairs_inputs6f.tar OUT=adapter_$V EPOCHS=1 SAVE_STEPS=25 SAVE_LIMIT=4 MAX_SEC=14400 RPO=${RPO:-} LOSS=${LOSS:-} \
  run adapter_${V}_100 "$L/loco_dpo_$V.out" bash kgpu/run_pod.sh loco_dpo.sh INPUT_TAR OUT EPOCHS SAVE_STEPS SAVE_LIMIT RPO LOSS \
  && echo "[$(date +%H:%M)] 학습 완료" || { echo "[$(date +%H:%M)] ❌ 학습 실패 — 중단"; exit 1; }
sleep 700   # 앞 pod 가 자기 반납(GRACE 600초)을 끝낼 때까지 — 한 번에 pod 하나
for pair in "loco_eval6.sh eval_$V.jsonl" "loco_eval_med6.sh eval_med_$V.jsonl"; do
  set -- $pair
  V=$V MAX_SEC=18000 run "$2" "$L/${1%.sh}_$V.out" bash kgpu/run_pod.sh "$1" HF_TOKEN MAX_SEC V \
    && echo "[$(date +%H:%M)] $2 완료" || { echo "[$(date +%H:%M)] ❌ $1 실패 — 중단"; exit 1; }
  sleep 700
done
echo "[$(date +%H:%M)] ALL_DONE"
