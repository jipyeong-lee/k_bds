#!/usr/bin/env bash
# loco_chain_v6.sh — 풀이 줄이기 6차: 4차 전량 평가(loco_chain_full.sh)가 끝나면 학습 1개 → 평가 2개를 한 번에 pod 하나씩 이어 돌린다.
# 6차 = 5차 쌍 중 4차와 안 겹치는 800쌍(loco/pairs_inputs6.tar) · 4차와 같은 설정 · 100 step 에서 멈추고 25 step 마다 어댑터를 남긴다.
# 왜: 5차는 574 step 학습으로 좋은 답 확률까지 무너져 같은 줄을 되풀이했다. 데이터 양이 아니라 step 수가 원인인지 가린다.
# 사용: set -a; . ./.env; set +a; python3 로 start_new_session 분리 실행 > logs/loco_chain_v6.out
cd "$(dirname "$0")/.." || exit 1
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/loco | grep -q "\"$1\""; }
echo "[$(date +%H:%M)] 4차 전량 평가 끝나길 기다림"; until grep -q "ALL_DONE\|❌" logs/loco_chain_full.out; do sleep 300; done
sleep 700   # 앞 pod 가 자기 반납(GRACE 600초)을 끝낼 때까지 — 한 번에 pod 하나
for try in 1 2 3; do
  have adapter_v6_100 && break
  echo "[$(date +%H:%M)] loco_dpo 6차 시작(시도 $try)"
  INPUT_TAR=pairs_inputs6.tar OUT=adapter_v6 EPOCHS=1 SAVE_STEPS=25 SAVE_LIMIT=4 MAX_SEC=14400 \
    bash kgpu/run_pod.sh loco_dpo.sh INPUT_TAR OUT EPOCHS SAVE_STEPS SAVE_LIMIT > logs/loco_dpo6.out 2>&1
  sleep 30
done
have adapter_v6_100 && echo "[$(date +%H:%M)] 학습 완료" || { echo "[$(date +%H:%M)] ❌ 학습 3번 실패 — 중단"; exit 1; }
sleep 700
for pair in "loco_eval6.sh eval_v6.jsonl" "loco_eval_med6.sh eval_med_v6.jsonl"; do
  set -- $pair
  for try in 1 2 3; do
    have $2 && break
    echo "[$(date +%H:%M)] $1 시작(시도 $try)"
    MAX_SEC=18000 bash kgpu/run_pod.sh $1 HF_TOKEN MAX_SEC > logs/${1%.sh}.out 2>&1
    sleep 30
  done
  have $2 && echo "[$(date +%H:%M)] $2 완료" || { echo "[$(date +%H:%M)] ❌ $1 3번 실패 — 중단"; exit 1; }
done
echo "[$(date +%H:%M)] ALL_DONE"
