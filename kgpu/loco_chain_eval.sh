#!/usr/bin/env bash
# loco_chain_eval.sh — 5차 학습(adapter_v5)이 끝나면 평가 4개를 한 번에 하나씩 이어서 돌린다(수학·의료 × 5차·SFT 모델).
# 사용: set -a; . ./.env; set +a; python3 로 start_new_session 분리 실행 > logs/loco_chain_eval.out
# 각 평가는 결과 파일이 생길 때까지 최대 3번 시도한다(시작 실패 시 pod 가 바로 반납되기 때문).
cd "$(dirname "$0")/.." || exit 1
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/loco | grep -q "\"$1\""; }
echo "[$(date +%H:%M)] adapter_v5 기다림"; until have adapter_v5; do sleep 180; done; echo "[$(date +%H:%M)] 학습 완료"
sleep 700   # 학습 pod 가 자기 반납(GRACE 600초)을 끝낼 때까지 기다린다 — 동시에 두 pod 를 빌리지 않는다
for pair in "loco_eval5.sh eval_v5.jsonl" "loco_eval_med5.sh eval_med_v5.jsonl" "loco_eval_sft.sh eval_sft.jsonl" "loco_eval_med_sft.sh eval_med_sft.jsonl"; do
  set -- $pair
  for try in 1 2 3; do
    have $2 && break
    echo "[$(date +%H:%M)] $1 시작(시도 $try)"
    MAX_SEC=10800 bash kgpu/run_pod.sh $1 HF_TOKEN MAX_SEC > logs/${1%.sh}.out 2>&1
    sleep 30
  done
  have $2 && echo "[$(date +%H:%M)] $2 완료" || { echo "[$(date +%H:%M)] ❌ $1 3번 실패 — 중단"; exit 1; }
done
echo "[$(date +%H:%M)] ALL_DONE"
