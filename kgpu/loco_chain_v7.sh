#!/usr/bin/env bash
# loco_chain_v7.sh — 풀이 줄이기 7차: 6차 체인(loco_chain_v6.sh)이 끝나면 학습 1개 → 평가 2개를 한 번에 pod 하나씩 이어 돌린다.
# 7차 = 6차와 같은 800쌍·100 step·설정에서 LoRA 대상만 all-linear(선형 어텐션 24층·MLP 32층까지). 25 step 마다 어댑터를 남긴다.
# 왜: q/k/v/o 는 Qwen3.5 32층 중 일반 어텐션 8층에만 붙는다. 학습 범위를 넓히면 풀이가 더 줄어드는지 본다.
# 사용: set -a; . ./.env; set +a; python3 로 start_new_session 분리 실행 > logs/loco_chain_v7.out
cd "$(dirname "$0")/.." || exit 1
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/loco | grep -q "\"$1\""; }
echo "[$(date +%H:%M)] 6차 체인 끝나길 기다림"; until grep -q "ALL_DONE\|❌" logs/loco_chain_v6.out; do sleep 300; done
mv logs/loco_dpo_live.txt logs/loco_dpo6_live.txt 2>/dev/null   # 6차 실시간 손실을 떼어 둔다(7차가 같은 파일을 덮어쓴다)
sleep 700   # 앞 pod 가 자기 반납(GRACE 600초)을 끝낼 때까지 — 한 번에 pod 하나
for try in 1 2 3; do
  have adapter_v7_100 && break
  echo "[$(date +%H:%M)] loco_dpo 7차 시작(시도 $try)"
  INPUT_TAR=pairs_inputs6.tar OUT=adapter_v7 TARGETS=all-linear EPOCHS=1 SAVE_STEPS=25 SAVE_LIMIT=4 MAX_SEC=21600 \
    bash kgpu/run_pod.sh loco_dpo.sh INPUT_TAR OUT TARGETS EPOCHS SAVE_STEPS SAVE_LIMIT > logs/loco_dpo7.out 2>&1
  sleep 30
done
have adapter_v7_100 && echo "[$(date +%H:%M)] 학습 완료" || { echo "[$(date +%H:%M)] ❌ 학습 3번 실패 — 중단"; exit 1; }
sleep 700
for pair in "loco_eval7.sh eval_v7.jsonl" "loco_eval_med7.sh eval_med_v7.jsonl"; do
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
