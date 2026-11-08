#!/usr/bin/env bash
# loco_chain_d.sh — 5차용 원본 답 모으기를 한 번에 하나씩 이어서 돌린다(d2 는 이미 돌고 있어 끝나길 기다린다).
# 사용: set -a; . ./.env; set +a; setsid nohup bash kgpu/loco_chain_d.sh > logs/loco_chain_d.out 2>&1 < /dev/null & disown
# 각 조각은 결과 파일 loco/rollouts_dN.jsonl 이 생길 때까지 최대 3번 시도한다(시작 실패 시 pod 가 바로 반납되기 때문).
cd "$(dirname "$0")/.." || exit 1
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/loco | grep -q "rollouts_d$1.jsonl"; }
echo "[$(date +%H:%M)] d2 끝나길 기다림"; until have 2; do sleep 120; done; echo "[$(date +%H:%M)] d2 완료"
for n in 3 4 5 6 7 8; do
  for try in 1 2 3; do
    have $n && break
    echo "[$(date +%H:%M)] d$n 시작(시도 $try)"
    OFFSET=$((3000 + 1250 * (n - 1))) TAG=_d$n SKIP_API=1 N_Q=1250 N_GEN=4 MAX_SEC=28800 \
      bash kgpu/run_pod.sh loco_rollouts.sh HF_TOKEN OFFSET TAG SKIP_API N_Q N_GEN > logs/loco_rollouts_d$n.out 2>&1
    sleep 30
  done
  have $n && echo "[$(date +%H:%M)] d$n 완료" || { echo "[$(date +%H:%M)] ❌ d$n 3번 실패 — 중단"; exit 1; }
done
echo "[$(date +%H:%M)] ALL_DONE"
