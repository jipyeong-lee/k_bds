#!/usr/bin/env bash
# loco_chain_full.sh — 5차 평가 체인(loco_chain_eval.sh)이 끝나면 4차 모델의 도메인 홀드아웃(1,772)·글로벌 벤치(3,712) 전량 평가를 돌린다.
# 조건은 기존 full_base_* 와 같다(merge_screen.sh, greedy, 홀드아웃 8192 · 글로벌 4096) → 원본 Qwen3.5-9B 와 문항끼리 비교할 수 있다.
# 사용: set -a; . ./.env; set +a; python3 로 start_new_session 분리 실행 > logs/loco_chain_full.out
cd "$(dirname "$0")/.." || exit 1
have() { curl -s -H "Authorization: Bearer $KGPU_API_TOKEN" https://api.kgpu.net/v1/files/kbds/logs | grep -q "\"$1\""; }
C=${CAND:-loco4}   # 예: CAND=loco6_75 → loco/adapter_v6_75
echo "[$(date +%H:%M)] 평가 체인 끝나길 기다림"; until grep -q "ALL_DONE\|❌" logs/loco_chain_eval.out; do sleep 300; done
sleep "${WAIT:-700}"   # 앞 pod 가 자기 반납(GRACE 600초)을 끝낼 때까지 — 한 번에 pod 하나
for try in 1 2 3; do
  have full_${C}_gb.jsonl && break
  echo "[$(date +%H:%M)] merge_screen $C 시작(시도 $try)"
  CANDS=$C HOLD_N=all GB_N=all RES_PREFIX=full MAX_SEC=36000 bash kgpu/run_pod.sh merge_screen.sh HF_TOKEN CANDS HOLD_N GB_N RES_PREFIX > logs/merge_screen_$C.out 2>&1
  sleep 30
done
have full_${C}_gb.jsonl && echo "[$(date +%H:%M)] ALL_DONE" || echo "[$(date +%H:%M)] ❌ 3번 실패"
