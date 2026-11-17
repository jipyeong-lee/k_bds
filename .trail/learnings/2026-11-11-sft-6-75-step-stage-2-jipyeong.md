---
title: SFT 대신 풀이 줄이기 6차 75 step 을 Stage-2 시작 모델로 시험한다 — 먼저 kgpu 1장에서 deepvision GRPO 40 step 동작 확인
author: Jipyeong-lee
date: 2026-11-11
branch: main
id: l-945c2044
scope: repo
status: active
src: agent
touches:
  - CLAUDE.md
  - kgpu/loco_chain_d.sh
  - kgpu/loco_dpo.sh
  - kgpu/merge_screen.sh
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe.sh
  - kgpu/rubric_rollouts.sh
---

## Decided
- 풀이 줄이기 6차 75 step(공개 Qwen3.5-9B + 6차 LoRA)을 SFT 대체 1순위로 둔다. 전량 평가(2026-11-11): 홀드아웃 60.6%, 글로벌 벤치 56.8%. 4차와 차이는 +0.3 [-1.5, +2.1], +1.1 [-0.2, +2.4]로 유의하지 않지만 4차와 겹치지 않는 쌍으로 학습해 같은 효과가 났다. 같은 600문항에서 SFT 대비 홀드아웃 +14.3 [10.0, 18.3], 글로벌 +7.7 [3.3, 12.0].
- GRPO 를 MOPD 보다 먼저 한다. MOPD 교사는 새 시작 모델에서 키운 전문가여야 하고, 교사 서버 경로는 아직 신호가 비어 있다(4.1.3·4.6.0).
- 본학습 전에 kgpu RTX PRO 6000 1장에서 deepvision GRPO 40 step 동작 확인을 한다. 보는 것: 형식 점수 시작값, 잘림 비율, 길이가 줄어드는지, step 시간.

## Tried and ruled out
- KISTI 8gpu 에서 바로 동작 확인: 2026-11-11 노드 3개가 모두 2~4일짜리 작업으로 차 있고 우선순위 높은 대기 17개가 있어 시작이 며칠 늦어진다.
- B200: .env 에 접속 토큰이 없어 쓸 수 없다.
- kgpu 의 다른 GPU(rtx5090 32GB 등): 9B GRPO 와 6144 출력을 담기에 메모리가 작다.

## Watch out
- 6차 모델은 홀드아웃 답의 16%가 8192 토큰에서 잘린다. 형식 실패율(16.3%)과 같아서 형식 실패는 사실상 잘림이다. deepvision 상한 6144 에서는 33%가 넘는다(mmk12 11.5%, pmcvqa 0.8%). 형식 점수가 자동 중단 기준 0.85 아래에서 시작하므로 KISTI 에서 돌릴 때는 감시자 기준을 조정해야 한다.
- 1장이라 step 이 길다. mmk12(답 약 450 토큰)가 80~100 s/step 이었으니 답이 약 3,000 토큰인 이 모델은 step당 8~10분으로 추정한다(미실측).

In the code:
- kgpu/dv_loco_smoke.sh — 병합(Qwen3.5-9B c202236 + loco/adapter_v6_75) → ~/files/kbds/loco6_75_merged 저장 → swift GRPO(stable 레시피, PDTBS×ACCUM=2×16). 지표는 5분마다 logs/dv_loco_smoke_<시각>.logging.jsonl
- 실행: bash kgpu/run_pod.sh dv_loco_smoke.sh HF_TOKEN MAX_STEPS=40 SAVE_STEPS=20
- kgpu/merge_screen.sh: CANDS 에 loco6_75 처럼 밑줄 이름을 쓰면 병합 준비를 건너뛰게 고쳤다. kgpu/loco_chain_full.sh 는 CAND·WAIT 를 받는다.
