---
title: "5차 학습 데이터는 원본 모델 자신의 답 1만 문항으로 모으고, 한 번에 pod 하나만 돌린다"
author: Jipyeong-lee
date: 2026-11-08
branch: main
id: l-a37502e4
scope: repo
status: active
src: agent
touches:
  - CLAUDE.md
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe.sh
  - kgpu/rubric_rollouts.sh
  - scripts/judge_server.sh
---

## Decided
- 4차(2200문항, 754쌍)에서 길이 감소 효과가 확인되어, 5차는 수학 문항 1만 개로 원본 모델 자신의 정답을 모아 쌍을 만든다. 끝나면 sft_mixed_merged와 같은 수학 150문항·의료 200문항에서 비교한다.
- 사용자가 pod는 한 번에 하나만 돌리라고 정했다. 그래서 1250문항 조각 8개를 이어서 돌린다.

## Tried and ruled out
- 4만 문항 전부: 4만 문항은 시간이 너무 길어 제외했다. 4000문항: 규모가 작아 제외했다.
- pod 8개 병렬: kgpu가 동시 4개까지만 허용했다(gpu_quota_exceeded). 4개로 띄워도 3개가 Pending에서 시작 실패로 반납됐다.

## Watch out
- 원본 답 생성은 끝에서만 결과를 저장한다. 시작 직후에는 `pod_not_running` 실패가 로그에 늦게 찍히니 몇 분 뒤 다시 확인한다.
- 1조각이 3시간 반쯤 걸려 8조각이면 약 30시간이다. 쌍 수·학습 시간(약 20시간)은 4차 비율에서 잡은 추정이다.

In the code: kgpu/loco_chain_d.sh(조각 d2~d8을 이어 돌림, 실패 시 3번 재시도) · OFFSET=3000+1250*(n-1), N_Q=1250, N_GEN=4 · 결과 loco/rollouts_dN.jsonl · 진행도 dashboard/data_src/plan.json
