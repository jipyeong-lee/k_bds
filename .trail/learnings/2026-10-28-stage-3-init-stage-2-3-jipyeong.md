---
title: Stage-3 init 은 Stage-2 전문가 3종(deepvision·mmk12·pmcvqa)을 통합한 모델이어야 한다
author: Jipyeong-lee
date: 2026-10-28
id: l-1af93410
scope: repo
status: active
touches:
  - HANDOFF.md
---

## Decided
파이프라인 순서: Stage-2 도메인 전문가 3종 → 어댑터 통합(가중평균 병합 또는 OPD 증류, HANDOFF.md §5-3) → Stage-3 의료 RL.
2026-10-15 에 kgpu Stage-3 스모크·측정을 구 혼합 실행 승자 `dr_grpo_merged` 로 돌렸다가 사용자가 "3가지 teacher 만들고 그거 기반으로 더 학습해야 하는 거 아니냐"고 짚었다.

2026-10-15 KISTI 상태(`work/checkpoints/`): deepvision 이어학습 `expert_deepvision_continued_0904` 2579/2700 진행 중,
pmcvqa `expert_pmcvqa_0821-2028/checkpoint-4896` 완주, mmk12 는 `probe_mmk12_beta001` 이 붕괴 지점(450)을 넘어 500 까지 형식 유지.

**Why:** kgpu 측정값(85 s/step)·judge 선정은 그대로 유효하지만 init 이 계획서와 다르면 본실행 결과가 산출물로 쓸모없다.

## Watch out
Stage-3 본실행 제안 전에 전문가 3종 완료 여부와 통합 방식부터 확인할 것. 관련 compute-availability-2026-10.

_From Claude Code's memory of this repo, last updated Oct 15._
