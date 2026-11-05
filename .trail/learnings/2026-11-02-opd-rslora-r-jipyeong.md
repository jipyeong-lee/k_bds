---
title: OPD 어댑터는 rsLoRA 라서 병합할 때 스케일을 α/√r 로 써야 한다
author: Jipyeong-lee
date: 2026-11-02
branch: main
id: l-a1a43c7b
scope: repo
status: active
src: agent
touches:
  - scripts/merge_experts.py
  - scripts/test_merge_experts.py
---

## Decided
- OPD 학습은 --use_rslora true 로 어댑터를 만든다. 병합 도구가 rsLoRA 를 거부해서 첫 평가 병합(job 84053)이 1분 만에 실패했다(2026-11-02). 병합 도구가 rsLoRA 를 받아 스케일을 peft 와 같은 α/√r 로 계산하게 고쳤다. r32·α32 면 스케일이 1 이 아니라 √32≈5.66 이다.
- 모듈별 r/α(rank_pattern·alpha_pattern)와 DoRA 는 여전히 거부한다. 조용히 틀린 모델을 만들지 않게 하려는 것이다.

In the code:
- scripts/merge_experts.py load_adapter: scale = lora_alpha / (√r if use_rslora else r)
- scripts/test_merge_experts.py 에 rslora 케이스 추가, 10/10 통과
- 재제출: merge 84071 → holdout 84072 · gbench 84073

## Tried and ruled out
- 학습을 rsLoRA 없이 다시 돌리기: 이미 끝난 100 step(약 1시간 40분, 8 노드시간)을 버리게 된다.
- rsLoRA 어댑터를 α/r 로 병합하기: 업데이트 크기가 1/√r 배(r32 면 약 1/5.7)로 줄어든 틀린 모델이 나온다. 오류 없이 돌아가서 평가 점수만 이상하게 나온다.

## Watch out
- KISTI 로그인 노드의 python3 에는 torch 가 없다. 테스트는 source scripts/00_common.sh 뒤 run_py python scripts/test_merge_experts.py 로 컨테이너 안에서 돌린다.
- 병합이 실패하면 afterok 로 묶인 평가 job 은 PENDING(Dependency)으로 영영 남는다. scancel 하고 다시 제출해야 한다.
- KISTI 의 merge_experts.py 가 로컬보다 오래된 판이었다(.cache 제외 수정이 빠져 있었다). 올리기 전에 diff 할 것.
