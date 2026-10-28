---
title: "OPD 증류는 배선 스모크부터 하고, 효과가 분명할 때만 통합 모델을 대체한다"
author: Jipyeong-lee
date: 2026-10-28
branch: main
cost: { minutes: 1, tokens: 3388 }
id: l-2275492f
scope: repo
status: active
src: agent
touches:
  - kgpu/run_pod.sh
  - kgpu/opd_smoke.sh
  - docs/21_stage2a_design.html
---

## Decided
- OPD(온폴리시 증류)는 본실행 전에 배선 스모크로 시작했다. 학생은 통합 모델(ta_w07), 교사는 SFT 모델에 deepvision 전문가 어댑터를 붙인 것이다(ms-swift GKD, kgpu 1 GPU).
- 교사에 어댑터가 실제로 붙는다는 것을 확정했다(2026-10-28). 같은 고정 응답 4개를 먹인 step 1 loss 가 어댑터 있음 0.3856 대 없음 0.7088 이고, grad_norm 은 41.2 대 64.6 이다. 메모리 차이 0.16 GiB 는 어댑터 파일 크기(173MB)와 같다.
- OPD 결과가 ta_w07 보다 분명히 좋을 때만 Stage-3 초기 모델로 바꾼다. 효과가 노이즈 범위(holdout 표준오차 약 1.5pp)면 ta_w07 을 그대로 쓴다. 줄일 수 있는 격차는 최대 3pp 안팎이다(시각추론 holdout 64.20 대 67.39, 글로벌 벤치 56.22 대 57.52). kgpu rtxpro6000 노드가 1개뿐이라 OPD 와 Stage-3 는 순서대로 한다.

In the code:
- kgpu/opd_smoke.sh: FIXED_JSON(고정 응답, --lmbda 0, vLLM 끔)과 NO_ADAPTER=1(어댑터 뺀 대조)을 같은 설정으로 두 번 돌려 step 1 loss 를 비교한다. kgpu/run_pod.sh opd_smoke.sh HF_TOKEN … 로 pod 를 빌려 돌린다.
- 측정: 10 step 에 34 s/it, 최대 73.4 GiB(학생+교사+vLLM, 96GB 안).
- 설계 근거: docs/21_stage2a_design.html 4부.

## Tried and ruled out
- 샘플링한 응답으로 비교하기: 응답이 달라서 loss 차이를 어댑터 탓으로 돌릴 수 없었다. top_k=1(그리디)로도 응답이 같지 않았다. 확률이 거의 같은 토큰에서 갈리고 뒤로 갈수록 길이가 달라진다.
- 메모리 간격 2.5~3 GiB 를 어댑터 증거로 본 것은 철회했다. 응답 길이 같은 요인이 섞였다. 응답을 고정하면 0.16 GiB 다.
- 계단 1 순차 GKD 본실행과 계단 2 도메인 라우팅 다중 교사: 아직 안 했다. 계단 2 는 ms-swift 가 교사를 하나만 받아서 site-packages 패치가 필요하다.

## Watch out
- ms-swift 4.1.3 은 교사 어댑터를 --reward_adapters 에서 읽는다(pipelines/train/rlhf.py:94). --teacher_adapters 는 조용히 무시된다.
- GKD 에 vLLM colocate 를 쓰려면 swift rlhf 앞에 NPROC_PER_NODE=1 이 있어야 한다. 없으면 KeyError: 'RANK' 로 죽는다.
- use_vllm=false 일 때 --vllm_mode 를 주면 인자 검증에서 거부된다(rlhf_args.py:393).
- 스모크 프롬프트는 medix 200행이다. 본실행은 도메인 학습 프롬프트가 필요하고 deepvision 학습 데이터는 KISTI 에만 있다.
