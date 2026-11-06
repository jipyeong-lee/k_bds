---
title: ms-swift 4.6.0 교사 서버 모드는 학생 답은 보내지만 교사 응답의 prompt_logprobs 가 비어 학습 손실이 0이다 — 4.1.3·4.6.0 모두 지금은 쓸 수 없다
author: Jipyeong-lee
date: 2026-11-06
branch: main
id: l-23f7ed5d
scope: repo
status: active
replaces: l-d0c5a554
src: agent
touches:
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe46.sh
  - kgpu/stage3_smoke.sh
---

## Decided
- 2026-11-05~06 kgpu rtxpro6000 1장 실측(kgpu/opd_server_probe46.sh, 조건 A 교사 템플릿 qwen3_8, GKD 2 step, 3회 실행): 교사 요청의 마지막 메시지는 학생 답(assistant)이다(8/8). 그러나 교사 응답의 prompt_logprobs 가 4/4 None 이고, 조립된 top-k 에서 학생 응답 위치의 다음 토큰이 교사 top-20 에 드는 비율은 0/3,313(≈0). 학습 loss·grad_norm 이 0 인데 오류도 경고도 없이 끝난다.
- 4.1.3 은 학생 답이 교사에게 안 가고(l-0cd61c58), 4.6.0 은 보내지만 교사 신호가 비어 있다. 지금 스택으로 27B 교사 서버 모드 OPD 는 시작할 수 없다. 이전 판단('4.6.0 에서 4.1.3 문제가 해소된다')은 소스 읽기만 근거였고 실측으로 틀렸다.
- 원인은 확정하지 못했다. 소스상 서버(grpo_vllm_engine)는 응답을 RolloutOutput(response=ChatCompletionResponse(prompt_logprobs=...)) 로 감싸는데 trainer(utils.parse_prompt_logprobs)는 바깥 필드를 읽는다 — 이 가설로 응답 객체를 풀어 읽는 우회를 넣었으나 여전히 None 이었다. 서버가 요청의 prompt_logprobs 를 엔진까지 전달하지 않거나 응답 직렬화에서 빠지는 것으로 보인다. 확인하려면 서버에 직접 요청을 보내 응답 JSON 원문을 봐야 한다(pod 30분 이상).

## Tried and ruled out
- 응답 객체를 풀어 읽는 우회(opd_server_probe.py run46 의 parse 래퍼): 효과 없음.
- 조건 B(교사 템플릿 qwen3_5)는 서버가 기동하지 않아 측정하지 못했다. 서버 종료를 프로세스 그룹째 하도록 고쳤으나 B 기동은 여전히 실패했다.

## Watch out
- GKD 가 교사 신호 없이 조용히 학습된다 — 교사 서버 모드를 쓰는 모든 실행에서 첫 step 의 loss 가 0 이 아닌지, 교사 응답의 prompt_logprobs 가 비어 있지 않은지 반드시 확인할 것.
- kgpu/opd_server_probe46.sh 에서 setsid 로 띄우면 $! 의 생사를 kill -0 으로 볼 수 없다.

In the code:
- kgpu/opd_server_probe.py (run46·check46, parse 래퍼), kgpu/opd_server_probe46.sh
- 4.6.0: swift/rlhf_trainers/utils.py:840-866(parse_prompt_logprobs), rollout_mixin.py:279-290, infer_engine/grpo_vllm_engine.py:60-96
