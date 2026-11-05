---
title: ms-swift 4.6.0 은 교사 서버에 학생 답을 토큰 id 그대로 보낸다 — 대신 교사 시퀀스가 학생 시퀀스와 앞부분까지 토큰 단위로 같아야 한다
author: Jipyeong-lee
date: 2026-11-05
branch: main
id: l-d0c5a554
scope: repo
status: active
src: agent
---

## Decided
- 4.6.0 소스(휠, 2026-11-05 확인): 롤아웃 샘플로 교사 요청을 만들고(gkd_helpers.build_teacher_requests), 학생이 샘플링한 response_token_ids 를 그대로 assistant 자리에 넣어 보낸다(replace_assistant_response_with_ids). 4.1.3 의 '교사가 학생 답을 못 받는' 문제(learning l-0cd61c58)는 4.6.0 에서 해소된다.
- 대신 위치 맞춤이 '끝에서 n 개'가 아니라 '앞에서부터'다(utils.assemble_teacher_topk_logprobs: 비패킹은 attention_mask 첫 위치부터, 패킹은 학생 쪽 cu_seqlens 경계). 교사 서버가 렌더한 프롬프트가 학생 인코딩과 한 토큰이라도 다르면 전체가 밀린다. 패킹 모드만 경고를 내고 비패킹은 조용히 진행한다.
- 따라서 27B 교사(Qwen3.8)를 쓰려면: vllm serve --chat-template <학생 chat_template.jinja>(Qwen3.8 기본 템플릿은 'Reasoning effort xhigh' 시스템 문장을 넣는다) + 이미지 토큰 수가 학생(Qwen3.5) 처리와 같은지 확인이 필요하다. 판정 기준은 교사 prompt_token_ids == 학생 input_ids 전체 일치.
- 의존성 핀: transformers<5.20 · trl<1.0 · peft<0.21 · datasets>=4.0,<4.8.5 — 현재 스택 중 datasets==3.6.0 은 범위 밖이라 올려야 한다. gkd_domain_teacher.patch 는 4.6.0 에 적용되지 않는다(트레이너 재작성).

In the code:
- 4.6.0: swift/rlhf_trainers/gkd_helpers.py:83-123(요청), :126-190(조립) · utils.py:869-907(오프셋)
