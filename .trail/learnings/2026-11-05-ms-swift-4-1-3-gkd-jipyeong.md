---
title: ms-swift 4.1.3 GKD 교사 서버 모드는 멀티모달 온폴리시에서 교사에게 학생 답을 보내지 않는다 — 패치 없이 27B 서버 교사 OPD 를 돌리면 안 된다
author: Jipyeong-lee
date: 2026-11-05
branch: main
id: l-0cd61c58
scope: repo
status: active
src: agent
touches:
  - kgpu/opd_server_probe.sh
---

## Decided
- 2026-11-05 kgpu rtxpro6000 1장 실측(kgpu/opd_server_probe.sh, 조건별 GKD 2 step): 교사 서버가 받은 messages 는 [system, user] 뿐이었다(4/4, 두 조건 모두). 학생이 생성한 답이 없다.
- 그 결과 ms-swift 는 교사가 '질문 끝부분'에 매긴 logprob 를 학생 응답 위치에 붙인다(gkd_trainer.py:663-670 꼬리 맞춤). 꼬리 토큰 일치 0/4, 학생 토큰의 교사 top-20 포함률 9~11% — 사실상 무작위 신호인데 학습은 오류 없이 돈다(SWIFT_EXIT=0).
- 원인: vLLM 롤아웃 경로는 생성 결과를 generated_inputs 에 담지만, 교사 API 호출은 raw_inputs=inputs(데이터셋 원본)를 넘긴다(gkd_trainer.py:601-602). OPSD(teacher_prompt 열) 경로만 학생 답을 붙인다(:522-530).
- 따라서 27B 서버 교사 OPD 는 (a) _fetch_teacher_logprobs_from_api 에 generated_inputs 를 넘기도록 패치하거나 (b) 4.6.0 에서 고쳐졌는지 확인한 뒤에만 돌린다. 패치 후에도 교사 템플릿을 학생 템플릿으로 맞춰야 한다(Qwen3.8 기본 템플릿은 빈 사고 블록을 끼워 답 부분이 30/30 다르다 — vllm serve --chat-template 로 해결, 로컬 렌더링 30/30 일치).

## Tried and ruled out
- 공용 kgpu API(qwen3.8-27b)로 top-k 측정: 도중에 prompt_logprobs 를 무시하는 백엔드로 바뀌었다. 2문항(1,652 토큰)만 측정 — 템플릿을 맞추면 top-20 포함 99.15%(temp 0 학생 답이라 상한값).
- GPU 2장 pod: kgpu rtxpro6000 은 전체 1장이라 영원히 Pending(대기 중에도 과금, 560 크레딧).

## Watch out
- torchrun 없이 swift 를 같은 프로세스에서 돌리면 vLLM colocate 가 KeyError 'RANK' → RANK/WORLD_SIZE 등 단일 rank env 를 직접 준다.
- 한 GPU 에 교사 서버 0.42 + 학생 vLLM 0.22 는 학생 KV 자리가 없어 실패, 0.40 + 0.30 은 통과(MAX_COMP 1024).
- pod 안에서 pkill -f '<스크립트명>' 은 exec 셸 자신도 죽인다 — PID 로 죽일 것.

In the code:
- kgpu/opd_server_probe.{py,sh} — 결과 ~/files/kbds/logs/opd_probe_{A_default,B_student_tpl}/
