---
title: 27B 교사 OPD 는 deepvision·mmk12 만 27B 로 하고 의료는 제외한다 — 서버 방식(멀티 교사 라우팅)으로 재설계
author: Jipyeong-lee
date: 2026-11-04
branch: main
id: l-0dd1ad14
scope: repo
status: active
src: agent
touches:
  - scripts/merge_experts.py
  - scripts/test_merge_experts.py
---

## Decided
- Qwen3.8-27B-FP8 은 학생(ta_w07)보다 강하다: holdout 0.733 vs 0.633(+9.99pp), deepvision +16.15pp, mmk12 +3.75pp(p=0.049). gbench 는 전체 0.604 vs 0.562, medxpertqa_mm +9.6pp.
- 그러나 pmcvqa +1.25pp 는 유의하지 않고, mmmu_hm 은 −2.16pp(p=0.039)로 학생보다 낮다. 의료 문항에 이 교사를 쓸 근거가 없다 → OPD 의 27B 교사는 deepvision·mmk12 에만 쓴다.
- A100 에서 27B 를 로컬 교사로 올리지 않고, 같은 노드의 다른 GPU 에 vLLM 서버로 띄워 GKD 의 teacher_model_server(top-k prompt_logprobs)로 쓴다.
- 지난 OPD(교사가 학생보다 약함)는 답이 22~41% 짧아졌다. 학습 MAX_COMP 3072 vs 평가 8192 불일치가 의심되므로 MAX_COMP 를 4096 이상으로 올린다.

## Tried and ruled out
- 27B 를 학생 프로세스에 로컬 교사로: bf16 54GB 가 A100 80GB 에 학생과 같이 안 들어가고, transformers 의 FP8 로더는 compute capability 8.9 이상만 지원해 A100(8.0)에서 못 쓴다.
- ms-swift 4.5.0 업그레이드: GRPO mismatch 인자가 4.1.3 과 같아 이득 없음(기존 결론). 4.6.0 은 GKD 쪽에 이득이 있다(아래).

## Watch out
- gbench(global_bench.jsonl 3,712건)가 공식 셋 전체인지 확인하지 못했다: mmmu_hm 1,712건이 기억하는 공식 H&M 문항 수보다 적고, MMMU 테스트 정답은 비공개라 원본이 불명확하다. 모델끼리 비교에만 쓰고 공식 리더보드와 직접 비교하지 말 것.
- gbench 형식 준수율 0.796, 답이 4096 토큰에서 잘렸을 가능성(약 20%)이 있어 27B 점수는 하한일 수 있다.
- 서버 모드는 top-k 밖 교사 확률이 빠진다(reverse KL 이 가장 벌주는 토큰). 학생 샘플 토큰의 top-k 포함률과 Qwen3.8 채팅 템플릿(thinking) 정렬은 학습 전에 재야 한다. 미검증.
- ms-swift 4.6.0: teacher_model_server 에 [{url, tags}] 멀티 교사 라우팅과 teacher_adapters 가 추가돼 gkd_domain_teacher.patch 를 대체할 수 있다. GKD 트레이너가 재작성돼(1071→546줄) 패치는 4.6.0 에 적용되지 않는다. 요구 transformers·vLLM 버전과 get_tag 가 읽는 필드는 미확인, 설치도 하지 않았다.

In the code:
- scripts/36_opd_domain.slurm, kgpu/gkd_domain_teacher.patch (4.1.3 기준)
- sandbox 의 swift/rlhf_trainers/gkd_trainer.py:646 fetch_teacher_logprobs (서버 모드, chat completions + prompt_logprobs)
- 4.6.0: swift/rlhf_trainers/gkd_helpers.py:276-352 (멀티 교사 태그 라우팅)
- 비교 결과: logs/eval_items_init_84238.jsonl(holdout), logs/eval_items_gbench_sft_84316.jsonl(gbench), scripts/eval_paired.py
