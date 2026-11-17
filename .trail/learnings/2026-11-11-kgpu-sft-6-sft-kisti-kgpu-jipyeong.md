---
title: kgpu 드라이브에서 SFT 병합 모델과 6차 병합 모델을 지웠다 — SFT 는 이제 KISTI 에만 있으니 kgpu 에서 SFT 를 읽는 작업 전에 다시 올린다
author: Jipyeong-lee
date: 2026-11-11
branch: main
id: l-f4aca935
scope: repo
status: active
src: agent
touches:
  - CLAUDE.md
  - configs/accuracy.py
  - kgpu/loco_chain_d.sh
  - kgpu/loco_dpo.sh
  - kgpu/merge_screen.sh
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe.sh
  - kgpu/rubric_rollouts.sh
  - scripts/21_rlvr_grpo_adv.slurm
---

## Decided
- 2026-11-11 kgpu ~/files/kbds 의 sft_mixed_merged(18.8GB)와 loco6_75_merged(19.3GB)를 지웠다. KISTI work/checkpoints 의 같은 이름 폴더와 파일별 sha256 이 29/29 일치함을 확인한 뒤다(kgpu 파일을 KISTI 로 흘려 해시한 값과 KISTI 사본의 해시를 비교). 사용량 87.9GB → 30.9GB(할당 100GB).
- 같은 날 중복 어댑터(adapter_v5·v6·v7 의 _partial, 최종 adapter_v6·adapter_v7)도 지웠다. 같은 step 의 사본(v6_100, v7_100, v6_75, v7_75)과 5차 최종(v5)은 남겼다.

## Tried and ruled out
- 이름과 크기만 맞춰 'KISTI 에 있다'고 판단하기: 같은 구조의 모델과 어댑터는 파일 크기가 같아 서로 다른 모델이 일치로 잡힌다(v2~v7 어댑터가 전부 v6_75 와 일치로 나왔다). 해시로만 믿었다.
- 데이터 tar(holdout·global_bench·mmk12_data)와 7차·옛 어댑터: KISTI 에 같은 파일이 없어 지우지 않았다.

## Watch out
- SFT 모델은 이제 KISTI 에만 있다. HF 에는 ta_w07 하나뿐이다. KISTI 는 과제가 끝나면 1주 뒤 데이터가 삭제되니, 끝나기 전에 HF 비공개 저장소에 올리는 편이 안전하다.
- kgpu 스크립트 8개가 sft_mixed_merged 를 읽는다(global_bench, loco_eval_sft, loco_eval_med_sft, merge_screen 의 sft·전문가 병합, mmk12_probe, opd_server_probe, opd_smoke, timing_bench). 돌리기 전에 KISTI 에서 다시 올린다(KISTI → kgpu 직접 PUT 약 37MB/s, 18.8GB 에 약 9분). dv_loco_smoke.sh 는 loco6_75_merged 를 어댑터로 다시 만들어 쓰므로 그대로 돈다.
- pilot·restricted·g1~g18·data·code(3.9GB)는 다른 프로젝트 파일이라 그대로 두었다.

In the code:
- 지운 경로: ~/files/kbds/{sft_mixed_merged, loco6_75_merged, loco/adapter_v5_partial, loco/adapter_v6_partial, loco/adapter_v7_partial, loco/adapter_v7, loco/adapter_v6}
