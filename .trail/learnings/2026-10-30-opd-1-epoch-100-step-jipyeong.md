---
title: OPD 본실행은 전체 1 epoch 이 아니라 100 step 으로 효과부터 확인한다
author: Jipyeong-lee
date: 2026-10-30
branch: main
id: l-4d65ed8f
scope: repo
status: active
src: agent
touches:
  - scripts/36_opd_domain.slurm
---

## Decided
- OPD 도메인 증류의 다음 본실행은 100 step 이다. 도메인당 약 133행, 합계 400행을 쓰고 step 50·100 에서 어댑터를 저장한다. 평가에서 ta_w07 과의 차이가 표준오차(약 1.5pp)보다 분명히 클 때만 규모를 키운다.
- 20 step 스모크를 다시 돌렸다. 데이터 섞기를 끄면 교사 호출 카운터가 균등해진다. 4 step 시점에 5/6/5 였고, 섞기가 켜진 이전 실행(83739)은 28/19/33 이었다.

In the code:
- scripts/36_opd_domain.slurm 의 swift rlhf gkd 에 --dataset_shuffle false --train_dataloader_shuffle false 를 넣었다. 4.1.3 에서 두 인자 모두 오류 없이 받아들여진다.
- 제출 예정: MAX_STEPS=100 SMOKE_ROWS=400 SAVE_STEPS=50 sbatch --time=04:00:00 scripts/36_opd_domain.slurm
- KISTI 측정: 78.9 s/step. job 83821(스모크 재실행).
- 설정 문서: docs/2026-2차/setup_opd_domain.html

## Tried and ruled out
- 전체 1 epoch(75k행, 약 18,750 step): 약 410시간, 약 3,300 GPU·시간이 든다. HANDOFF 의 잔여 예산(1,996, 오래된 값)을 넘는다. job 은 2시간 제한이고 저장·재개 체인이 없어 진행분이 사라진다.
- 도메인당 2~4천 행(1,500~3,000 step): 2~3일 걸린다. 효과를 모르는 상태에서는 이르다고 보고 뒤로 미뤘다.

## Watch out
- 8gpu 노드는 배타 점유인데 GPU 는 1장만 쓴다. 시간당 8 노드시간이 든다.
- 최대 메모리 여유가 약 8 GiB 뿐이다. 긴 응답이 나오면 VLLM_UTIL 이나 MAX_COMP 를 낮춰야 한다.
- 섞기를 끄면 앞쪽 행만 쓰므로 도메인 안의 데이터 다양성이 좁아진다. 본실행이 커지면 섞기 설정을 다시 정해야 한다.
