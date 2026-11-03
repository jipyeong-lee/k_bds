---
title: 27B 모델(교사 후보·Stage-3 judge)은 Qwen3.6 대신 Qwen3.8-27B-FP8 로 통일한다
author: Jipyeong-lee
date: 2026-11-03
branch: main
id: l-b8ffae4a
scope: repo
status: active
src: agent
touches:
  - scripts/30_medical_rl.slurm
  - scripts/31_judge_smoke.slurm
  - scripts/33_judge_probe.slurm
  - scripts/34_rubric_compare.slurm
  - scripts/35_stage3_smoke.slurm
  - scripts/45_healthbench_smoke.slurm
  - scripts/judge_server.sh
  - scripts/judge_server.slurm
  - scripts/merge_experts.py
  - scripts/run_healthbench.py
---

## Decided
- 2026-11-03: 27B 급 모델은 Qwen3.8-27B-FP8 로 통일한다. 3.6 은 코드 참조와 가중치(266·301 각 31GB)를 모두 없앴다. 사용자가 "더 최신 모델을 쓰겠다"고 정했다.
- Qwen3.8-27B 는 학생(sft_mixed_merged)과 어휘가 완전히 같고(248,044개, 토큰-id 매핑까지 비교) 비전을 포함해, OPD 교사 후보로 구조상 문제가 없다. 점수는 아직 모른다.

## Tried and ruled out
- 로그인 노드(A100 40GB)에 27B 를 띄워 평가: /etc/motd 가 로그인 노드의 분석 작업을 명시적으로 제한한다("예고 없이 프로세스 제한"). 하지 않았다.
- 계산 노드 → 로그인 노드 → kgpu API 중계: 계산 노드는 DNS 와 외부 IP 가 모두 막혀 있다(job 84235 실측). 의도된 격리를 우회하는 일이라 하지 않았다.
- kgpu API(qwen3.8-27b)로 holdout 평가: 이날 백엔드가 세 번 502 로 죽어 받은 답이 0건이었다. 재시도·캐시를 넣은 스크립트는 만들어 두었다.
- 코드만 교체하고 가중치는 보존: 선택하지 않았다.

## Watch out
- 3.8 을 judge 로 쓴 검증은 없다. 3.6 으로 확인한 judge 유닛 테스트 29/29 와 루브릭 점수는 다시 확인해야 한다. run_healthbench.py 의 "thinking 모델이라 content 가 비면 파싱 실패" 우회가 3.8 에도 맞는지도 모른다.
- 스크립트 수정은 로컬에만 있다(미커밋). 266·301 의 스크립트는 여전히 3.6 경로를 가리키니, 푸시 후 pull 하기 전에는 judge 스크립트를 돌리지 않는다. 지금 judge 는 KHDP deepseek 라서 영향은 없다.
- huggingface_hub 2.x 는 blob 을 공유 hub/blobs/<2자리>/ 아래 두고 모델 폴더에는 링크만 둔다. sha256 검증은 find -L 로 링크를 따라가야 하며, 따라가지 않으면 파일 0개로 "통과"처럼 보인다.
- 301 로그인 노드 python 에는 huggingface_hub 가 없어 pip install --user 로 깔았다.

In the code:
- 교체한 스크립트: scripts/judge_server.sh, judge_server.slurm, 30_medical_rl.slurm, 31·33·34·35·45 slurm, run_healthbench.py, transfer_pull.sh. 서빙 이름 qwen36-judge → qwen38-judge.
- 가중치: work/hf_cache/hub/models--Qwen--Qwen3.8-27B-FP8 (snapshot 017b9c7…, 266 은 job 84239, 301 은 job 84277 로 sha256 67개 일치).
- 평가: 266 job 84238(holdout 1,772문항, 8gpu 1장, 결과 logs/eval_final_q38_27b_hold.jsonl). 비교 기준은 ta_w07 의 logs/eval_items_init_83037.jsonl, scripts/eval_paired.py 로 검정.
