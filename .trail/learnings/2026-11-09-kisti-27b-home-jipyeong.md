---
title: KISTI 컴퓨트 노드에서 27B 채점 서버를 띄울 때는 HOME 이 아니라 캐시·설정 폴더 변수를 옮겨야 한다
author: Jipyeong-lee
date: 2026-11-09
branch: main
id: l-91b2d469
scope: repo
status: active
src: agent
touches:
  - CLAUDE.md
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe.sh
  - kgpu/rubric_rollouts.sh
---

## Decided
- KISTI 1gpu 노드(A100 40GB)에서 Qwen3.8-27B-FP8 채점 서버는 뜬다. 세 번째 시도(84894)에서 서버 기동 570초, GPU 메모리 36.7GiB, 채점 시험에서 좋은 답 1.0점 대 나쁜 답 0점으로 통과했다.
- 컨테이너 안의 HOME 이 apptainer 설치 폴더(읽기 전용)로 잡혀서 `.config`·`.cache/vllm` 쓰기가 막혔다. 캐시·설정 폴더 변수(XDG_CACHE_HOME, XDG_CONFIG_HOME, FLASHINFER_WORKSPACE_BASE)를 /tmp 아래로 옮겨서 풀었다.

## Tried and ruled out
- `--env HOME=...`: apptainer가 `Overriding HOME environment variable with APPTAINERENV_HOME is not permitted` 경고를 내고 무시한다. 효과가 없고, XDG 변수만으로 충분했다.
- 로그인 노드에서 singularity 로 미리 시험하는 것: 로그인 노드에는 libsubid 라이브러리가 없어서 실행되지 않는다. 컴퓨트 노드에서만 시험할 수 있다.

## Watch out
- 서버가 뜨는 데 570초가 걸려 스모크 스크립트의 대기 한도(600초)에 거의 닿았다. 노드가 조금만 느리면 서버가 곧 떠도 실패로 판정된다. 한도를 900초쯤으로 늘리는 것이 안전하다.
- 이 시험은 채점 1건만 확인했다. 연속 채점의 속도와 안정성은 확인하지 못했다.

In the code: scripts/judge_server.sh(JH=/tmp/judge_home_$USER, --env XDG_*) · scripts/31_judge_smoke.slurm(ready 폴링 60회×10초) · 로그 logs/judge_smoke_84894.log
