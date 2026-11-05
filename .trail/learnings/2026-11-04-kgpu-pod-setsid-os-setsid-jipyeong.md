---
title: "로컬 맥에서 kgpu pod 실행기를 분리해 띄울 때는 setsid 대신 파이썬 os.setsid 를 쓴다"
author: Jipyeong-lee
date: 2026-11-04
branch: main
id: l-ec1afb62
scope: repo
status: active
src: agent
touches:
  - scripts/merge_experts.py
  - scripts/test_merge_experts.py
---

## Decided
로컬(macOS)에서 kgpu 실행기를 세션과 분리해 띄울 때 파이썬 `os.setsid()` 후 `bash` 를 exec 하는 방식을 쓴다.

## Tried and ruled out
- `setsid nohup bash ...`: macOS 에 setsid 명령이 없어 "command not found" 로 pod 를 만들기도 전에 끝났다.

## Watch out
- 분리하지 않으면 에이전트 세션이 끝날 때 trap 이 pod 를 반납한다(CLAUDE.md §2.5).
- 토큰(HF_TOKEN)은 env 로만 넘기고 파일·커밋에 남기지 않는다.

In the code:
- `export HF_TOKEN=... && nohup python3 -c 'import os,sys;os.setsid();os.execvp("bash",["bash"]+sys.argv[1:])' kgpu/run_pod.sh diag_vis.sh HF_TOKEN > logs/kgpu_diag_vis.out 2>&1 < /dev/null & disown`
