---
title: KISTI 의 저장소 사본은 git pull 로 갱신할 수 없다 — 파일을 비교해 scp 로 올린다
author: Jipyeong-lee
date: 2026-11-05
branch: main
id: l-8f46d077
scope: repo
status: active
src: agent
touches:
  - configs/medical_reward.py
  - scripts/transfer_pull.sh
---

## Decided
- KISTI(k266a01) ~/kbds_project 는 git 체크아웃이지만 HEAD 가 2026-08-14(63ebeff)에 멈춰 있고 GitHub 키가 없어 fetch 가 Permission denied 다. 그동안 스크립트는 파일 단위로 올려 왔고, 수정 70개가 git 에 '변경됨'으로 보인다. "push 후 pull" 은 되지 않는다.
- 올리는 절차: 원격 파일이 '로컬 변경 직전 커밋 판'과 바이트 단위로 같은지 cmp 로 확인 → 같으면 원격을 *.pre<태그> 로 백업하고 scp -p → md5 로 확인. 다르면 원격 고유 수정이 있다는 뜻이라 올리지 않는다.
- 2026-11-05: 3.8 교체 스크립트 9개와 configs/medical_reward.py 를 이 절차로 올렸다(백업 *.pre38). scripts/transfer_pull.sh 는 원격 고유 수정이 있어 보류.

## Watch out
- configs/medical_reward.py 원격 판은 어떤 커밋과도 일치하지 않았다(기본 judge gemma4-26b, c2 인용 지시가 옛 순서). 차이가 모두 로컬 최신판에 흡수된 옛 내용이라 덮어썼다.
- md5 를 grep 으로 줄 맞춰 비교하다 오판한 적이 있다 — cmp 로 직접 비교할 것. zsh 에서는 "$c:path" 가 수식어로 먹히니 "${c}:path" 로 쓴다.

In the code:
- 확인: git show <커밋>^:<파일> | cmp -s - <원격에서 scp 로 받은 사본>
