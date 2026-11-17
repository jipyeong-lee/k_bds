---
title: "풀이 줄이기 의료 평가 채점은 답이 보기 글자와 정확히 같아야 해서, 내용을 붙여 쓰는 모델은 거의 전부 오답이 된다"
author: Jipyeong-lee
date: 2026-11-10
branch: main
id: l-4d35eb48
scope: repo
status: active
src: agent
touches:
  - CLAUDE.md
  - kgpu/loco_chain_d.sh
  - kgpu/loco_dpo.sh
  - kgpu/merge_screen.sh
  - kgpu/opd_server_probe.py
  - kgpu/opd_server_probe.sh
  - kgpu/rubric_rollouts.sh
---

## Decided
- 의료 평가 점수는 `<answer>` 안의 보기 글자만 뽑아 정답과 비교해 다시 채점한다. SFT 모델(sft_mixed_merged)은 `<answer>(B) sodium atom ...</answer>` 처럼 쓰는 버릇이 있어 원래 채점으로는 5.5%, 다시 채점하면 47.2% 였다. 다른 모델은 0~0.5%p 만 바뀌었다(원본 52.0→52.5, 4차 60.5→61.0, 5차 그대로).

## Tried and ruled out
- 평가를 다시 돌리는 것: 답은 이미 저장돼 있어 필요 없다. 정답만 있으면 로컬에서 다시 채점된다.

## Watch out
- 평가 스크립트 채점기 `ok()` 는 고치지 않았다. 다음 의료 평가도 같은 문제가 생기니 원본 `correct` 값을 그대로 믿지 말 것.
- 수학 채점도 단위·설명을 붙인 답(`8 meters`, `\(\dfrac{16}{3}cm^3\)`)을 오답으로 센다. SFT 모델 수학 오답 202개 중 33개가 이런 형태라 32.7% 는 낮게 잡혔을 수 있다. 다시 채점하지 않았다.
- 정답 파일은 kgpu `global_bench.tar`(873MB)에 있다. bsdtar 로 `curl ... | tar -xqf - global_bench.jsonl` 하면 그 파일만 받고 멈춘다. 문항 id 는 `<_source>-<줄 번호>`.

In the code:
- `kgpu/loco_eval_med_sft.sh`·`kgpu/loco_eval_med5.sh` 의 `ok()`(정확 일치 → math_verify)
- `dashboard/build_loco_eval.py` 의 `letter()`·`regrade()`, 정답 `work/loco/global_bench.jsonl`(git 제외)
