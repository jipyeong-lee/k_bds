---
title: kgpu 저장소의 큰 중복 파일은 로컬로 받지 말고 싼 pod 안에서 바이트 비교로 확인한 뒤 지운다
author: Jipyeong-lee
date: 2026-10-28
branch: main
cost: { minutes: 30, tokens: 96539 }
id: l-39fb531d
scope: repo
status: active
src: agent
touches:
  - kgpu/stage3_smoke.sh
  - kgpu/run_pod.sh
---

## Decided
- kgpu ~/files 의 큰 파일이 중복인지는 로컬로 내려받아 해시하지 않고, 싼 pod(rtx3090, 600 크레딧/시간) 안에서 cmp(바이트 비교)로 확인한다. 40GB 비교가 약 8분, 약 150 크레딧이었다.
- 지운 뒤에는 DELETE 응답만 믿지 말고 같은 경로를 다시 조회해 404 가 나오는지, 전체 파일 수·크기가 줄었는지, 다른 파일이 안 바뀌었는지 확인한다.
- 2026-10-28 에 mmk12_probe/ 폴더(20.68GB)와 kbds/dr_grpo_merged(18.84GB)를 지웠다. 사용량 69GB 에서 29.5GB 로 줄었다. 앞의 것은 kbds/ 에 같은 파일이 있고, 뒤의 것은 KISTI 에 사본이 있다.

In the code:
- 비교는 kgpu/run_pod.sh 로 pod 를 빌려 ~/files 를 직접 읽는 일회용 스크립트로 했다(예: GPU_MODEL=rtx3090 MEM=16Gi CPU=4 DISK=30 bash kgpu/run_pod.sh <script>). 삭제는 DELETE /v1/files/<경로>?recursive=true.
- KISTI 사본의 온전성은 로그인 노드에서 safetensors 헤더 길이와 파일 크기, index 의 total_size 합계로 확인했다(18GB 를 읽지 않는다).

## Tried and ruled out
- 로컬(WSL)에서 스트리밍 해시: 처리량이 2.4 MB/s 까지 떨어져 12GB 에 한 시간 넘게 걸렸다. pod 안에서는 110 MB/s 였다.
- 크기가 같으면 같은 파일이라는 가정: mmk12_data.tar 와 mmk12_images.tar 는 크기가 달랐지만 앞의 것이 뒤의 것을 통째로 포함했다. 크기가 같은 tar 3개는 실제로 같았다. 크기만으로는 중복 여부를 가릴 수 없다.

## Watch out
- ~/files 에는 이 저장소와 무관한 다른 프로젝트 파일(pilot/, restricted/, g1~g18, data/eicu, data/kmimic)이 같이 있다. 접근 제한 데이터에서 만든 파일일 수 있어 주인이 판단하기 전에는 건드리지 않는다.
- ckpt_mm.tar, ckpt_mm350.tar, mmk12_ckpt350.tar 는 내용이 같지만 스크립트마다 다른 이름을 쓴다. 하나만 남기려면 스크립트도 같이 고쳐야 한다.
- scripts/launch_stage3.sh(KISTI)는 아직 dr_grpo_merged 를 Stage-3 초기 모델로 쓴다. KISTI 사본은 남아 있다. 통합 모델(merged_ta_w07)로 바꿔야 한다.
