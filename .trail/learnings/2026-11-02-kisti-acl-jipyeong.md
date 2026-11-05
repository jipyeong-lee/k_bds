---
title: "그룹이 다른 KISTI 계정 사이 이관은 소스 쪽에서 받는 계정에 읽기 ACL 을 걸어 직접 당긴다"
author: Jipyeong-lee
date: 2026-11-02
branch: main
id: l-58cf7c27
scope: repo
status: active
src: agent
touches:
  - scripts/36_opd_domain.slurm
---

## Decided
- 266(k266a01, 그룹 kbds0767) → 301(k301a01, 그룹 kbds0810) 이관(2026-11-02). 그룹이 달라 HANDOFF §2-A 의 "같은 그룹이면 pull 가능" 전제가 깨진다(301 에서 266 홈 Permission denied).
- 소스(266)에서 받는 계정 한 명에게만 Lustre ACL 로 읽기를 준다. 소규모 시험으로 동작 확인했다(chmod 600 파일 + setfacl u:k301a01:r → 301 이 읽음).
- 범위: Stage-2 필수분 + WITH_STAGE3(judge 27B) + WITH_CKPT(ck-850, 재생성 불가) ≈ 157G.
- 코드는 GitHub clone 이 정본(301 에 clone 완료, 78260ba).

## Tried and ruled out
- /scratch/migrate_k266_to_gpu (777) 경유: 복사 두 번 + 이관 동안 클러스터 전체에 데이터 노출 → 기각.
- 로그인 노드에서 직접 setfacl -R / rsync: 15분 워치독 위험 → 둘 다 cpu64-only CPU 잡으로.

## Watch out
- ACL 은 홈·kbds_project·work·hf_cache·hub·checkpoints 에 x 만, 대상 폴더에는 -R rX. 이관 후 회수할 것.
- 266 저장소는 옛 커밋 63ebeff + 미커밋 수정 다수 — 코드는 266 에서 가져오지 말 것.

In the code:
- 266: ~/acl_grant_301.sh (job 84065, 로그 ~/acl_grant_301.log). 회수: setfacl -R -x u:k301a01 <대상> 및 상위 디렉터리 setfacl -x u:k301a01
- 301: cd ~/kbds_project && SRC=/home01/k266a01/kbds_project WITH_STAGE3=1 WITH_CKPT=1 bash scripts/transfer_pull.sh (CPU 잡으로)
- 크기: data 47G · sft_mixed_merged 18G · Qwen3.5-9B 19G · images 42G · Qwen3.6-27B-FP8 30G · ck-850 0.5G · logs 182M (work 전체 474G)
