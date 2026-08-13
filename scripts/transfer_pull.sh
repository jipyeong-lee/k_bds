#!/bin/bash
# =============================================================================
# transfer_pull.sh — 받는 계정(예: k252a02)에서 실행. 소스 계정(k252a01)의 데이터·모델을
#   pull 하고 절대경로를 자기 경로로 치환한다. (재다운로드/재빌드 회피 — 같은 클러스터 전용)
#
#   전제: 같은 KISTI 클러스터 + 같은 그룹(kbds0754). 소스 홈이 group-readable(확인됨).
#         소스는 push 불가(남의 홈 write 없음) → 받는 쪽이 pull 하는 구조.
#
#   사용 (받는 계정에서, 자기 kbds_project 루트에서):
#     SRC=/home01/<소스계정>/kbds_project bash scripts/transfer_pull.sh   # Stage-2 필수 subset
#     WITH_STAGE3=1 ...   + judge(27B) + medix (Stage-3 까지)
#     WITH_CKPT=1   ...   + ck-850 (구 실행 최고점 — E1 대체 교사 선택지 유지용)
#     WITH_LOGS=0   ...   학습 로그 제외(기본은 포함, ~142M)
#
#   ⚠️ **다른 클러스터로 나갈 때는 이 스크립트를 쓸 수 없다**(홈 직접 read 전제).
#      데이터전송노드 kbds-dm.kisti.re.kr (FTP 21 / Aspera 33001) 경유 → HANDOFF.md §2-A
#
#   ⚠️ 코드·문서는 이 스크립트로 옮기지 말 것. **git 이 정본**이다:
#        git clone https://github.com/jipyeong-lee/k_bds.git kbds_project
#      그 다음 여기서 work/ 만 채운다. 이 스크립트도 scripts/ 경로 치환을 하므로
#      clone 이 먼저 끝나 있어야 한다.
# =============================================================================
set -euo pipefail
SRC="${SRC:-/home01/k252a02/kbds_project}"
DST="${DST:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "[pull] SRC=$SRC"
echo "[pull] DST=$DST"
[ -r "$SRC/work/data/stage2_expanded_train.jsonl" ] || { echo "❌ 소스 읽기 불가 — 같은 그룹/권한 확인"; exit 1; }
mkdir -p "$DST/work/data" "$DST/work/checkpoints" "$DST/work/hf_cache/hub" "$DST/work/images"

echo "[pull] ① 데이터(jsonl+이미지) …"
rsync -a "$SRC/work/data/" "$DST/work/data/"
echo "[pull] ② v3 init(sft_mixed_merged) …"
rsync -a "$SRC/work/checkpoints/sft_mixed_merged" "$DST/work/checkpoints/"
echo "[pull] ③ base 모델(Qwen3.5-9B) …"
rsync -a "$SRC/work/hf_cache/hub/models--Qwen--Qwen3.5-9B" "$DST/work/hf_cache/hub/"
echo "[pull] ④ 컨테이너 sandbox …"
rsync -a "$SRC/work/images/" "$DST/work/images/"

if [ "${WITH_STAGE3:-0}" = 1 ]; then
  echo "[pull] ⑤ (Stage-3) judge(27B) + medix …"
  rsync -a "$SRC/work/hf_cache/hub/models--Qwen--Qwen3.6-27B-FP8" "$DST/work/hf_cache/hub/"
  rsync -a "$SRC/work/data/medix_rl_train.jsonl" "$DST/work/data/" 2>/dev/null || true
fi

if [ "${WITH_CKPT:-0}" = 1 ]; then
  #  ck-850 = 구 혼합 실행의 최고점(홀드아웃 51.52%, init 대비 +8.18pp p<0.0001).
  #  전문가 3분할에서 **E1(deepvision)을 빼고 이걸 일반 교사로 쓰는 선택**이 살아 있다
  #  → launch_domain_experts.sh 헤더. 버리면 그 선택지가 사라진다. 재생성 불가.
  echo "[pull] ⑦ (선택) 최고점 체크포인트 ck-850 …"
  rsync -a "$SRC/work/checkpoints/_mideval_snap_step850" "$DST/work/checkpoints/"
fi

if [ "${WITH_LOGS:-1}" = 1 ]; then
  #  logs/ 는 .gitignore 대상이라 git 으로 안 간다. 그런데 사후분석·붕괴 진단·감시자
  #  임계값이 전부 이 로그에서 나온 수치다. watch_format_collapse.py --simulate 도 이걸 먹는다.
  #  텍스트라 142M 밖에 안 된다 — 안 가져갈 이유가 없다.
  echo "[pull] ⑧ 학습 로그(재현·감시자 검증용, ~142M) …"
  mkdir -p "$DST/logs"
  rsync -a "$SRC/logs/" "$DST/logs/"
fi

echo "[pull] ⑨ 절대경로 치환(scripts + 모든 jsonl) …"
#  🚨 예전엔 work/data/*.jsonl 만 훑었다. domains/ 하위(도메인 3분할)가 통째로 빠져
#     이미지 경로가 소스 계정을 가리킨 채 남는다 → 학습이 이미지 못 찾고 죽는다.
#     그래서 -r 로 work/data 전체를 훑는다. logs 도 경로가 박혀 있어 같이 친다.
grep -rl "$SRC" "$DST/scripts/" "$DST/work/data/" 2>/dev/null | \
  xargs -r sed -i "s#$SRC#$DST#g"

echo "[pull] ✅ 완료. 검증(치환이 남았는지 = 0 이어야 한다):"
echo "      grep -rl '$SRC' work/data/ scripts/ | head"
echo "      head -1 work/data/domains/stage2_pmcvqa.jsonl | grep -o '\"images\": \\[[^]]*\\]'"
echo "      bash scripts/probe_1gpu.sh        # 유휴 debug-1gpu 에서 3 step 배선 확인"
