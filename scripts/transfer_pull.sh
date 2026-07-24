#!/bin/bash
# =============================================================================
# transfer_pull.sh — 받는 계정(예: k252a02)에서 실행. 소스 계정(k252a01)의 데이터·모델을
#   pull 하고 절대경로를 자기 경로로 치환한다. (재다운로드/재빌드 회피 — 같은 클러스터 전용)
#
#   전제: 같은 KISTI 클러스터 + 같은 그룹(kbds0754). 소스 홈이 group-readable(확인됨).
#         소스는 push 불가(남의 홈 write 없음) → 받는 쪽이 pull 하는 구조.
#
#   사용 (k252a02 계정에서, 자기 kbds_project 루트에서):
#     bash scripts/transfer_pull.sh                 # Stage-2 필수 subset
#     WITH_STAGE3=1 bash scripts/transfer_pull.sh    # + judge(27B) + medix (Stage-3 까지)
# =============================================================================
set -euo pipefail
SRC="${SRC:-/home01/k252a01/kbds_project}"
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

echo "[pull] ⑥ 절대경로 치환(scripts + jsonl 이미지경로) …"
grep -rl "$SRC" "$DST/scripts/" "$DST"/work/data/*.jsonl 2>/dev/null | \
  xargs -r sed -i "s#$SRC#$DST#g"

echo "[pull] ✅ 완료. 검증:"
echo "      head -1 work/data/stage2_expanded_train.jsonl | grep -o '\"images\": \\[[^]]*\\]'"
echo "      SMOKE=1 bash scripts/launch_stage2_expanded.sh"
