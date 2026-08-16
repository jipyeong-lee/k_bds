#!/usr/bin/env bash
# pull_file.sh — $ORCH_HOME 의 파일을 GPU 세션 없이 그대로 내려받는다.
#
#   ORCH_PAT=… ORCH_BASE_URL=… bash b200/pull_file.sh train_deepvision_ep1_gdpo_async_tis.log /tmp/train.log
#
# 왜 필요한가: 학습이 GPU 8 장을 다 잡고 있으면 세션이 안 열려서(`session create failed`)
# 노드 안에서 도는 dump_metrics.sh·check_progress.sh 를 job 교체 틈에서만 쓸 수 있었다.
# `/me/data/file` 은 세션과 무관한 파일 API 라 학습 중에도 언제든 읽힌다.
# 덤으로 exec stdout 8KB 절단도 없어서 로그를 통째로 받아 전 step 을 파싱할 수 있다.
# (`/me/data?path=<dir>` 는 목록, `/me/data/file?path=<file>` 이 내용. download·cat 은 404.)
set -uo pipefail
: "${ORCH_BASE_URL:?ORCH_BASE_URL not set}"
: "${ORCH_PAT:?ORCH_PAT not set}"
REL="${1:?usage: pull_file.sh <ORCH_HOME 기준 상대경로> [저장경로]}"
OUT="${2:-$(basename "$REL")}"

curl -sk --max-time 300 "$ORCH_BASE_URL/me/data/file?path=$REL" \
     -H "Authorization: Bearer $ORCH_PAT" \
     -o "$OUT" -w 'http %{http_code}  %{size_download}B  %{time_total}s\n'
[ -s "$OUT" ] || { echo "❌ 빈 파일 — 경로를 확인할 것: $REL"; exit 1; }
ls -la "$OUT"
