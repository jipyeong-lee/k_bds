#!/usr/bin/env bash
# drive_node.sh — run a LOCAL script ON the Jukyung-Yadok GPU node via the orchestrator API.
# create session -> exec (script via stdin: `bash -s`) -> poll -> print stdout/stderr -> release.
# Usage:  ORCH_BASE_URL=https://... ORCH_PAT=pat_xxx bash drive_node.sh <payload.sh> [gpu_count] [timeout_sec]
set -u
BASE="${ORCH_BASE_URL:?set ORCH_BASE_URL to the platform API base (do not hardcode/commit it)}"
PAT="${ORCH_PAT:?set ORCH_PAT to your platform PAT (do not hardcode/commit it)}"
NODE="${ORCH_NODE:-2dfaecc46f5240f084d44b7614560fd1}"   # gpu-node-1
AUTH="Authorization: Bearer $PAT"
PAYLOAD="${1:?usage: ORCH_PAT=... bash drive_node.sh <payload.sh> [gpu_count] [timeout_sec]}"
GPU="${2:-1}"; TMO="${3:-1800}"
CURL="curl -sk --max-time 60"
[ -f "$PAYLOAD" ] || { echo "❌ payload not found: $PAYLOAD"; exit 1; }

pyget(){ python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: print(''); sys.exit(0)
for p in '$1'.split('.'):
    if isinstance(d,dict) and p in d: d=d[p]
    else: d=''; break
print('' if d is None else d)"; }

echo "[drive] open session on node=$NODE gpu_count=$GPU"
SID=$($CURL -X POST "$BASE/nodes/$NODE/sessions" -H "$AUTH" -H "Content-Type: application/json" -d "{\"gpu_count\":$GPU}" | pyget id)
[ -n "$SID" ] || { echo "[drive] ❌ session create failed"; exit 1; }
echo "[drive] session=$SID"
REQ=$(python3 -c "import json;print(json.dumps({'command':'bash -s','stdin':open('$PAYLOAD').read(),'timeout_sec':$TMO}))")
JOB=$(printf '%s' "$REQ" | $CURL -X POST "$BASE/sessions/$SID/exec" -H "$AUTH" -H "Content-Type: application/json" -d @- | pyget job_id)
[ -n "$JOB" ] || { echo "[drive] ❌ exec failed"; $CURL -X DELETE "$BASE/sessions/$SID" -H "$AUTH" >/dev/null; exit 1; }
echo "[drive] job=$JOB polling…"
START=$(date +%s); ST=""
while true; do
  RESP=$($CURL "$BASE/jobs/$JOB" -H "$AUTH"); ST=$(printf '%s' "$RESP" | pyget job.state)
  case "$ST" in succeeded|failed|killed) break ;; esac
  [ $(( $(date +%s) - START )) -gt $((TMO+120)) ] && { echo "[drive] ⏱ poll timeout"; break; }
  sleep 5
done
echo "[drive] state=$ST exit_code=$(printf '%s' "$RESP" | pyget job.exit_code)"
echo "----- stdout -----"; printf '%s' "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('stdout_tail',''))"
echo "----- stderr -----"; printf '%s' "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('stderr_tail',''))"
$CURL -X DELETE "$BASE/sessions/$SID" -H "$AUTH" >/dev/null && echo "[drive] session released ✓"
