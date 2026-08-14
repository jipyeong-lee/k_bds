#!/bin/bash
# upload_chunked.sh — push tarballs from KISTI to the GPU platform via /me/data/upload.
# The server cuts single upload requests at ~60s → split big tars into ≤3.5G parts.
# Node reassembles with: cat parts/<name>.tar.*.part | tar xf -
# Usage (run ON KISTI):  ORCH_BASE_URL=https://... ORCH_PAT=pat_xxx bash upload_chunked.sh <tardir>
set -u
BASE="${ORCH_BASE_URL:?set ORCH_BASE_URL to the platform API base (do not hardcode/commit it)}"
PAT="${ORCH_PAT:?set ORCH_PAT to your platform PAT}"
AUTH="Authorization: Bearer $PAT"
R="${1:-/scratch/migrate_k266_to_gpu}"
CHUNK="${CHUNK:-3500M}"
# Which tars to push. Narrow it on follow-up runs so already-uploaded tars aren't re-sent:
#   TARS="data.tar" bash upload_chunked.sh <tardir>
TARS="${TARS:-model.tar data.tar pmcvqa_data.tar}"
CODE="${CODE:-1}"                  # CODE=0 to skip re-uploading code.tar.gz

up(){ local f="$1" dest="$2" n code; n=$(basename "$f")
  code=$(curl -sk --max-time 90 -X POST "$BASE/me/data/upload?path=$dest" -H "$AUTH" -H "Expect:" \
         -F "file=@$f" -o /tmp/_upresp -w '%{http_code}')
  echo "  $n -> $dest : http $code $(head -c 120 /tmp/_upresp)"; [ "$code" = "200" ]; }

mkdir -p "$R/parts"
echo "== split large tars ($CHUNK) — TARS='$TARS' =="
for t in $TARS; do
  [ -f "$R/$t" ] || { echo "  skip $t (not found)"; continue; }
  rm -f "$R/parts/$t".*.part          # only this tar's parts, so other tars stay uploadable
  split -b "$CHUNK" -d --additional-suffix=.part "$R/$t" "$R/parts/${t}."
done
ls -lh "$R"/parts/*.part 2>/dev/null | awk '{print "  "$5, $9}'

[ "$CODE" = "1" ] && { echo "== upload code (whole) =="; [ -f "$R/code.tar.gz" ] && up "$R/code.tar.gz" uploads; }
echo "== upload parts =="
for t in $TARS; do
  for f in "$R/parts/$t".*.part; do
    [ -f "$f" ] || continue
    up "$f" uploads/parts || { sleep 2; up "$f" uploads/parts || echo "  FAILED $(basename "$f")"; }
  done
done
echo "== listing =="; curl -sk "$BASE/me/data?path=uploads/parts" -H "$AUTH"; echo
