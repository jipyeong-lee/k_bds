#!/usr/bin/env python3
"""pull_all.py — B200 플랫폼의 파일을 재귀적으로 내려받는다. 세션 불필요.

왜 tar/split 이 아닌가: 업로드는 단일 요청 60초 제한 때문에 3.5G 청크가 필요했지만,
다운로드는 `GET /me/data/file` 이 파일 단위라 최대 개별 파일(331MB, optimizer.pt)이면 충분히 작다.
tar 를 만들려면 GPU 세션 exec 이 필요하고 B200 디스크도 2배로 먹는다 — 파일 단위가 더 싸다.

실행 위치: KISTI 가 가장 빠르다(업로드 때 138 MB/s 실측). 로컬에서도 동작한다.
자격증명: 환경변수 ORCH_BASE_URL·ORCH_PAT, 없으면 --cred 로 준 sh 파일에서 BASE=/PAT= 를 읽는다.

이미 있는 파일은 **크기가 같으면 건너뛴다** → 중단 후 재실행이 곧 이어받기다.

  python3 pull_all.py --dest ~/b200_migrate runs kbds_project/logs safe_ckpt uploads
  python3 pull_all.py --dest ~/b200_migrate --cred /scratch/migrate_k266_to_gpu/upload_to_platform_v2.sh runs
"""
import argparse, json, os, re, ssl, sys, time, urllib.parse, urllib.request

def creds(cred_file):
    base, pat = os.environ.get("ORCH_BASE_URL"), os.environ.get("ORCH_PAT")
    if base and pat:
        return base.rstrip("/"), pat
    if cred_file and os.path.exists(cred_file):
        txt = open(cred_file, encoding="utf-8", errors="replace").read()
        b = re.search(r'^\s*BASE\s*=\s*"?([^"\s]+)', txt, re.M)
        p = re.search(r'^\s*PAT\s*=\s*"?([^"\s]+)', txt, re.M)
        if b and p:
            return b.group(1).rstrip("/"), p.group(1)
    sys.exit("자격증명 없음: ORCH_BASE_URL/ORCH_PAT 환경변수 또는 --cred 파일이 필요하다")

CTX = ssl._create_unverified_context()

def req(base, pat, path_qs, binary_to=None, tries=3):
    url = base + path_qs
    last = None
    for a in range(tries):
        try:
            r = urllib.request.Request(url, headers={"Authorization": "Bearer " + pat})
            with urllib.request.urlopen(r, context=CTX, timeout=600) as resp:
                if binary_to is None:
                    return json.load(resp)
                tmp = binary_to + ".part"
                got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk); got += len(chunk)
                os.replace(tmp, binary_to)
                return got
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"{path_qs} 실패: {last}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="$ORCH_HOME 기준 상대경로 (디렉터리 또는 파일)")
    ap.add_argument("--dest", required=True, help="내려받을 로컬 디렉터리")
    ap.add_argument("--cred", default="/scratch/migrate_k266_to_gpu/upload_to_platform_v2.sh")
    ap.add_argument("--dry", action="store_true", help="목록만 세고 내려받지 않는다")
    a = ap.parse_args()
    base, pat = creds(a.cred)

    def ls(p):
        return req(base, pat, "/me/data?path=" + urllib.parse.quote(p)).get("entries", [])

    def resolve(p):
        """입력 경로가 파일인지 디렉터리인지 부모 목록에서 판별한다(크기도 같이 얻는다)."""
        parent, name = (p.rsplit("/", 1) if "/" in p else ("", p))
        for e in ls(parent):
            if e["name"] == name:
                return e
        return None

    # 1) 트리를 먼저 다 훑어 총량을 확정한다 — 진행률을 보여주기 위해서다.
    files, t0 = [], time.time()
    stack = []
    for p in a.paths:
        e = resolve(p)
        if e is None:
            print(f"  ⚠️ 원격에 없음: {p}", flush=True)
        elif e["is_dir"]:
            stack.append(p)
        else:
            files.append((p, e.get("size", 0)))
    while stack:
        p = stack.pop()
        try:
            ents = ls(p)
        except Exception as e:
            print(f"  ⚠️ 목록 실패 {p}: {e}", flush=True); continue
        for e in ents:
            child = f"{p}/{e['name']}"
            (stack if e["is_dir"] else files).append(child if e["is_dir"] else (child, e.get("size", 0)))
        if len(files) % 20000 < len(ents):
            print(f"  …훑는 중: {len(files):,} 파일", flush=True)
    total = sum(s for _, s in files)
    print(f"대상 {len(files):,} 파일 · {total/2**30:.2f} GiB · 훑기 {time.time()-t0:.0f}s", flush=True)
    if a.dry:
        return

    done = skip = fail = 0
    got_bytes = 0
    t0 = time.time()
    for i, (rel, size) in enumerate(sorted(files), 1):
        out = os.path.join(a.dest, rel)
        if os.path.exists(out) and os.path.getsize(out) == size:
            skip += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            n = req(base, pat, "/me/data/file?path=" + urllib.parse.quote(rel), binary_to=out)
            if size and n != size:
                print(f"  ⚠️ 크기 불일치 {rel}: 받은 {n} ≠ 기대 {size}", flush=True)
            done += 1; got_bytes += n
        except Exception as e:
            fail += 1
            print(f"  ❌ {rel}: {e}", flush=True)
        if i % 500 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  [{i:,}/{len(files):,}] 받음 {done:,} 건너뜀 {skip:,} 실패 {fail} · "
                  f"{got_bytes/2**30:.2f} GiB · {got_bytes/2**20/max(el,1):.0f} MB/s", flush=True)
    print(f"\n완료: 받음 {done:,} · 건너뜀 {skip:,} · 실패 {fail} · {got_bytes/2**30:.2f} GiB "
          f"· {time.time()-t0:.0f}s")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
