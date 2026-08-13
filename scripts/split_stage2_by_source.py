#!/usr/bin/env python3
"""split_stage2_by_source.py — Stage-2 확장셋을 도메인별로 분할(전문가 학습용).

DeepSeek-V4 구조(도메인별 전문가 → 통합)로 가려면 학습셋부터 도메인별로 나뉘어야 한다.
→ docs/deepseek_v4_pipeline_adoption.md §2

**소스 판별 근거는 이미지 경로다.** 학습셋에 source 컬럼이 없다(키 = messages/images/solution).
`scripts/train_source_trend.py:74` 의 `source_of()` 와 동일한 규칙을 쓴다 — 규칙이 갈라지면
소스별 추세 분석과 전문가 학습이 서로 다른 정의를 쓰게 되므로 여기서도 같은 것을 쓴다.

⚠️ **미분류를 조용히 버리지 않는다.** 하나라도 나오면 기본적으로 실패한다(`--allow-unknown` 으로만 허용).
   분할은 되돌리기 어렵고, 조용히 샌 샘플은 나중에 "왜 건수가 안 맞지"로 돌아온다.

사용:
    python3 scripts/split_stage2_by_source.py --dry          # 미리보기(파일 안 씀)
    python3 scripts/split_stage2_by_source.py                # 분할 수행
    python3 scripts/split_stage2_by_source.py --verify       # 기존 분할본 검증만

출력: $DATA_DIR/domains/stage2_<source>.jsonl  (+ split_manifest.json)
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

# train_source_trend.py:33 과 동일 — 갈라지면 안 된다
SOURCES = ('deepvision', 'mmk12', 'pmcvqa')
PROMPTS_PER_STEP = 8          # pdtbs(1) × world(8) × accum(4) ÷ num_generations(4)


def source_of(path):
    """train_source_trend.py:74 와 동일 규칙. 이미지 경로가 소스의 유일한 근거다."""
    p = (path or '').lower()
    return next((s for s in SOURCES if s in p), 'unknown')


def sha1(path, buf=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()[:12]


def load(path):
    rows = []
    for i, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f'❌ {path}:{i} JSON 파싱 실패 — {e}')
        imgs = d.get('images') or ['']
        rows.append((source_of(imgs[0]), line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=None, help='기본 = $DATA_DIR/stage2_expanded_train.jsonl')
    ap.add_argument('--outdir', default=None, help='기본 = $DATA_DIR/domains')
    ap.add_argument('--dry', action='store_true', help='통계만 내고 파일을 쓰지 않는다')
    ap.add_argument('--verify', action='store_true', help='기존 분할본이 원본과 일치하는지만 검사')
    ap.add_argument('--allow-unknown', action='store_true',
                    help='미분류 샘플을 unknown 으로 따로 저장하고 계속 진행')
    a = ap.parse_args()

    proj = Path(__file__).resolve().parent.parent
    data_dir = Path(os.environ.get('DATA_DIR', proj / 'work' / 'data'))
    src = Path(a.input) if a.input else data_dir / 'stage2_expanded_train.jsonl'
    out = Path(a.outdir) if a.outdir else data_dir / 'domains'

    if not src.exists():
        sys.exit(f'❌ 원본 없음: {src}')

    rows = load(src)
    cnt = Counter(s for s, _ in rows)
    total = len(rows)
    print(f'[split] 원본 {src}')
    print(f'[split] {total:,}건 · sha1 {sha1(src)}\n')

    hdr = f"{'소스':<12}{'건수':>9}{'비중':>8}{'1 epoch':>11}{'600 step 노출':>16}"
    print(hdr)
    print('-' * len(hdr.encode('utf-8').decode('utf-8')) if False else '-' * 58)
    for s in SOURCES + ('unknown', ):
        n = cnt.get(s, 0)
        if not n:
            continue
        ep_steps = n / PROMPTS_PER_STEP
        seen600 = 600 * PROMPTS_PER_STEP / n
        print(f'{s:<12}{n:>9,}{100*n/total:>7.1f}%{ep_steps:>9,.0f} step{seen600:>13.2f} epoch')
    print()

    unk = cnt.get('unknown', 0)
    if unk and not a.allow_unknown:
        print(f'❌ 미분류 {unk:,}건. 이미지 경로에 {SOURCES} 중 어느 것도 없다.')
        print('   예시:')
        for s, line in rows:
            if s == 'unknown':
                print('    ', (json.loads(line).get('images') or [''])[0][:110])
                break
        sys.exit('   --allow-unknown 을 주면 별도 파일로 빼고 진행한다. 그 전에 원인을 확인할 것.')

    groups = sorted(SOURCES + (('unknown', ) if unk else ()))
    targets = {s: out / f'stage2_{s}.jsonl' for s in groups}

    if a.verify:
        ok = True
        for s, p in targets.items():
            if not p.exists():
                print(f'  ✗ 없음   {p}')
                ok = False
                continue
            n = sum(1 for _ in open(p))
            good = n == cnt.get(s, 0)
            ok &= good
            print(f"  {'✅' if good else '🚨'} {s:<11} {n:>8,}건  (원본 기준 {cnt.get(s,0):,})")
        sys.exit(0 if ok else '🚨 분할본이 원본과 불일치 — 다시 분할할 것')

    if a.dry:
        print('[split] --dry: 파일을 쓰지 않았다. 출력 예정 경로:')
        for s, p in targets.items():
            print(f'    {p}')
        return

    out.mkdir(parents=True, exist_ok=True)
    written = Counter()
    handles = {s: open(p, 'w') for s, p in targets.items()}
    try:
        for s, line in rows:
            handles[s].write(line + '\n')
            written[s] += 1
    finally:
        for h in handles.values():
            h.close()

    # ── 무결성: 건수 일치 + 총합 보존 ──────────────────────────────────
    bad = [s for s in groups if written[s] != cnt.get(s, 0)]
    if bad or sum(written.values()) != total:
        sys.exit(f'🚨 분할 중 손실 — 불일치 {bad} · 총합 {sum(written.values()):,}/{total:,}')

    manifest = {
        'source_file': str(src),
        'source_sha1': sha1(src),
        'total': total,
        'prompts_per_step': PROMPTS_PER_STEP,
        'splits': {
            s: {
                'path': str(targets[s]),
                'count': written[s],
                'sha1': sha1(targets[s]),
                'one_epoch_steps': round(written[s] / PROMPTS_PER_STEP),
            }
            for s in groups
        },
    }
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print('[split] 완료 — 건수 일치, 총합 보존 확인')
    for s in groups:
        m = manifest['splits'][s]
        print(f"    {s:<11} {m['count']:>8,}건  sha1 {m['sha1']}  1 epoch = {m['one_epoch_steps']:,} step")
    print(f"\n[split] manifest: {out / 'split_manifest.json'}")
    print('[split] 검증: python3 scripts/split_stage2_by_source.py --verify')


if __name__ == '__main__':
    main()
