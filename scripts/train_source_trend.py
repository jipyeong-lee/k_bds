#!/usr/bin/env python3
"""train_source_trend.py — 학습 로그에서 소스별(deepvision/mmk12/pmcvqa) 정확도 추세 측정.

`log_completions true` 가 남기는 completions.jsonl 에는 문항별 정답 여부가 들어 있다.
홀드아웃 평가(n=300, 검출 하한 8.0pp)보다 두 자릿수 배 큰 표본이라, 홀드아웃이 분해하지
못하는 완만한 기울기를 여기서는 볼 수 있다. 배경 = docs/stage2_run73924_progress.md §6-c

교란요인 두 가지를 같이 확인해 출력한다.
  1) 데이터 순서 — 학습셋이 소스순 정렬이라 셔플이 없으면 추세가 전부 artifact 다.
  2) 절단(clip)  — 길이 인플레이션 구간에서 절단이 늘면 정확도가 기계적으로 떨어진다.
                   AccMix ≈ (1-clip)·a 로 보정한 a 도 같이 낸다.

호스트 python3(표준 라이브러리)만으로 동작한다 — 계산노드 붙일 필요 없다.

usage: train_source_trend.py [--run-dir DIR ...] [--train JSONL] [--bucket N]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

SOURCES = ('deepvision', 'mmk12', 'pmcvqa')
USER_RE = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.S)
VIS_RE = re.compile(r'<\|vision_start\|>.*?<\|vision_end\|>', re.S)
KEY = 150          # 질문 앞 150자를 조인 키로 쓴다(전문 보관 시 메모리 낭비)


def source_of(path):
    p = path.lower()
    return next((s for s in SOURCES if s in p), '?')


def load_key_to_source(train_jsonl):
    """학습셋에서 '질문 앞머리 → 소스' 사전. 이미지 경로가 소스의 유일한 근거다."""
    m = {}
    with open(train_jsonl) as f:
        for line in f:
            d = json.loads(line)
            q = d['messages'][0]['content'].replace('<image>', '').strip()
            m[q[:KEY]] = source_of((d.get('images') or [''])[0])
    return m


def collect(run_dirs, key2src):
    """completions.jsonl 스트리밍 → step 별 (소스, 점수, 절단여부)."""
    pts = defaultdict(list)                     # source -> [(step, acc, clipped)]
    total = matched = 0
    for d in run_dirs:
        path = os.path.join(d, 'completions.jsonl')
        if not os.path.exists(path):
            continue
        with open(path, errors='ignore') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                     # 실행 중 잘린 마지막 줄
                steps, prompts, accs = rec.get('step'), rec.get('prompt'), rec.get('AccuracyMix')
                if not (steps and prompts and accs):
                    continue
                # SoftOverlong 는 길이 페널티 — 0 이 아니면 soft 구간(길이 초과 근처)이다
                soft = rec.get('SoftOverlong') or [0.0] * len(accs)
                for s, p, a, so in zip(steps, prompts, accs, soft):
                    total += 1
                    m = USER_RE.search(p)
                    if not m:
                        continue
                    q = VIS_RE.sub('', m.group(1)).replace('<image>', '').strip()[:KEY]
                    src = key2src.get(q)
                    if src is None:
                        continue
                    matched += 1
                    pts[src].append((int(s), float(a), float(so) != 0.0))
    return pts, total, matched


def regress(xs, ys):
    """단순 선형회귀 → (기울기, 기울기 표준오차). 단위는 입력 그대로."""
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan')
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float('nan'), float('nan')
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    return b, se


def by_step(rows):
    """스텝별 평균으로 접는다. 한 스텝 안의 4개 생성은 같은 프롬프트라 독립이 아니다 —
    스텝을 단위로 삼으면 그 상관을 가정 없이 흡수한다."""
    d = defaultdict(list)
    for s, a, _ in rows:
        d[s].append(a)
    return {s: st.mean(v) for s, v in d.items()}


def check_shuffle(args_json):
    """소스순 정렬된 학습셋에서 셔플이 꺼져 있으면 추세는 전부 데이터 순서 artifact 다."""
    try:
        with open(args_json) as f:
            a = json.load(f)
    except (OSError, ValueError):
        return None
    return {k: a.get(k) for k in ('dataset_shuffle', 'train_dataloader_shuffle')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', action='append', default=None,
                    help='checkpoints/grpo_expanded_gdpo/v*/ (반복 지정 가능, 기본=전부)')
    ap.add_argument('--train', default='work/data/stage2_expanded_train.jsonl')
    ap.add_argument('--log-glob', default='logs/grpo_adv_739*.log',
                    help='절단률(completions/clipped_ratio) 을 읽을 학습 로그')
    ap.add_argument('--bucket', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=2337)
    args = ap.parse_args()

    runs = args.run_dir or sorted(glob.glob('work/checkpoints/grpo_expanded_gdpo/v*/'))
    if not runs:
        raise SystemExit('run 디렉터리를 못 찾았다 — --run-dir 로 지정할 것')

    for r in runs:
        sh = check_shuffle(os.path.join(r, 'args.json'))
        if sh and not all(sh.values()):
            print(f'⚠️  {r}: 셔플 꺼짐 {sh} — 학습셋이 소스순 정렬이라 아래 추세는 신뢰할 수 없다')
        elif sh:
            print(f'✓ {r}: {sh}')

    key2src = load_key_to_source(args.train)
    pts, total, matched = collect(runs, key2src)
    if not matched:
        raise SystemExit('매칭된 완성문이 없다 — --train 이 학습에 쓴 파일과 같은지 확인할 것')
    print(f'\n완성문 {matched:,}/{total:,} 매칭 ({matched / total * 100:.1f}%)\n')

    B = args.bucket
    last = max(s for rows in pts.values() for s, _, _ in rows)
    edges = list(range(0, last + B, B))
    hdr = '  '.join(f'{b + 1}-{b + B:<7}' for b in edges)
    print(f'{"소스":<12}{"n":>7}   {hdr}')
    print('-' * (21 + len(hdr)))

    stats = {}
    for src in sorted(pts):
        rows = pts[src]
        bs = by_step(rows)
        stats[src] = bs
        cells = []
        for b in edges:
            v = [bs[s] for s in bs if b < s <= b + B]
            cells.append(f'{st.mean(v) * 100:>6.2f}%  ' if v else '   -     ')
        print(f'{src:<12}{len(rows):>7}   ' + ''.join(cells))

    remain = args.max_steps - last
    print(f'\n=== 선형 추세 (스텝 단위 회귀) · 잔여 {remain:,} step 외삽 ===')
    for src in sorted(stats):
        bs = stats[src]
        xs = sorted(bs)
        b, se = regress(xs, [bs[s] for s in xs])
        lo, hi = (b - 1.96 * se) * 100 * remain, (b + 1.96 * se) * 100 * remain
        flag = '유의' if abs(b / se) > 1.96 else '미검출'
        print(f'  {src:<12}{b * 10000:+7.3f} pp/100step (t={b / se:+5.2f}, {flag})'
              f'  →  {b * 100 * remain:+6.1f}pp  [95% {lo:+.1f} ~ {hi:+.1f}]')

    # 절단 보정은 학습 .log 의 completions/clipped_ratio 를 쓴다 — 이것이 진짜 절단률이다.
    # completions.jsonl 의 SoftOverlong 은 "soft 페널티 구간(>4,096자)" 이라 절단보다 넓은 집합이라
    # 대용으로 쓰면 보정이 과해진다. 소스별로는 절단률을 알 수 없어 전체 합산으로만 낸다.
    logs = sorted(glob.glob(args.log_glob))
    if logs:
        agg = {}
        for lf in logs:
            with open(lf, errors='ignore') as f:
                for line in f:
                    if "'global_step/max_steps'" not in line:
                        continue
                    m = re.search(r"'global_step/max_steps': '(\d+)/", line)
                    vals = {k: re.search(r"'%s': '([-\d.e+]+)'" % re.escape(k), line)
                            for k in ('rewards/AccuracyMix/mean', 'completions/clipped_ratio')}
                    if m and all(vals.values()):
                        agg[int(m.group(1))] = (float(vals['rewards/AccuracyMix/mean'].group(1)),
                                                float(vals['completions/clipped_ratio'].group(1)))
        if agg:
            print('\n=== 절단 보정(전체) — AccMix ≈ (1-clip)·a, 100 step 단위 ===')
            print(f'  {"구간":>12} {"AccMix":>8} {"clip":>7} {"보정 a":>8}')
            for b in range(0, max(agg) + 100, 100):
                seg = [agg[s] for s in agg if b < s <= b + 100]
                if not seg:
                    continue
                acc = st.mean(a for a, _ in seg)
                clip = st.mean(c for _, c in seg)
                print(f'  {b + 1:>5}-{b + 100:<6} {acc * 100:>7.2f}% {clip * 100:>6.2f}% {acc / (1 - clip) * 100:>7.2f}%')

    print('\n※ 학습 정확도는 temperature=0.9 샘플이고 홀드아웃 평가는 temperature=0.0 이라'
          ' 절대 수준은 다르다. 비교 가능한 것은 기울기다.')


if __name__ == '__main__':
    main()
