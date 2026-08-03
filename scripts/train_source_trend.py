#!/usr/bin/env python3
"""train_source_trend.py — 학습 로그에서 소스별(deepvision/mmk12/pmcvqa) 정확도 추세 측정.

`log_completions true` 가 남기는 completions.jsonl 에는 문항별 정답 여부가 들어 있다.
홀드아웃 평가(n=300, 검출 하한 8.0pp)보다 두 자릿수 배 큰 표본이라, 홀드아웃이 분해하지
못하는 완만한 기울기를 여기서는 볼 수 있다. 배경 = docs/stage2_run73924_progress.md §6-c

교란요인 세 가지를 같이 확인해 출력한다.
  1) 데이터 순서 — 학습셋이 소스순 정렬이라 셔플이 없으면 추세가 전부 artifact 다.
  2) 재개 중복  — 체인 잡은 마지막 저장 이후 구간을 다시 돈다. 그대로 합치면 그 구간만 표본 2배.
  3) 길이 구성  — ★ 가장 중요. 장문 completion 은 정답률이 크게 낮다. 길이 인플레이션이
                   왔다 가면 "정답률이 오른 것처럼" 보이지만 실제로는 장문 비중이 준 것뿐이다.
                   → 장문(SoftOverlong≠0) 제외 계열을 같이 내서 구성효과와 실력 향상을 가른다.

⚠️ 원(raw) 계열의 기울기를 그대로 외삽하지 말 것. 길이 회복은 1회성이라 되돌아올 여지가 없다.
   판단은 반드시 **장문제외 계열**로 한다. 배경 = docs/stage2_run73924_progress.md §6-c

수치만 낼 때는 호스트 python3(표준 라이브러리)로 충분하다 — 계산노드 붙일 필요 없다.
`-o` 로 그림까지 그릴 때만 matplotlib 이 필요하므로 `./bin/python` 으로 실행할 것.

usage: train_source_trend.py [--run-dir DIR ...] [--train JSONL] [--bucket N] [-o PNG]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

SOURCES = ('deepvision', 'mmk12', 'pmcvqa')
USER_RE = re.compile(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', re.S)
VIS_RE = re.compile(r'<\|vision_start\|>.*?<\|vision_end\|>', re.S)
KEY = 150          # 질문 앞 150자를 조인 키로 쓴다(전문 보관 시 메모리 낭비)

# 팔레트·폰트 폴백은 plot_train_curves.py 와 동일 규약을 따른다
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, RED = "#e1e0d9", "#c3c2b7", "#c8402d"
COLOR = {'deepvision': BLUE, 'mmk12': ORANGE, 'pmcvqa': AQUA}

KO = {
    'title': '학습 로그 {n:,}건으로도 "더 학습하면 오르나"는 답이 안 나온다',
    'sub': 'completions.jsonl 문항별 채점 (step ≤ {u}) · 홀드아웃 n=300 의 {x:.0f}배인데도 그렇다',
    'a': 'deepvision — 최근 상승의 대부분은 길이 구성 변화다',
    'a_sub': '장문(SoftOverlong≠0)은 정답률이 크게 낮다 → 비중이 줄면 평균은 저절로 오른다',
    'b': '앞으로의 기울기 — 이 데이터로도 판별되지 않는다',
    'b_sub': 'step {a}~{b} 기울기를 잔여 구간에 외삽 · 가로선 = 95% 구간 (장문 제외 기준)',
    'x': 'global step', 'y': '정확도', 'raw': '원(raw) — 구성효과 포함',
    'adj': '장문 제외 ★ 판단 기준', 'share': '장문 비율 (우축)',
    'pp': '잔여 구간 기대 정확도 변화 (pp)', 'zero': '변화 없음',
    'verdict': '세 소스 모두 0 을 크게 포함한다 →\n판정은 step 1200 홀드아웃(n=1,200 + 짝지음)으로',
    'sig': '유의', 'nosig': '미검출',
    'note': '학습은 temperature=0.9 · 홀드아웃 평가는 0.0 — 비교 가능한 것은 기울기다',
}
EN = {
    'title': 'Even {n:,} training scores cannot answer "will more training help?"',
    'sub': 'per-item scores from completions.jsonl (step <= {u}) · {x:.0f}x the holdout n=300, and still not enough',
    'a': 'deepvision — most of the recent gain is length composition',
    'a_sub': 'long completions score far worse -> their share falling lifts the mean on its own',
    'b': 'The forward slope — this data cannot resolve it either',
    'b_sub': 'step {a}-{b} slope extrapolated over the remainder · bars = 95% CI (long excluded)',
    'x': 'global step', 'y': 'accuracy', 'raw': 'raw — includes composition',
    'adj': 'long excluded  * basis for judgement', 'share': 'long share (right axis)',
    'pp': 'expected accuracy change over remainder (pp)', 'zero': 'no change',
    'verdict': 'all three straddle zero by a wide margin ->\ndecide at step 1200 (n=1,200 + paired)',
    'sig': 'significant', 'nosig': 'not detected',
    'note': 'training samples at temperature=0.9, holdout eval at 0.0 — only slopes are comparable',
}


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
    """completions.jsonl 스트리밍 → step 별 (소스, 점수, 절단여부).

    ⚠️ 재개(resume) 구간은 두 런에 같은 step 이 모두 남는다 — 73924 가 step 787 까지
    갔는데 checkpoint-750 에서 재개했으므로 751~787 이 양쪽 파일에 있다. 그대로 합치면
    그 구간만 표본이 2배가 된다. plot_train_curves.py 와 같은 규약으로 **나중 런이 이긴다**.
    """
    per_run = []                                # [(run_idx, {step: [(src, acc, clipped)]})]
    total = matched = 0
    for ri, d in enumerate(run_dirs):
        path = os.path.join(d, 'completions.jsonl')
        if not os.path.exists(path):
            continue
        cur = defaultdict(list)
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
                    cur[int(s)].append((src, float(a), float(so) != 0.0))
        per_run.append((ri, cur))

    # 같은 step 이 여러 런에 있으면 가장 나중 런만 남긴다(재개 중복 제거)
    owner = {}
    for ri, cur in per_run:
        for s in cur:
            owner[s] = ri
    pts, kept, dropped = defaultdict(list), 0, 0
    for ri, cur in per_run:
        for s, items in cur.items():
            if owner[s] != ri:
                dropped += len(items)
                continue
            kept += len(items)
            for src, a, c in items:
                pts[src].append((s, a, c))
    if dropped:
        print(f'[dedup] 재개 중복 {dropped:,} 건 제거(같은 step 이 여러 런에 존재)')
    return pts, total, kept


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


def by_step(rows, exclude_long=False):
    """스텝별 평균으로 접는다. 한 스텝 안의 4개 생성은 같은 프롬프트라 독립이 아니다 —
    스텝을 단위로 삼으면 그 상관을 가정 없이 흡수한다.

    exclude_long=True 면 장문 페널티 구간(SoftOverlong≠0)을 뺀다. 길이 구성 변화가
    만들어내는 가짜 추세를 걷어내는 용도다."""
    d = defaultdict(list)
    for s, a, long_ in rows:
        if exclude_long and long_:
            continue
        d[s].append(a)
    return {s: st.mean(v) for s, v in d.items() if v}


def check_shuffle(args_json):
    """소스순 정렬된 학습셋에서 셔플이 꺼져 있으면 추세는 전부 데이터 순서 artifact 다."""
    try:
        with open(args_json) as f:
            a = json.load(f)
    except (OSError, ValueError):
        return None
    return {k: a.get(k) for k in ('dataset_shuffle', 'train_dataloader_shuffle')}


def plot(pts, stats, raw_stats, late, half, bucket, max_steps, out_path, n_total):
    """2 패널: ① deepvision 원 계열 vs 장문제외(구성효과 노출)  ② 장문제외 소스별 추세."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    font = next((c for c in ('Pretendard', 'NanumGothic', 'Nanum Gothic',
                             'Noto Sans CJK KR', 'AppleGothic', 'Malgun Gothic')
                 if c in have), None)
    L = KO if font else EN
    if not font:
        print('[plot] 한글 폰트 없음 → 영문 레이블로 렌더', file=sys.stderr)
    plt.rcParams.update({
        **({'font.family': font} if font else {}),
        'axes.unicode_minus': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'axes.edgecolor': BASE, 'axes.linewidth': 0.8,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
    })

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    last = max(s for rows in pts.values() for s, _, _ in rows)

    def buckets(bs):
        cx, cy = [], []
        for e in range(0, last + bucket, bucket):
            v = [bs[s] for s in bs if e < s <= e + bucket]
            if v:
                cx.append(e + bucket / 2)
                cy.append(st.mean(v) * 100)
        return np.array(cx), np.array(cy)

    # ── ① deepvision: 원 계열이 오르는 것처럼 보이는 이유 ─────────────
    dvr, dva = raw_stats.get('deepvision', {}), stats.get('deepvision', {})
    rx, ry = buckets(dvr)
    axx, ay = buckets(dva)
    ax.plot(rx, ry, 'o--', color=MUTED, ms=8, lw=2, alpha=.85, label=L['raw'], zorder=3)
    ax.plot(axx, ay, 'o-', color=BLUE, ms=9, lw=2.8, label=L['adj'], zorder=4)
    for x, y in zip(rx, ry):
        ax.annotate(f'{y:.1f}', (x, y), color=MUTED, fontsize=9, ha='center',
                    va='top', xytext=(0, -9), textcoords='offset points')
    for x, y in zip(axx, ay):
        ax.annotate(f'{y:.1f}', (x, y), color=BLUE, fontsize=9.5, fontweight='700',
                    ha='center', va='bottom', xytext=(0, 9), textcoords='offset points')

    d = defaultdict(list)
    for s, _, lg in pts.get('deepvision', []):
        d[s].append(1.0 if lg else 0.0)
    sx, sy = buckets({s: st.mean(v) for s, v in d.items()})
    ax2 = ax.twinx()
    ax2.bar(sx, sy, width=bucket * .55, color=ORANGE, alpha=.20, zorder=1,
            label=L['share'])
    ax2.set_ylim(0, max(sy) * 3.4)
    ax2.set_ylabel(L['share'], color=ORANGE, fontsize=9)
    ax2.tick_params(axis='y', colors=ORANGE, labelsize=8.5)
    ax2.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')   # buckets() 가 이미 ×100 했다
    for s in ('top', 'left'):
        ax2.spines[s].set_visible(False)
    ax2.spines['right'].set_color(BASE)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylim(min(ry.min(), ay.min()) - 2.2, max(ry.max(), ay.max()) + 2.2)
    ax.set_xlim(0, last)
    ax.set_title(L['a'], color=INK, fontweight='600', loc='left', pad=16, fontsize=12)
    ax.text(0, 1.015, L['a_sub'], transform=ax.transAxes, color=MUTED,
            fontsize=8.5, va='bottom', ha='left')
    ax.set_xlabel(L['x'], color=MUTED, fontsize=9)
    ax.set_ylabel(L['y'] + ' (%)', color=MUTED, fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8.5, labelcolor=INK2,
              frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=.95)

    # ── ② 앞으로의 기울기는 이 데이터로도 판별되지 않는다 ─────────────
    remain = max_steps - last
    rows = [(s, late[s][0] * 100 * remain, 1.96 * late[s][1] * 100 * remain)
            for s in SOURCES if s in late]
    ypos = list(range(len(rows)))[::-1]
    lim = max(abs(v) + e for _, v, e in rows) * 1.12
    for y, (src, v, e) in zip(ypos, rows):
        c = COLOR.get(src, MUTED)
        bx.errorbar(v, y, xerr=e, fmt='o', color=c, ms=10, elinewidth=2.6,
                    capsize=7, capthick=2.2, zorder=4)
        bx.text(v, y + .30, f'{v:+.1f}pp  [{v - e:+.0f} ~ {v + e:+.0f}]', color=c,
                fontsize=10, fontweight='700', ha='center', va='bottom', zorder=5)
    bx.axvline(0, color=INK2, lw=1.6, zorder=2)
    bx.text(0, -.52, L['zero'], color=INK2, fontsize=9, ha='center', va='center',
            bbox=dict(fc=SURFACE, ec='none', pad=2))
    bx.set_yticks(ypos)
    bx.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=10)
    bx.tick_params(axis='y', length=0)
    bx.set_xlim(-lim, lim)
    bx.set_ylim(-.75, len(rows) - .25)
    bx.set_title(L['b'], color=INK, fontweight='600', loc='left', pad=16, fontsize=12)
    bx.text(0, 1.015, L['b_sub'].format(a=half + 1, b=last), transform=bx.transAxes,
            color=MUTED, fontsize=8.5, va='bottom', ha='left')
    bx.set_xlabel(L['pp'], color=MUTED, fontsize=9)
    bx.grid(axis='x', color=GRID, lw=0.7)
    bx.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        bx.spines[s].set_visible(False)
    bx.text(.5, .06, L['verdict'], transform=bx.transAxes, color=RED, fontsize=10.5,
            fontweight='700', ha='center', va='center',
            bbox=dict(fc=SURFACE, ec=RED, lw=1.1, pad=6, alpha=.95))

    fig.suptitle(L['title'].format(n=n_total), x=0.008, y=0.975, ha='left', color=INK,
                 fontsize=16, fontweight='700')
    fig.text(0.008, 0.925, L['sub'].format(x=n_total / 300, u=last),
             ha='left', color=INK2, fontsize=10)
    fig.text(0.992, 0.925, L['note'], ha='right', color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    print(f'[plot] saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', action='append', default=None,
                    help='checkpoints/grpo_expanded_gdpo/v*/ (반복 지정 가능, 기본=전부)')
    ap.add_argument('--train', default='work/data/stage2_expanded_train.jsonl')
    ap.add_argument('--log-glob', default='logs/grpo_adv_739*.log',
                    help='절단률(completions/clipped_ratio) 을 읽을 학습 로그')
    ap.add_argument('--bucket', type=int, default=200)
    ap.add_argument('--max-steps', type=int, default=2337)
    ap.add_argument('-o', '--out', default=None,
                    help='PNG 경로. 지정하면 matplotlib 필요 → ./bin/python 으로 실행할 것')
    ap.add_argument('--until', type=int, default=None,
                    help='이 step 까지만 집계. 학습이 계속 도는 동안 문서의 수치를 '
                         '재현 가능하게 고정하려면 반드시 지정할 것')
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
    if args.until:
        pts = {s: [r for r in rows if r[0] <= args.until] for s, rows in pts.items()}
        pts = {s: rows for s, rows in pts.items() if rows}
        matched = sum(len(r) for r in pts.values())
        print(f'\n완성문 {matched:,} (step ≤ {args.until} 로 절단)\n')
    else:
        print(f'\n완성문 {matched:,}/{total:,} 매칭 ({matched / total * 100:.1f}%)'
              f'  ⚠️ --until 미지정 — 학습이 진행되면 수치가 바뀐다\n')

    B = args.bucket
    last = max(s for rows in pts.values() for s, _, _ in rows)
    edges = list(range(0, last + B, B))
    hdr = '  '.join(f'{b + 1}-{b + B:<7}' for b in edges)

    def bucket_row(bs):
        out = []
        for b in edges:
            v = [bs[s] for s in bs if b < s <= b + B]
            out.append(f'{st.mean(v) * 100:>6.2f}%  ' if v else '   -     ')
        return ''.join(out)

    stats, raw_stats = {}, {}
    print('① 원(raw) AccuracyMix — 길이 구성 변화가 섞여 있다')
    print(f'{"소스":<12}{"n":>7}   {hdr}')
    for src in [s for s in SOURCES if s in pts]:
        raw_stats[src] = by_step(pts[src])
        print(f'{src:<12}{len(pts[src]):>7}   ' + bucket_row(raw_stats[src]))

    print('\n② 장문(SoftOverlong≠0) 제외 — ★ 판단은 이 계열로 한다')
    print(f'{"소스":<12}{"n":>7}   {hdr}')
    for src in [s for s in SOURCES if s in pts]:
        stats[src] = by_step(pts[src], exclude_long=True)
        n = sum(1 for _, _, lg in pts[src] if not lg)
        print(f'{src:<12}{n:>7}   ' + bucket_row(stats[src]))

    print('\n③ 장문 비율 — ②에서 빠진 비중. 이게 움직이면 ①의 추세는 구성효과다')
    print(f'{"소스":<12}{"":>7}   {hdr}')
    for src in [s for s in SOURCES if s in pts]:
        d = defaultdict(list)
        for s, _, lg in pts[src]:
            d[s].append(1.0 if lg else 0.0)
        print(f'{src:<12}{"":>7}   ' + bucket_row({s: st.mean(v) for s, v in d.items()}))

    remain = args.max_steps - last
    half = last // 2 // 100 * 100          # 후반 구간 시작(100 단위로 내림)
    print(f'\n=== 선형 추세 (장문제외 계열, 스텝 단위 회귀) · 잔여 {remain:,} step 외삽 ===')
    print('  ⚠️ 전구간 기울기는 "초기 상승 후 평탄" 곡선에 직선을 맞춘 값이라 그대로 외삽하면 안 된다.')
    print(f'  ⚠️ 결정에 필요한 것은 **지금부터의 기울기**이므로 {half + 1}~{last} 구간을 따로 본다.\n')
    fits, late = {}, {}
    for lo, hi, lab in ((1, last, f'전구간 1~{last}'),
                        (1, half, f'초기 1~{half}'),
                        (half + 1, last, f'★ 이후 {half + 1}~{last}')):
        print(f'  [{lab}]')
        for src in [s for s in SOURCES if s in stats]:
            bs = stats[src]
            xs = [s for s in sorted(bs) if lo <= s <= hi]
            b, se = regress(xs, [bs[s] for s in xs])
            if (lo, hi) == (1, last):
                fits[src] = (b, se)
            elif lo == half + 1:
                late[src] = (b, se)
            l95, h95 = (b - 1.96 * se) * 100 * remain, (b + 1.96 * se) * 100 * remain
            flag = '유의' if abs(b / se) > 1.96 else '미검출'
            print(f'    {src:<12}{b * 10000:+7.3f} pp/100step (t={b / se:+5.2f}, {flag})'
                  f'  →  {b * 100 * remain:+6.1f}pp  [95% {l95:+.1f} ~ {h95:+.1f}]')

    print('\n  ⚠️ 원 계열이 오르고 장문제외 계열이 평평하면, 그 상승은 실력이 아니라'
          ' **장문 비중 감소**다. 길이 회복은 1회성이라 외삽 대상이 아니다.')
    span = max(abs(v) for s in late for v in
               ((late[s][0] - 1.96 * late[s][1]) * 100 * remain,
                (late[s][0] + 1.96 * late[s][1]) * 100 * remain))
    print(f'  ⇒ **후반 구간 기울기의 95% 구간은 ±{span:.0f}pp 폭이다.** 즉 이 학습 로그로도'
          f' 앞으로의 기울기는 판별되지 않는다 — 표본이 24,000 건이어도 그렇다.\n'
          f'     정확도 수준을 재는 것과 기울기를 재는 것은 다른 문제이고, 결정에 필요한 것은 후자다.')

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

    if args.out:
        plot(pts, stats, raw_stats, late, half, B, args.max_steps, args.out, matched)


if __name__ == '__main__':
    main()
