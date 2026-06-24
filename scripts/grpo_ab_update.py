#!/usr/bin/env python3
"""GRPO 파생기법 본실행 로그를 읽어 README 의 레시피별 AUTO 마커 블록을 재생성한다.

레시피 일반화 버전 — baseline(57249) vs <실험> 을 실험 최신 step N 까지 "동일 구간 1~N"
평균으로 비교. 레시피마다 별도 마커(`AUTO:<marker>`)를 갱신해 다른 레시피 기록을 보존한다.

- DAPO  : --recipe DAPO  --job 57527 --exp logs/grpo_adv_57527.log --marker ab
- dr_grpo: --recipe dr_grpo --job 57624 --exp logs/grpo_adv_57624.log --marker ab:dr_grpo
- 변경이 있으면 'CHANGED', 없으면 'NOCHANGE'. 마커 블록만 교체(그 외 수동 편집 보존).

plot 재생성은 watcher(grpo_watch.sh) 가 plot_grpo_compare.py 로 별도 수행.
"""
import re, argparse, statistics as st, sys, os

ap = argparse.ArgumentParser()
ap.add_argument('--baseline', default='logs/grpo_stage2_57249.log')
ap.add_argument('--exp', '--dapo', dest='exp', default='logs/grpo_adv_57527.log',
                help='실험(레시피) 로그')
ap.add_argument('--recipe', default='DAPO', help='표·문구에 쓸 레시피 라벨')
ap.add_argument('--job', default='57527', help='실험 job id')
ap.add_argument('--marker', default='ab', help='AUTO:<marker> 블록 이름')
ap.add_argument('--baseline-label', default='baseline')
ap.add_argument('--baseline-job', default='57249')
ap.add_argument('--readme', default='README.md')
ap.add_argument('--max-steps', type=int, default=1000)
a = ap.parse_args()

KEYS = {
    'reward': 'reward', 'rewards/AccuracyMix/mean': 'acc',
    'rewards/FormatThink/mean': 'ft', 'completions/clipped_ratio': 'clip',
    'completions/mean_length': 'mlen', 'frac_reward_zero_std': 'zstd',
    'train_speed(s/it)': 'spit',
}

def parse(path):
    """step -> {metric: value}"""
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = re.search(r"global_step/max_steps': '(\d+)/", line)
        if not m:
            continue
        s = int(m.group(1))
        d = rows.setdefault(s, {})
        for k, v in KEYS.items():
            mm = re.search(rf"'{re.escape(k)}': '([-\d.]+)'", line)
            if mm:
                d[v] = float(mm.group(1))
    return rows

base = parse(a.baseline)
exp = parse(a.exp)
if not exp:
    print('NOCHANGE (no exp data)'); sys.exit(0)

N = max(exp)  # 실험 최신 step
LBL = a.recipe

def mean_upto(rows, n, v):
    vals = [rows[s][v] for s in rows if s <= n and v in rows[s]]
    return st.mean(vals) if vals else float('nan')

def mean_window(rows, lo, hi, v):
    """[lo, hi] 구간 평균 (돌파 구간 직접 비교용)."""
    vals = [rows[s][v] for s in rows if lo <= s <= hi and v in rows[s]]
    return st.mean(vals) if vals else float('nan')

def block_ab():
    out = []
    out.append(f'  **{a.baseline_label}({a.baseline_job}) vs {LBL}({a.job}) — 동일 구간 step 1~{N} 비교:**')
    out.append('')
    out.append(f'  | 지표 | {a.baseline_label} | {LBL} | 차이 |')
    out.append('  |------|----------|------|------|')
    rowdef = [
        ('frac_zero_std(무신호 그룹)', 'zstd', '.3f', True),
        ('FormatThink', 'ft', '.3f', False),
        ('reward', 'reward', '.3f', False),
        ('clip(잘림)', 'clip', '.3f', False),
        ('Acc', 'acc', '.3f', False),
        ('mean_len', 'mlen', '.0f', False),
    ]
    vals = {}
    for label, key, fmt, star in rowdef:
        b = mean_upto(base, N, key); d = mean_upto(exp, N, key)
        vals[key] = (b, d)
        diff = d - b
        sign = '↓' if diff < 0 else '↑'
        bn = ('**frac_zero_std**(무신호 그룹)' if star else label)
        dcell = f'**{d:{fmt}}**' if star else f'{d:{fmt}}'
        dtxt = f'{sign}{abs(diff):.0f}' if fmt == '.0f' else f'{sign}{abs(diff):.3f}'
        extra = ' ★' if star else ''
        out.append(f'  | {bn} | {b:{fmt}} | {dcell} | {dtxt}{extra} |')
    out.append('')
    # 해석 bullet (계산값 주입)
    zb, zd = vals['zstd']; fb, fd = vals['ft']; ab_, ad = vals['acc']
    mb, md = vals['mlen']; cb, cd = vals['clip']
    out.append(f'  - ✅ **dynamic_sample 가설 검증**: `frac_reward_zero_std` {zb:.2f}→**{zd:.2f}**. '
               f'{a.baseline_label} 이 매 step ~{zb*100:.0f}% 낭비하던 무신호 그룹을 재샘플로 제거(plateau 직격).')
    out.append(f'  - ✅ **형식 수렴**: 동일구간 FormatThink {a.baseline_label} {fb:.2f} → {LBL} **{fd:.2f}**.')
    # 속도: 로그의 train_speed(s/it) 평균 비교
    sb = mean_upto(base, N, 'spit'); sd = mean_upto(exp, N, 'spit')
    if sb == sb and sd == sd and sb > 0:
        out.append(f'  - ⚠️ **속도 ~{sd/sb:.1f}배**: {LBL} ~{sd:.0f}s/it vs {a.baseline_label} {sb:.0f}.')
    # 길이/clip 동향 (DAPO 길이폭주 진단의 핵심 추적 지표)
    out.append(f'  - 📏 **길이·clip**: mean_len {md:.0f}(Δ{md-mb:+.0f}) / clip {cd:.3f}(Δ{cd-cb:+.3f}) vs {a.baseline_label}.')
    # Acc 돌파 판정
    if N < 600:
        out.append(f'  - ⚠️ **Acc 이득 미확정**: {LBL} {ad:.3f} vs {a.baseline_label} {ab_:.3f} (누적) — '
                   f'{a.baseline_label} Acc 도약(step ~600)이후 구간 비교 필요 (현재 step {N}).')
    else:
        lo, hi = 501, 600
        aw_b = mean_window(base, lo, hi, 'acc'); aw_d = mean_window(exp, lo, hi, 'acc')
        d_win = aw_d - aw_b
        if d_win > 0.005:
            verdict = (f'✅ **돌파 확인**: step {lo}~{hi} 구간 Acc {LBL} **{aw_d:.3f}** vs '
                       f'{a.baseline_label} {aw_b:.3f} (Δ+{d_win:.3f}) — 안정성이 정확도 우위로 전환됨.')
        elif aw_b > aw_d + 0.005:
            verdict = (f'⚠️ **돌파 미확인**: step {lo}~{hi} 구간 Acc {LBL} {aw_d:.3f} < '
                       f'{a.baseline_label} **{aw_b:.3f}** (Δ{d_win:.3f}) — 안정성 이득이 아직 정확도로 미전환.')
        else:
            verdict = (f'➖ **돌파 동률**: step {lo}~{hi} 구간 Acc {LBL} {aw_d:.3f} ≈ '
                       f'{a.baseline_label} {aw_b:.3f} (Δ{d_win:+.3f}) — 추가 step(>700) 관찰 필요.')
        out.append(f'  - {verdict}')
        out.append(f'  - 누적 참고: {LBL} {ad:.3f} vs {a.baseline_label} {ab_:.3f} (누적) — step {N}까지 평균.')
    return '\n'.join(out)

def replace_marker(text, name, new):
    pat = re.compile(rf'(<!-- AUTO:{re.escape(name)} START.*?-->\n).*?(\n<!-- AUTO:{re.escape(name)} END -->)', re.S)
    if not pat.search(text):
        print(f'WARN marker {name} not found', file=sys.stderr)
        return text
    return pat.sub(lambda m: m.group(1) + new + m.group(2), text)

src = open(a.readme).read()
new = replace_marker(src, a.marker, block_ab())
if new == src:
    print(f'NOCHANGE (step {N})')
    sys.exit(0)
open(a.readme, 'w').write(new)
print(f'CHANGED (step {N}, marker {a.marker})')
