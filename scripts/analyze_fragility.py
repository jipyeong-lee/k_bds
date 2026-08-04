#!/usr/bin/env python3
"""analyze_fragility.py — 형식 취약성 스윕 결과 분석 + 플롯.

probe_format_fragility.py 는 (tag, temp) 셀마다 master[:n] 을 **입력 순서 그대로**
기록한다. 시드가 고정이라 셀 안 i 번째 줄은 어느 체크포인트에서든 같은 프롬프트다.
→ 줄 인덱스로 조인하면 체크포인트 간 짝지음(McNemar) 이 성립한다. 비짝지음
   Fisher 로 가면 같은 n 에서 검출 하한이 몇 배 올라간다.

사용: python scripts/analyze_fragility.py [--results F] [--samples-glob G] [--out PNG]
"""
import os, json, glob, argparse, math
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='logs/probe_fragility_results.jsonl')
ap.add_argument('--samples-glob', default='logs/probe_frag_samples_step*.jsonl')
ap.add_argument('--out', default='docs/assets/stage2_format_fragility.png')
a = ap.parse_args()


def step_of(tag):
    return int(str(tag).replace('step', ''))


# ---- 집계표 ---------------------------------------------------------------
rows = [json.loads(l) for l in open(a.results)]
cells = {(step_of(r['tag']), r['temp']): r for r in rows}       # 재실행 시 뒤가 이김
steps = sorted({s for s, _ in cells})
temps = sorted({t for _, t in cells})

print('=' * 78)
print('비절단 형식실패율 (주 지표)      — 절단은 제외. 조기경보 지표와 정의 동일')
print('=' * 78)
print(f"{'step':>6} " + ' '.join(f'{"T="+str(t):>12}' for t in temps))
for s in steps:
    line = f'{s:>6} '
    for t in temps:
        c = cells.get((s, t))
        cell = f"{c['fail_nontrunc'] * 100:.2f}%" if c else '—'
        line += cell.rjust(12) + ' '
    print(line.rstrip())

for key, title, pct in (('trunc_rate', '절단율', True),
                        ('mean_chars', '평균 길이(자)', False),
                        ('rep_rate', '반복 출력 비율(문자 30-gram >50%)', True)):
    print()
    print(f'--- {title} ---')
    print(f"{'step':>6} " + ' '.join(f'{"T="+str(t):>12}' for t in temps))
    for s in steps:
        line = f'{s:>6} '
        for t in temps:
            c = cells.get((s, t))
            if not c:
                line += f"{'—':>12} "
            elif pct:
                line += f"{c[key]*100:.2f}%".rjust(12) + ' '
            else:
                line += f"{c[key]:.0f}".rjust(12) + ' '
        print(line.rstrip())

# ---- 짝지음 검정 -----------------------------------------------------------
per = defaultdict(list)          # (step, temp) -> [ (fmt, trunc), ... ]  입력 순서
for f in sorted(glob.glob(a.samples_glob)):
    for l in open(f):
        d = json.loads(l)
        per[(step_of(d['tag']), d['temp'])].append((d['fmt'], d['trunc']))


def mcnemar_exact(b, c):
    """양측 exact binomial. b,c = 불일치쌍."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


print()
print('=' * 78)
print('짝지음 비교 (같은 프롬프트, 양쪽 모두 비절단인 항목만)')
print('=' * 78)
print(f"{'비교':>16} {'T':>5} {'n':>5} {'실패 A':>8} {'실패 B':>8} {'차이':>9} "
      f"{'b':>4} {'c':>4} {'p':>8}")
base = steps[0]
for t in temps:
    for i, s1 in enumerate(steps):
        for s2 in steps[i + 1:]:
            A, B = per.get((s1, t), []), per.get((s2, t), [])
            n = min(len(A), len(B))
            if n == 0:
                continue
            pairs = [(A[j], B[j]) for j in range(n) if not A[j][1] and not B[j][1]]
            if not pairs:
                continue
            fa = sum(1 for x, _ in pairs if x[0] == 0) / len(pairs)
            fb = sum(1 for _, y in pairs if y[0] == 0) / len(pairs)
            b = sum(1 for x, y in pairs if x[0] == 0 and y[0] == 1)   # A만 실패
            c = sum(1 for x, y in pairs if x[0] == 1 and y[0] == 0)   # B만 실패
            p = mcnemar_exact(b, c)
            flag = ' ***' if p < 0.001 else (' **' if p < 0.01 else (' *' if p < 0.05 else ''))
            print(f"{f'{s1} vs {s2}':>16} {t:>5} {len(pairs):>5} {fa*100:>7.2f}% "
                  f"{fb*100:>7.2f}% {(fb-fa)*100:>+8.2f}p {b:>4} {c:>4} {p:>8.2g}{flag}")

# ---- 플롯 -----------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print(f'\n[plot] skip ({e})')
    raise SystemExit(0)

for _f in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):   # 한글 라벨 — 그리기 전에 잡아야 한다
    if any(_f == fm.name for fm in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.get_cmap('viridis')
norm = {s: i / max(1, len(steps) - 1) for i, s in enumerate(steps)}

ax = axes[0]
for s in steps:
    xs = [t for t in temps if (s, t) in cells]
    ys = [cells[(s, t)]['fail_nontrunc'] * 100 for t in xs]
    ax.plot(xs, ys, 'o-', color=cmap(norm[s]), label=f'step {s}', lw=2, ms=6)
ax.set_xlabel('sampling temperature')
ax.set_ylabel('비절단 형식실패율 (%)')
ax.set_title('형식 취약성 — temperature 스윕')
ax.axvline(0.9, color='crimson', ls=':', lw=1.2)
ax.text(0.9, ax.get_ylim()[1] * 0.96, ' 학습 T=0.9', color='crimson', fontsize=9, va='top')
ax.grid(alpha=.3)
ax.legend(fontsize=9)

ax = axes[1]
for t in temps:
    xs = [s for s in steps if (s, t) in cells]
    ys = [cells[(s, t)]['fail_nontrunc'] * 100 for s in xs]
    ax.plot(xs, ys, 'o-', label=f'T={t}', lw=2, ms=6)
ax.axvspan(850, 900, color='crimson', alpha=.12)
ax.text(875, ax.get_ylim()[1] * 0.5, '절벽\n850→900', ha='center', fontsize=9, color='crimson')
ax.set_xlabel('학습 step')
ax.set_ylabel('비절단 형식실패율 (%)')
ax.set_title('학습 진행에 따른 취약성')
ax.grid(alpha=.3)
ax.legend(fontsize=9)

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
fig.tight_layout()
fig.savefig(a.out, dpi=140)
print(f'\n[plot] → {a.out}')
