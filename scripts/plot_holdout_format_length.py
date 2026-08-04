#!/usr/bin/env python3
"""plot_holdout_format_length.py — 홀드아웃 형식·길이·층별 정확도 종합.

기존 `stage2_holdout_paired.png` 는 정확도만 본다. 이 그림은 그 정확도가
**형식·길이와 어떻게 맞물리는지**를 붙인다. 네 가지를 한 장에서 답한다.

  ① 형식 두 잣대가 중반에 왜 갈리는가 (홀드아웃 느슨 vs 학습 FormatThink)
  ② 길이 인플레이션과 2,048 토큰 상한의 관계
  ③ 소스 안 층별 — 400 이후 오른 것은 deepvision·vl 하나뿐이다
  ④ step 900 채점 구멍 — 형식은 균일하게 깨졌는데 점수는 소스별로 갈린다

🚨 ①의 두 곡선은 **직접 비교 불가**다. 지표 정의(느슨한 <answer> 존재 여부 vs
   엄격한 FormatThink), temperature(0.0 vs 0.9), 토큰 상한(2,048 vs 6,144)이
   전부 다르다. 같은 축에 그리는 것은 "추세가 언제 갈라지는가"를 보기 위해서지
   수준을 비교하기 위해서가 아니다 — 캡션에 명시한다.

사용: python3 scripts/plot_holdout_format_length.py [-o docs/assets/....png]
"""
import argparse, csv, glob, json, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--out', default='docs/assets/stage2_holdout_format_length.png')
ap.add_argument('--items-dir', default='logs', help='eval_items_*.jsonl 위치')
ap.add_argument('--train-csv', default='logs/train_metrics.csv')
ap.add_argument('--probe', default='logs/probe_fragility_results.jsonl')
a = ap.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for _f in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
    if any(_f == m.name for m in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break
matplotlib.rcParams['axes.unicode_minus'] = False

X = {'init': 0, 'step400': 400, 'step500': 500, 'step600': 600,
     'step700': 700, 'step850': 850, 'step900': 900, 'step1000': 1000}
ORDER = list(X)

# ---- 홀드아웃 집계 (n=1,772 전량만) ---------------------------------------
HO = {}
for p in glob.glob('logs/eval_midtrain_results*.jsonl'):
    for line in open(p):
        d = json.loads(line)
        if d.get('n') == 1772 and d['tag'] in X:
            HO[d['tag']] = d
seq = [t for t in ORDER if t in HO]
print('홀드아웃 지점:', seq)

# ---- 학습측 (25-step 이동평균) --------------------------------------------
tr = {}
with open(a.train_csv) as f:
    for row in csv.DictReader(f):
        try:
            tr[int(row['step'])] = row
        except (TypeError, ValueError):
            continue
tsteps = sorted(tr)


def tw(step, key, w=25):
    v = [float(tr[s][key]) for s in tsteps if step - w < s <= step
         and tr[s].get(key) not in (None, '')]
    return st.mean(v) if v else None


fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.4))
fig.subplots_adjust(top=.855, bottom=.075, left=.062, right=.955, hspace=.34, wspace=.30)
COL = {'deepvision': '#2E6FBF', 'mmk12': '#159947', 'pmcvqa': '#C43D3D'}


def mark_cliff(ax):
    ax.axvspan(850, 900, color='crimson', alpha=.10, zorder=0)


# --- ① 형식 두 잣대 --------------------------------------------------------
ax = axes[0][0]
xs = [X[t] for t in seq]
ax.plot(xs, [HO[t]['format'] for t in seq], 'o-', color='#111', lw=2.2, ms=6,
        label='홀드아웃 format (느슨: <answer> 존재)')
ft = [(s, tw(s, 'rewards/FormatThink/mean')) for s in range(25, max(tsteps) + 1, 5)]
ax.plot([s for s, v in ft if v is not None], [v for _, v in ft if v is not None],
        '-', color='#159947', lw=1.6, alpha=.85, label='학습 FormatThink (엄격, 25-step 평균)')
mark_cliff(ax)
ax.set_title('① 형식 — 학습 쪽만 회복한다', fontsize=12.5, weight='bold')
ax.set_ylabel('format score')
ax.set_ylim(-.03, 1.05)
ax.legend(fontsize=8.5, loc='lower left')
ax.annotate('학습 쪽은 800 까지 회복(0.965)\n홀드아웃은 850 까지 0.89 대 정체',
            xy=(800, .965), xytext=(430, .55), fontsize=8.5, color='#444',
            arrowprops=dict(arrowstyle='->', color='#888', lw=.9))

# --- ② 길이 ----------------------------------------------------------------
ax = axes[0][1]
ax.plot(xs, [HO[t]['mean_chars'] for t in seq], 'o-', color='#111', lw=2.2, ms=6,
        label='홀드아웃 mean_chars')
ax.set_ylabel('holdout mean_chars', color='#111')
ax2 = ax.twinx()
ml = [(s, tw(s, 'completions/mean_length')) for s in range(25, max(tsteps) + 1, 5)]
ax2.plot([s for s, v in ml if v is not None], [v for _, v in ml if v is not None],
         '-', color='#E07B20', lw=1.6, alpha=.85, label='학습 mean_length (tok)')
ax2.axhline(2048, color='crimson', ls=':', lw=1.4)
ax2.text(30, 2110, '2,048 tok = 홀드아웃 생성 상한 (학습은 6,144) — 오른쪽 토큰 축에서 읽을 것',
         fontsize=8, color='crimson')
ax2.set_ylabel('train mean_length (tok)', color='#E07B20')
mark_cliff(ax)
ax.set_title('② 길이 — RL 은 처음부터 끝까지 출력을 부풀렸다', fontsize=12.5, weight='bold')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper left')

# --- ③ 소스 안 층별 --------------------------------------------------------
ax = axes[1][0]
IT = {}
for t in seq:
    g = sorted(glob.glob(os.path.join(a.items_dir, f'eval_items_{t}_*.jsonl')))
    if g:
        IT[t] = [json.loads(l) for l in open(g[-1])]
if IT:
    keys = [('deepvision', 'vl', '-'), ('deepvision', 'math', '--'),
            ('mmk12', 'numeric', '-'), ('mmk12', 'symbolic', '--'),
            ('pmcvqa', 'letter', '-')]
    for src, strat, ls in keys:
        ys, n = [], 0
        for t in seq:
            v = [r['score'] for r in IT[t] if r['source'] == src and r['stratum'] == strat]
            n = len(v); ys.append(st.mean(v) * 100)
        ax.plot([X[t] for t in seq], ys, ls, color=COL[src], lw=2, marker='o', ms=4,
                label=f'{src}·{strat} (n={n})')
    ax.annotate('400 이후 오른 층은\ndeepvision·vl 하나뿐', xy=(850, 52.4), xytext=(430, 62),
                fontsize=8.5, color='#2E6FBF',
                arrowprops=dict(arrowstyle='->', color='#2E6FBF', lw=.9))
else:
    ax.text(.5, .5, 'eval_items_*.jsonl 없음', ha='center', transform=ax.transAxes)
mark_cliff(ax)
ax.set_title('③ 소스 안 층별 정확도', fontsize=12.5, weight='bold')
ax.set_ylabel('accuracy (%)')
ax.legend(fontsize=8, loc='lower left', ncol=2)

# --- ④ step 900 채점 구멍 --------------------------------------------------
ax = axes[1][1]
PB = {}
for line in open(a.probe):
    d = json.loads(line)
    if d['temp'] == 0.0:
        PB[d['tag']] = d
SRC = ['deepvision', 'mmk12', 'pmcvqa']
NAME = {'deepvision': 'deepvision\n(일반 VL)', 'mmk12': 'mmk12\n(수학)', 'pmcvqa': 'pmcvqa\n(의료)'}
if 'step900' in PB:
    xpos = range(len(SRC))
    fail = [PB['step900']['by_source'][s] * 100 for s in SRC]
    acc9 = [HO['step900']['per_source'][s] * 100 for s in SRC]
    acc8 = [HO['step850']['per_source'][s] * 100 for s in SRC]
    ax.bar([x - .27 for x in xpos], fail, .27, color='#999', label='형식 실패율 (프로브, greedy)')
    ax.bar([x + .0 for x in xpos], acc8, .27, color='#7FB3E8', label='정확도 @ step 850')
    ax.bar([x + .27 for x in xpos], acc9, .27, color='#C43D3D', label='정확도 @ step 900')
    for x, (f_, a8, a9) in enumerate(zip(fail, acc8, acc9)):
        ax.text(x - .27, f_ + 1.5, f'{f_:.0f}%', ha='center', fontsize=8.5, color='#555')
        ax.text(x + .27, a9 + 1.5, f'{a9:.1f}', ha='center', fontsize=8.5, color='#C43D3D')
    ax.set_xticks(list(xpos)); ax.set_xticklabels([NAME[s] for s in SRC], fontsize=9)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8.5, loc='upper right')
ax.set_title('④ step 900 — 형식은 균일하게 깨졌는데 점수는 갈린다',
             fontsize=12.5, weight='bold')
ax.set_ylabel('%')
ax.text(.02, .40, '채점기가 본문에서 숫자를 뽑으므로\n수학은 태그 없이도 50.0 을 받는다.\n'
                  '의료는 letter 정규식이 전체 일치를\n요구해 0 점 — 우연(25%) 보다 낮다.',
        transform=ax.transAxes, fontsize=8.5, color='#444')

for row in axes:
    for ax in row:
        ax.grid(alpha=.3)
        if ax is not axes[1][1]:
            ax.set_xlabel('학습 step')
            ax.set_xticks([X[t] for t in seq])
            ax.set_xticklabels([t.replace('step', '') or '0' for t in seq],
                               rotation=45, ha='right', rotation_mode='anchor', fontsize=8.5)

fig.suptitle('Stage-2 홀드아웃 — 형식 · 길이 · 층별 정확도 종합   '
             '[run 73924 · 전량 1,772 · greedy]', fontsize=14, weight='bold', y=.975)
fig.text(.5, .905,
         '주의: ①의 두 곡선은 수준 비교 불가 — 지표 정의(느슨/엄격) · temperature(0.0/0.9) · '
         '토큰 상한(2,048/6,144)이 모두 다르다. 갈라지는 시점만 읽을 것.',
         ha='center', fontsize=9.5, color='#B03030')
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
fig.savefig(a.out, dpi=145)
print('→', a.out)
