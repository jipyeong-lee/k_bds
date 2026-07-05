#!/usr/bin/env python3
"""plot_grpo_multi.py — Stage-2 GRPO 계열 N개 기법 성능 비교 plot(2x3 패널).

각 run 을 `label:color:style:logpath` 로 넘기면 6개 핵심지표(Acc·reward·FormatThink·
mean_len·zero_std·clip)를 50-step 구간평균으로 오버레이. 기법 수 무관(일반화).

사용:
  singularity exec work/images/ms-swift-413-sandbox python scripts/plot_grpo_multi.py \
    OUT.png  "baseline:#1f77b4:-:logs/a.log"  "dr_grpo:#2ca02c:-.:logs/b.log" ...
  (첫 인자 = 출력 png, 이후 = run 스펙들)
"""
import re, sys, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
SPECS = sys.argv[2:]
BIN = 50
KEYS = {'rewards/AccuracyMix/mean': 'acc', 'reward': 'reward',
        'rewards/FormatThink/mean': 'fmt', 'completions/mean_length': 'mlen',
        'frac_reward_zero_std': 'zstd', 'clip_ratio/region_mean': 'clip'}


def load(path):
    seg = {}
    for line in open(path, errors='ignore'):
        m = re.search(r"global_step/max_steps': '(\d+)/", line)
        if not m:
            continue
        s = int(m.group(1)); b = (s - 1) // BIN * BIN + BIN
        d = seg.setdefault(b, {v: [] for v in KEYS.values()})
        for k, v in KEYS.items():
            mm = re.search(rf"'{re.escape(k)}': '([-\d.eE]+)'", line)
            if mm:
                d[v].append(float(mm.group(1)))
    xs = sorted(seg)
    avg = lambda b, v: (st.mean(seg[b][v]) if seg[b][v] else float('nan'))
    return xs, {v: [avg(b, v) for b in xs] for v in KEYS.values()}


runs = []
for spec in SPECS:
    label, color, ls, path = spec.split(':', 3)
    runs.append((label, color, ls, load(path)))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panels = [('acc', 'AccuracyMix (accuracy)', axes[0, 0]),
          ('reward', 'reward (weighted total)', axes[0, 1]),
          ('fmt', 'FormatThink reward', axes[0, 2]),
          ('mlen', 'completion mean_length', axes[1, 0]),
          ('zstd', 'frac_reward_zero_std', axes[1, 1]),
          ('clip', 'clip_ratio/region_mean', axes[1, 2])]
for key, title, ax in panels:
    for label, color, ls, (xs, D) in runs:
        ax.plot(xs, D[key], ls, color=color, lw=2, marker='o', markersize=3, label=label, alpha=.9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('step (per-50 segment mean)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
# Acc 패널: 0.50 정점 가이드선 + 판정창(501~600) 강조
axes[0, 0].axhline(0.50, color='gray', ls=':', lw=1, alpha=.6)
axes[0, 0].axvspan(500, 600, color='yellow', alpha=.12)

labels = ' vs '.join(r[0] for r in runs)
fig.suptitle(f'Stage-2 GRPO recipes — {labels}  (50-step segment mean; yellow band = decision window 501-600)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=120)
print(f'saved {OUT}  ({len(runs)} runs)')
