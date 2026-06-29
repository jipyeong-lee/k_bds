#!/usr/bin/env python3
"""plot_grpo_all.py — Stage-2 3기법(baseline/DAPO/dr_grpo) 통합 비교 plot 1장.

핵심 지표 2x2(Acc·reward·mean_len·zero_std)를 한 그림에 오버레이 → Stage-2 전체 서사를 단일 figure로.
사용:
  singularity exec work/images/ms-swift-413-sandbox python scripts/plot_grpo_all.py \
    logs/grpo_stage2_57249.log logs/grpo_adv_57527.log logs/grpo_adv_57624.log \
    docs/assets/grpo_stage2_all.png
"""
import re, sys, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE, DAPO, DRGRPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BIN = 50
KEYS = {'rewards/AccuracyMix/mean': 'acc', 'reward': 'reward',
        'completions/mean_length': 'mlen', 'frac_reward_zero_std': 'zstd'}


def load(path):
    seg = {}
    for line in open(path):
        m = re.search(r"global_step/max_steps': '(\d+)/", line)
        if not m:
            continue
        s = int(m.group(1)); b = (s - 1) // BIN * BIN + BIN
        d = seg.setdefault(b, {v: [] for v in KEYS.values()})
        for k, v in KEYS.items():
            mm = re.search(rf"'{re.escape(k)}': '([-\d.]+)'", line)
            if mm:
                d[v].append(float(mm.group(1)))
    xs = sorted(seg)
    avg = lambda b, v: (st.mean(seg[b][v]) if seg[b][v] else float('nan'))
    return xs, {v: [avg(b, v) for b in xs] for v in KEYS.values()}

runs = [('baseline', '#1f77b4', '-', load(BASE)),
        ('DAPO', '#ff7f0e', '--', load(DAPO)),
        ('dr_grpo', '#2ca02c', '-.', load(DRGRPO))]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [('acc', 'AccuracyMix (accuracy)', axes[0, 0]),
          ('reward', 'reward', axes[0, 1]),
          ('mlen', 'completion mean_length', axes[1, 0]),
          ('zstd', 'frac_reward_zero_std', axes[1, 1])]
for key, title, ax in panels:
    for label, color, ls, (xs, D) in runs:
        ax.plot(xs, D[key], ls, color=color, lw=2, marker='o', markersize=3, label=label, alpha=.9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('step (per-50 segment mean)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
# Acc 패널에 baseline 0.50 정점 가이드선 + 돌파구간 강조
axes[0, 0].axhline(0.50, color='gray', ls=':', lw=1, alpha=.6)
axes[0, 0].axvspan(500, 600, color='yellow', alpha=.10)

fig.suptitle('Stage-2 GRPO recipes: baseline vs DAPO vs dr_grpo  '
             '(only dr_grpo breaks Acc 0.50 at step 501-600 while suppressing length)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=120)
print(f'saved {OUT}')
