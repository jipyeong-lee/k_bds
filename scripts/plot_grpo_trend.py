#!/usr/bin/env python3
"""GRPO 학습 로그에서 N-step 구간평균 추세를 추출해 PNG plot 생성.

사용: python scripts/plot_grpo_trend.py <log_path> <out_png> [--bin 100] [--title "..."]
컨테이너 내 실행(matplotlib 내장):
  singularity exec work/images/ms-swift-413-sandbox python scripts/plot_grpo_trend.py ...
"""
import re, sys, argparse, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('log'); ap.add_argument('out')
ap.add_argument('--bin', type=int, default=100)
ap.add_argument('--title', default='GRPO training trend')
a = ap.parse_args()

KEYS = {
    'reward': 'reward', 'rewards/AccuracyMix/mean': 'acc',
    'rewards/FormatThink/mean': 'ft', 'completions/clipped_ratio': 'clip',
    'completions/mean_length': 'mlen', 'frac_reward_zero_std': 'zstd',
}
seg = {}
for line in open(a.log):
    m = re.search(r"global_step/max_steps': '(\d+)/", line)
    if not m:
        continue
    s = int(m.group(1)); b = (s - 1) // a.bin * a.bin + a.bin  # bin 우측 경계(=100,200,..)
    d = seg.setdefault(b, {v: [] for v in KEYS.values()})
    for k, v in KEYS.items():
        mm = re.search(rf"'{re.escape(k)}': '([-\d.]+)'", line)
        if mm:
            d[v].append(float(mm.group(1)))

xs = sorted(seg)
avg = lambda b, v: st.mean(seg[b][v]) if seg[b][v] else float('nan')
reward = [avg(b, 'reward') for b in xs]
acc = [avg(b, 'acc') for b in xs]
ft = [avg(b, 'ft') for b in xs]
clip = [avg(b, 'clip') for b in xs]
zstd = [avg(b, 'zstd') for b in xs]
mlen = [avg(b, 'mlen') for b in xs]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# 상단: reward / accuracy / format (0~1 보상 지표)
ax1.plot(xs, reward, 'o-', label='reward', color='#1f77b4', lw=2)
ax1.plot(xs, acc, 's-', label='AccuracyMix', color='#d62728', lw=2)
ax1.plot(xs, ft, '^-', label='FormatThink', color='#2ca02c', lw=2)
ax1.set_ylabel('reward / accuracy / format')
ax1.set_title(a.title)
ax1.grid(alpha=.3); ax1.legend(loc='lower right')

# 하단: clip & zero_std (좌축) + mean_len (우축)
ax2.plot(xs, clip, 'o-', label='clipped_ratio', color='#ff7f0e', lw=2)
ax2.plot(xs, zstd, 'd-', label='frac_reward_zero_std', color='#9467bd', lw=2)
ax2.set_ylabel('clip / zero_std ratio'); ax2.set_xlabel(f'step (per-{a.bin} segment mean)')
ax2.grid(alpha=.3)
ax2b = ax2.twinx()
ax2b.plot(xs, mlen, 'x--', label='mean_length', color='#8c564b', lw=1.5)
ax2b.set_ylabel('completion mean_length')
l1, lab1 = ax2.get_legend_handles_labels()
l2, lab2 = ax2b.get_legend_handles_labels()
ax2.legend(l1 + l2, lab1 + lab2, loc='upper right')

fig.tight_layout()
fig.savefig(a.out, dpi=120)
print(f'saved {a.out} ({len(xs)} points: {xs})')
