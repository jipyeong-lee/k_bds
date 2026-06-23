#!/usr/bin/env python3
"""두 GRPO 로그(baseline vs 파생기법)의 N-step 구간평균 추세를 오버레이 비교 PNG 생성.

사용: python scripts/plot_grpo_compare.py <logA> <labelA> <logB> <labelB> <out_png> [--bin 50] [--title "..."]
컨테이너 내 실행:
  singularity exec work/images/ms-swift-413-sandbox python scripts/plot_grpo_compare.py \
    logs/grpo_stage2_57249.log baseline logs/grpo_adv_57527.log DAPO docs/assets/grpo_dapo_vs_baseline.png
"""
import re, argparse, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('logA'); ap.add_argument('labelA')
ap.add_argument('logB'); ap.add_argument('labelB')
ap.add_argument('out')
ap.add_argument('--bin', type=int, default=50)
ap.add_argument('--title', default='GRPO A/B: baseline vs DAPO (per-segment mean)')
a = ap.parse_args()

KEYS = {
    'reward': 'reward', 'rewards/AccuracyMix/mean': 'acc',
    'rewards/FormatThink/mean': 'ft', 'completions/clipped_ratio': 'clip',
    'completions/mean_length': 'mlen', 'frac_reward_zero_std': 'zstd',
}

def load(path, binsz):
    seg = {}
    for line in open(path):
        m = re.search(r"global_step/max_steps': '(\d+)/", line)
        if not m:
            continue
        s = int(m.group(1)); b = (s - 1) // binsz * binsz + binsz
        d = seg.setdefault(b, {v: [] for v in KEYS.values()})
        for k, v in KEYS.items():
            mm = re.search(rf"'{re.escape(k)}': '([-\d.]+)'", line)
            if mm:
                d[v].append(float(mm.group(1)))
    xs = sorted(seg)
    avg = lambda b, v: st.mean(seg[b][v]) if seg[b][v] else float('nan')
    return xs, {v: [avg(b, v) for b in xs] for v in KEYS.values()}

xa, A = load(a.logA, a.bin)
xb, B = load(a.logB, a.bin)
print(f'{a.labelA}: {len(xa)} segs to step {xa[-1] if xa else 0} | {a.labelB}: {len(xb)} segs to step {xb[-1] if xb else 0}')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
C = {'reward': '#1f77b4', 'acc': '#d62728', 'ft': '#2ca02c', 'clip': '#ff7f0e', 'zstd': '#9467bd'}

# 상단: reward / Acc / FormatThink — baseline 실선, B 점선
for key, name, mk in [('reward', 'reward', 'o'), ('acc', 'AccuracyMix', 's'), ('ft', 'FormatThink', '^')]:
    ax1.plot(xa, A[key], mk + '-', color=C[key], lw=2, label=f'{name} ({a.labelA})')
    ax1.plot(xb, B[key], mk + '--', color=C[key], lw=2, alpha=.85, label=f'{name} ({a.labelB})')
ax1.set_ylabel('reward / accuracy / format')
ax1.set_title(a.title)
ax1.grid(alpha=.3); ax1.legend(loc='lower right', fontsize=8, ncol=2)

# 하단: clip & zero_std — zero_std 가 핵심(DAPO=0)
for key, name, mk in [('clip', 'clipped_ratio', 'o'), ('zstd', 'frac_reward_zero_std', 'd')]:
    ax2.plot(xa, A[key], mk + '-', color=C[key], lw=2, label=f'{name} ({a.labelA})')
    ax2.plot(xb, B[key], mk + '--', color=C[key], lw=2, alpha=.85, label=f'{name} ({a.labelB})')
ax2.set_ylabel('clip / zero_std ratio')
ax2.set_xlabel(f'step (per-{a.bin} segment mean)')
ax2.grid(alpha=.3); ax2.legend(loc='upper right', fontsize=8, ncol=2)

# DAPO 데이터 끝 지점 표시(비교 구간 한계 명시)
if xb:
    for ax in (ax1, ax2):
        ax.axvline(xb[-1], color='gray', ls=':', lw=1)
    ax1.annotate(f'{a.labelB} latest (step~{xb[-1]})', xy=(xb[-1], ax1.get_ylim()[0]),
                 xytext=(4, 6), textcoords='offset points', fontsize=8, color='gray')

fig.tight_layout()
fig.savefig(a.out, dpi=120)
print(f'saved {a.out}')
