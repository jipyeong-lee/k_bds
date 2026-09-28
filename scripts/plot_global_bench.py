#!/usr/bin/env python3
"""plot_global_bench.py — 글로벌 벤치마크(MedXpertQA-MM+MMMU-H&M) SFT vs RL 막대비교.

과제계획서 약속 지표. eval_global_bench.py 가 남긴 문항별 채점(eval_items_gbench_*.jsonl)에서
전체·소스별 accuracy 와 95% CI(정규근사)를 계산해 그룹 막대로 그린다.

사용: python3 scripts/plot_global_bench.py \
        logs/eval_items_gbench_sft_77389.jsonl logs/eval_items_gbench_rl_77390.jsonl \
        -o docs/assets/stage2_global_bench.png
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument('items', nargs=2, help='SFT jsonl, RL jsonl (이 순서)')
ap.add_argument('-o', '--out', default='docs/assets/stage2_global_bench.png')
a = ap.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for _f in ('AppleGothic', 'NanumGothic', 'Malgun Gothic'):
    if any(_f == m.name for m in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _f
        break
matplotlib.rcParams['axes.unicode_minus'] = False


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n) if n else 0.0


def load(path):
    by_source = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        by_source.setdefault(d['source'], []).append(d['score'])
        by_source.setdefault('전체', []).append(d['score'])
    return {k: (sum(v) / len(v), len(v)) for k, v in by_source.items()}


sft = load(a.items[0])
rl = load(a.items[1])
groups = ['전체'] + sorted(k for k in sft if k != '전체')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'

fig, ax = plt.subplots(figsize=(7.5, 5))
x = range(len(groups))
w = 0.32
sft_y = [sft[g][0] for g in groups]
sft_e = [ci95(*sft[g]) for g in groups]
rl_y = [rl[g][0] for g in groups]
rl_e = [ci95(*rl[g]) for g in groups]

ax.bar([i - w / 2 for i in x], sft_y, w, yerr=sft_e, capsize=4, color=BLUE, label='SFT (init)')
ax.bar([i + w / 2 for i in x], rl_y, w, yerr=rl_e, capsize=4, color=ORANGE, label='RL (pmcvqa checkpoint-4896)')

for i, g in enumerate(groups):
    ax.text(i - w / 2, sft_y[i] + sft_e[i] + .01, f'{sft_y[i]*100:.1f}%', ha='center', fontsize=9, color=INK)
    ax.text(i + w / 2, rl_y[i] + rl_e[i] + .01, f'{rl_y[i]*100:.1f}%', ha='center', fontsize=9, color=INK)
    ax.text(i, -0.07, f'n={sft[g][1]}', ha='center', fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())

ax.set_xticks(list(x))
ax.set_xticklabels(groups)
ax.set_ylabel('accuracy')
ax.set_ylim(0, max(sft_y + rl_y) + 0.15)
ax.set_title('글로벌 벤치마크(MedXpertQA-MM + MMMU-H&M) — SFT vs RL')
ax.grid(axis='y', alpha=.3, color=GRID)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(f'저장: {a.out}')
for g in groups:
    print(f'  {g:14} SFT {sft[g][0]*100:.2f}% (n={sft[g][1]}) -> RL {rl[g][0]*100:.2f}% (n={rl[g][1]})  Δ{(rl[g][0]-sft[g][0])*100:+.2f}pp')
