#!/usr/bin/env python3
"""plot_sft_curve.py — Stage-1 v3(sft_mixed) 콜드스타트 SFT 학습곡선 → PNG.

사용: singularity exec $SB python scripts/plot_sft_curve.py docs/assets/sft_mixed_traincurve.png
데이터: work/data/sft_mixed_traincurve.json  (logs/sft_66255.log 에서 추출, scripts 상단 주석 참고)
관례: docs/assets/ 에 dpi=120 저장 (plot_grpo_* 와 동일).
"""
import sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'docs/assets/sft_mixed_traincurve.png'
DATA = sys.argv[2] if len(sys.argv) > 2 else 'work/data/sft_mixed_traincurve.json'
d = json.load(open(DATA))
steps, evals, MAX = d['steps'], d['evals'], 298

xs = [s['step'] for s in steps]
loss = [s['loss'] for s in steps]
tacc = [s['tok_acc'] for s in steps]
ex = [round(e['epoch'] / 2 * MAX) for e in evals]      # epoch1→149, epoch2→298
eloss = [e['eval_loss'] for e in evals]
etacc = [e['eval_tok_acc'] for e in evals]

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.30,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
EP = '#94a3b8'

# ── (좌) Loss ────────────────────────────────────────────────────────────
ax1.plot(xs, loss, color='#2563eb', lw=1.7, label='train loss', alpha=0.9)
ax1.plot(ex, eloss, color='#dc2626', lw=1.4, ls='--', marker='o', ms=9,
         zorder=5, label='eval loss (epoch end)')
for x, y in zip(ex, eloss):
    ax1.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(7, 9),
                 color='#dc2626', fontsize=10, fontweight='bold')
ax1.axvline(149, color=EP, ls=':', lw=1.1)
ax1.text(153, max(loss) * 0.98, 'epoch1 | epoch2', color=EP, fontsize=8.5, va='top')
ax1.set_xlabel('step'); ax1.set_ylabel('loss'); ax1.set_xlim(0, MAX + 4)
ax1.set_title('Loss   ·   train 1.12→0.60,  eval 0.679→0.668', fontsize=11)
ax1.legend(loc='upper right', framealpha=0.9)

# ── (우) Token accuracy ──────────────────────────────────────────────────
ax2.plot(xs, tacc, color='#16a34a', lw=1.7, label='train token_acc', alpha=0.9)
ax2.plot(ex, etacc, color='#dc2626', lw=1.4, ls='--', marker='o', ms=9,
         zorder=5, label='eval token_acc')
for x, y in zip(ex, etacc):
    ax2.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(7, -15),
                 color='#dc2626', fontsize=10, fontweight='bold')
ax2.axvline(149, color=EP, ls=':', lw=1.1)
ax2.set_xlabel('step'); ax2.set_ylabel('token accuracy'); ax2.set_xlim(0, MAX + 4)
ax2.set_title('Token accuracy   ·   train→0.81,  eval 0.791→0.794', fontsize=11)
ax2.legend(loc='lower right', framealpha=0.9)

fig.suptitle('Stage-1 v3 Cold-Start SFT — Qwen3.5-9B LoRA · 9,507 samples · 2 epochs '
             '(job 66255, 41 min · grad_norm 0.29 stable · no overfit)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=120, bbox_inches='tight')
print('saved', OUT)
