#!/usr/bin/env python3
"""plot_eval_trend.py — 중간 홀드아웃 평가 결과(jsonl) → step별 정확도 추세 그림.

사용:
    ./bin/python scripts/plot_eval_trend.py logs/eval_midtrain_results_*.jsonl \
        --points init:0,step400:400,step500:500,trained:600 \
        --base base -o docs/assets/stage2_holdout_trend.png

  * eval_midtrain.slurm 이 남기는 결과 jsonl 을 여러 개 받아 tag 로 합친다.
  * --points 로 tag → step 을 지정한다(init 은 RL 0% = step 0).
  * --base 로 지정한 tag 는 점이 아니라 가로 기준선으로 그린다.
  * 오차막대 = 95% 정규근사 CI. n 이 작아 구간이 넓다는 사실 자체가 메시지다.
  * matplotlib 은 loader python 에만 있다 → `./bin/python` 으로 실행할 것.
"""
import argparse
import glob
import json
import math
from pathlib import Path

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE_C = "#e1e0d9", "#c3c2b7"
SRC_COLOR = {"deepvision": BLUE, "mmk12": ORANGE, "pmcvqa": AQUA}
SRC_LABEL_KO = {"deepvision": "deepvision (일반)", "mmk12": "mmk12 (math)", "pmcvqa": "pmcvqa (의료)"}
SRC_LABEL_EN = {"deepvision": "deepvision (general)", "mmk12": "mmk12 (math)", "pmcvqa": "pmcvqa (medical)"}


def ci95(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n) if n else 0.0


def load(paths):
    rows = {}
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            for line in Path(f).read_text().splitlines():
                line = line.strip()
                if line:
                    d = json.loads(line)
                    rows[d["tag"]] = d
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="eval_midtrain 결과 jsonl (glob 가능)")
    ap.add_argument("--points", required=True, help="tag:step 쉼표 목록 (예 init:0,step400:400)")
    ap.add_argument("--base", help="가로 기준선으로 그릴 tag (예 base)")
    ap.add_argument("-o", "--out", default="docs/assets/stage2_holdout_trend.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    rows = load(args.results)
    pts = []
    for item in args.points.split(","):
        tag, step = item.split(":")
        if tag not in rows:
            raise SystemExit(f"결과에 tag '{tag}' 가 없다. 있는 tag: {sorted(rows)}")
        pts.append((int(step), tag, rows[tag]))
    pts.sort()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.ticker import FuncFormatter

    have = {f.name for f in fm.fontManager.ttflist}
    font = next((c for c in ("Pretendard", "NanumGothic", "Nanum Gothic",
                             "Noto Sans CJK KR", "AppleGothic") if c in have), None)
    ko = font is not None
    SRC_LABEL = SRC_LABEL_KO if ko else SRC_LABEL_EN
    T = {
        "title": args.title or ("Stage-2 홀드아웃 정확도 추세" if ko else "Stage-2 holdout accuracy trend"),
        "left": "전체 (n=300)" if ko else "Overall (n=300)",
        "right": "소스별 (각 n=100)" if ko else "By source (n=100 each)",
        "x": "학습 step" if ko else "training step",
        "err": "오차막대 = 95% CI" if ko else "error bars = 95% CI",
        "base": "base(RL·SFT 전)" if ko else "base (no SFT/RL)",
        "init": "init = RL 0%" if ko else "init = RL 0%",
    }
    plt.rcParams.update({
        **({"font.family": font} if font else {}),
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE_C, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titlesize": 12,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.2))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.065, right=0.985, wspace=0.20)
    xs = [s for s, _, _ in pts]
    pad = (max(xs) - min(xs)) * 0.16 or 60

    def frame(ax, title, sub=None):
        ax.set_title(title, color=INK, fontweight="600", loc="left", pad=14 if sub else 8)
        if sub:
            ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5,
                    va="bottom", ha="left")
        ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel(T["x"], color=MUTED, fontsize=9.5)
        ax.set_xticks(xs)
        ax.set_xlim(min(xs) - pad * 0.5, max(xs) + pad)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    # ── 좌: 전체 ──────────────────────────────────────────────────────────
    frame(axL, T["left"], T["err"])
    ys = [d["accuracy"] for _, _, d in pts]
    es = [ci95(d["accuracy"], d["n"]) for _, _, d in pts]
    axL.errorbar(xs, ys, yerr=es, color=BLUE, lw=2.0, marker="o", ms=7,
                 capsize=4, elinewidth=1.2, zorder=3, solid_capstyle="round")
    for x, y in zip(xs, ys):
        axL.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 12),
                     ha="center", color=BLUE, fontsize=9.5, fontweight="600", zorder=4)
    if args.base and args.base in rows:
        b = rows[args.base]["accuracy"]
        axL.axhline(b, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        axL.text(max(xs) + pad * 0.05, b, f"{T['base']} {b:.3f}", color=MUTED,
                 fontsize=9, va="bottom")
    lo = min(min(y - e for y, e in zip(ys, es)), rows[args.base]["accuracy"] if args.base in rows else 1)
    axL.set_ylim(lo - 0.03, max(y + e for y, e in zip(ys, es)) + 0.045)

    # ── 우: 소스별 ────────────────────────────────────────────────────────
    frame(axR, T["right"], T["err"])
    for src in ("deepvision", "mmk12", "pmcvqa"):
        sy = [d["per_source"][src] for _, _, d in pts]
        se = [ci95(v, d["n"] // len(d["per_source"])) for v, (_, _, d) in zip(sy, pts)]
        c = SRC_COLOR[src]
        axR.errorbar(xs, sy, yerr=se, color=c, lw=2.0, marker="o", ms=6,
                     capsize=3, elinewidth=1.0, zorder=3, label=SRC_LABEL[src])
        axR.annotate(f"{sy[-1]:.2f}", (xs[-1], sy[-1]), textcoords="offset points",
                     xytext=(9, -3), color=c, fontsize=9.5, fontweight="600", zorder=4)
    leg = axR.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE,
                     edgecolor="none", framealpha=0.92, labelcolor=INK2, handlelength=1.6)
    leg.set_zorder(5)

    for ax in (axL, axR):
        ax.axvline(0, color=BASE_C, lw=0.8, zorder=1)

    fig.suptitle(T["title"], x=0.065, y=0.955, ha="left", color=INK,
                 fontsize=16, fontweight="700")
    fig.text(0.065, 0.885, T["init"] + " · " +
             ("동일 슬라이스·greedy·소스별 층화 표본" if ko else "same slice, greedy, source-stratified sample"),
             ha="left", color=INK2, fontsize=10)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190)
    print(f"[trend] saved: {args.out}")
    for s, tag, d in pts:
        print(f"  step {s:>4}  {tag:<9} acc {d['accuracy']:.4f} ±{ci95(d['accuracy'], d['n'])*100:.1f}pp"
              f"  {d['per_source']}")


if __name__ == "__main__":
    main()
