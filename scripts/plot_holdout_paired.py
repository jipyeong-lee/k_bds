#!/usr/bin/env python3
"""plot_holdout_paired.py — 전량 홀드아웃(1,772) 문항별 결과 → 궤적 + 짝지음 Δ 그림.

왜 새로 쓰나 (plot_eval_trend.py 를 안 쓰는 이유):
  1) 그 스크립트는 소스별 n 을 `d["n"] // 3` 으로 잡는다. 균등 층화 표본에서는 맞지만
     전량 홀드아웃은 972/400/400 이라 deepvision 오차막대가 틀린다.
  2) 집계 jsonl 만 읽어서 **비짝지음** CI 만 그릴 수 있다. 같은 문항을 푼 두 모델의
     차이는 짝지음이 맞고, 그래야 검출 하한이 ±9pp 가 아니라 ±2pp 가 된다.
     점별 오차막대가 겹친다고 "차이 없다"고 읽으면 오독이다 — 그래서 Δ 패널을 따로 둔다.

입력은 eval_compare.py 가 EVAL_ITEMS 로 떨군 문항별 jsonl 들. tag 에서 step 을 읽는다
(init→0, stepNNN→NNN). matplotlib 외에는 표준 라이브러리만 쓴다.

  ./bin/python scripts/plot_holdout_paired.py logs/eval_items_*.jsonl \
      -o docs/assets/stage2_holdout_paired.png
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE_C, RED = "#e1e0d9", "#c3c2b7", "#c0392b"
SRC_COLOR = {"deepvision": BLUE, "mmk12": ORANGE, "pmcvqa": AQUA}
SRC_KO = {"deepvision": "deepvision (일반)", "mmk12": "mmk12 (수학)", "pmcvqa": "pmcvqa (의료)"}
SRC_EN = {"deepvision": "deepvision (general)", "mmk12": "mmk12 (math)", "pmcvqa": "pmcvqa (medical)"}
SRCS = ("deepvision", "mmk12", "pmcvqa")

# 사전등록 판정선 — docs/stage2_run73924_progress.md §7-2b
GATE_GO, GATE_STOP = 3.0, -1.0


def step_of(tag):
    if tag == "init":
        return 0
    m = re.fullmatch(r"step(\d+)", tag)
    if not m:
        raise SystemExit(f"tag '{tag}' 에서 step 을 못 읽었다 (init 또는 stepNNN 이어야 한다)")
    return int(m.group(1))


def load(path):
    rows, tag = {}, None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["item_id"]] = r
        tag = r["tag"]
    return tag, rows


def ci95(p, n):
    """비짝지음 95% CI (정규근사). 점 하나의 절대 위치 불확실성."""
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n) * 100 if n else 0.0


def mcnemar(pairs):
    """짝지음 Δ(pp), 95% CI 반폭(pp), 정확검정 p. eval_paired.py 와 같은 식."""
    n = len(pairs)
    if not n:
        return 0.0, float("nan"), 1.0
    b = sum(1 for a, x in pairs if a >= 0.5 > x)   # A만 정답
    c = sum(1 for a, x in pairs if x >= 0.5 > a)   # B만 정답
    delta = (c - b) / n * 100
    d = b + c
    if d == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / (2 ** d))
    return delta, 1.96 * math.sqrt(d) / n * 100, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="eval_items_*.jsonl (glob 가능)")
    ap.add_argument("-o", "--out", default="docs/assets/stage2_holdout_paired.png")
    ap.add_argument("--gate-base", default="step400",
                    help="사전등록 게이트의 기준 tag (기본 step400)")
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="", help="캡션 끝에 덧붙일 문구 (예: 붕괴·취소 사실)")
    args = ap.parse_args()

    runs = {}
    for pat in args.items:
        for f in sorted(glob.glob(pat)):
            tag, rows = load(f)
            if tag in runs:
                continue          # 같은 tag 가 두 번 들어오면 먼저 것만
            runs[tag] = rows
    if len(runs) < 2:
        raise SystemExit(f"최소 2개 tag 가 필요하다. 읽은 것: {sorted(runs)}")

    tags = sorted(runs, key=step_of)
    steps = [step_of(t) for t in tags]
    # 모든 런에 공통으로 있는 문항만 쓴다 — 그래야 점끼리 같은 모집단이 된다.
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    n_all = len(common)
    by_src = defaultdict(list)
    for k in common:
        by_src[runs[tags[0]][k]["source"]].append(k)
    src_n = {s: len(by_src[s]) for s in SRCS if by_src[s]}

    def acc(tag, keys):
        return sum(runs[tag][k]["score"] for k in keys) / len(keys) * 100

    def pairs(a, b, keys):
        return [(runs[a][k]["score"], runs[b][k]["score"]) for k in keys]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.ticker import FuncFormatter

    have = {f.name for f in fm.fontManager.ttflist}
    font = next((c for c in ("Pretendard", "NanumGothic", "Nanum Gothic",
                             "Noto Sans CJK KR", "AppleGothic") if c in have), None)
    ko = font is not None
    SRC_LABEL = SRC_KO if ko else SRC_EN
    T = {
        "t1": "정확도 궤적" if ko else "Accuracy trajectory",
        "s1": (f"오차막대 = 95% CI(비짝지음) · 전체 n={n_all:,}" if ko
               else f"error bars = 95% CI (unpaired) · overall n={n_all:,}"),
        "t2": "init 대비 짝지음 Δ — RL 순효과" if ko else "Paired Δ vs init — net RL effect",
        "t3": (f"{args.gate_base} 대비 짝지음 Δ — 사전등록 판정축" if ko
               else f"Paired Δ vs {args.gate_base} — pre-registered gate"),
        "s23": ("오차막대 = McNemar 95% CI · ● 채움 = p<0.05" if ko
                else "error bars = McNemar 95% CI · filled = p<0.05"),
        "x": "학습 step" if ko else "training step",
        "y1": "홀드아웃 정확도 (%)" if ko else "holdout accuracy (%)",
        "y2": "Δ 정확도 (pp)" if ko else "Δ accuracy (pp)",
        "all": "전체" if ko else "overall",
        "go": "완주 기준 +3pp" if ko else "continue: +3pp",
        "stop": "즉시중단 −1pp" if ko else "abort: -1pp",
    }
    plt.rcParams.update({
        **({"font.family": font} if font else {}),
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE_C, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titlesize": 11.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.6))
    fig.subplots_adjust(top=0.695, bottom=0.115, left=0.052, right=0.988, wspace=0.215)
    axL, axM, axR = axes
    pad = (max(steps) - min(steps)) * 0.14 or 60

    def frame(ax, title, sub, ylab):
        ax.set_title(title, color=INK, fontweight="600", loc="left", pad=15)
        ax.text(0, 1.025, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5,
                va="bottom", ha="left")
        ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel(T["x"], color=MUTED, fontsize=9.5)
        ax.set_ylabel(ylab, color=MUTED, fontsize=9.5)
        ax.set_xticks(steps)
        ax.set_xlim(min(steps) - pad * 0.55, max(steps) + pad)
        ax.tick_params(axis="y", length=0)

    # ── 좌: 정확도 궤적 ───────────────────────────────────────────────────
    frame(axL, T["t1"], T["s1"], T["y1"])
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ys = [acc(t, common) for t in tags]
    es = [ci95(y / 100, n_all) for y in ys]
    handles = []
    handles.append(axL.errorbar(steps, ys, yerr=es, color=INK, lw=2.4, marker="o", ms=7,
                                capsize=4, elinewidth=1.2, zorder=4,
                                label=f"{T['all']} (n={n_all:,})"))
    # 값 라벨은 가로로 비켜 놓는다. 세로 오프셋으로는 오차막대(±2pp ≈ 50px)를 못 넘는다.
    # 위/아래는 오른쪽 선분이 올라가면 아래, 내려가면 위 — 선분을 피하는 쪽으로.
    for i, (x, y) in enumerate(zip(steps, ys)):
        up = i + 1 < len(ys) and ys[i + 1] > y
        axL.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(10, -6 if up else 6), ha="left",
                     va="top" if up else "bottom",
                     color=INK, fontsize=9, fontweight="600", zorder=6)
    for s in src_n:
        sy = [acc(t, by_src[s]) for t in tags]
        se = [ci95(v / 100, src_n[s]) for v in sy]
        handles.append(axL.errorbar(steps, sy, yerr=se, color=SRC_COLOR[s], lw=1.7,
                                    marker="o", ms=5, capsize=3, elinewidth=0.9,
                                    alpha=0.9, zorder=3,
                                    label=f"{SRC_LABEL[s]} (n={src_n[s]:,})"))

    # ── 중·우: 짝지음 Δ ───────────────────────────────────────────────────
    def delta_panel(ax, base, title, gate=False):
        frame(ax, title, T["s23"], T["y2"])
        ax.axhline(0, color=INK, lw=1.0, zorder=2)
        b0 = step_of(base)
        # 게이트 패널은 "기준 이후로 더 오르나"를 묻는 축이다. 기준보다 앞선 점(init)을
        # 넣으면 −5.7pp 가 y 범위를 늘려 정작 판정 구간(−1~+3pp)이 납작해진다.
        shown = [t for t in tags if not gate or step_of(t) >= b0]
        xs = [step_of(t) for t in shown]
        series = [("__all__", common, INK, f"{T['all']} (n={n_all:,})")] + \
                 [(s, by_src[s], SRC_COLOR[s], f"{SRC_LABEL[s]} (n={src_n[s]:,})") for s in src_n]
        for key, keys, color, lab in series:
            d, e, sig = [], [], []
            for t in shown:
                if t == base:          # 기준 자신은 정의상 Δ=0, 불확실성 없음
                    d.append(0.0); e.append(0.0); sig.append(None); continue
                dd, ee, pp = mcnemar(pairs(base, t, keys))
                d.append(dd); e.append(ee); sig.append(pp < 0.05)
            main = key == "__all__"
            ax.errorbar(xs, d, yerr=e, color=color, lw=2.4 if main else 1.7,
                        marker="", capsize=4 if main else 3,
                        elinewidth=1.2 if main else 0.9, alpha=1.0 if main else 0.9,
                        zorder=4 if main else 3, label=lab)
            # 채운 점 = 유의, 빈 점 = 미검출. 색만으로 구분하지 않는다.
            # 기준점은 측정값이 아니므로 작은 회색 점으로 따로 표시한다.
            for x, v, sg in zip(xs, d, sig):
                if sg is None:
                    ax.plot(x, v, marker="o", ms=5, color=MUTED, mfc=SURFACE,
                            mew=1.4, zorder=5)
                    continue
                ax.plot(x, v, marker="o", ms=7 if main else 5.5, color=color,
                        mfc=color if sg else SURFACE, mew=1.6,
                        zorder=5 if main else 4)
        if gate:
            # 기준 이전 점을 뺐으므로 x 범위도 좁힌다 — 안 그러면 왼쪽 절반이 빈다.
            ax.set_xticks(xs)
            ax.set_xlim(min(xs) - pad * 0.75, max(xs) + pad * 0.55)
            # 판정선(+3 / −1)은 4pp 밖에 안 떨어져 있다. 붕괴 지점이 들어와 y 범위가
            # 수십 pp 로 벌어지면 두 라벨이 같은 높이에 겹치므로 좌우로 갈라 놓는다.
            for v, lab, c, x, ha in ((GATE_GO, T["go"], AQUA, 0.008, "left"),
                                     (GATE_STOP, T["stop"], RED, 0.992, "right")):
                ax.axhline(v, color=c, lw=1.1, ls=(0, (5, 3)), zorder=2)
                ax.text(x, v, f" {lab} ", color=c, fontsize=8.5, va="bottom",
                        ha=ha, zorder=6, transform=ax.get_yaxis_transform())
            ax.margins(y=0.24)

    delta_panel(axM, "init", T["t2"])
    if args.gate_base in runs:
        delta_panel(axR, args.gate_base, T["t3"], gate=True)
    else:
        axR.set_visible(False)

    # 세 패널이 같은 4개 계열을 쓴다 → 범례는 그림 전체에 하나만. 패널마다 박스를
    # 두면 부제와 데이터를 가린다.
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="upper left", bbox_to_anchor=(0.052, 0.862), ncol=4, fontsize=9.5,
               frameon=False, labelcolor=INK2, handlelength=1.6, columnspacing=2.4)

    fig.suptitle(args.title or ("Stage-2 홀드아웃 전량 평가" if ko
                                else "Stage-2 full holdout evaluation"),
                 x=0.052, y=0.968, ha="left", color=INK, fontsize=16, fontweight="700")
    cap = (f"동일 {n_all:,}문항·greedy·같은 프롬프트 · init = RL 0% · "
           "점별 CI 는 겹쳐도 짝지음 Δ 는 갈릴 수 있다(오른쪽 두 패널이 판정축)" if ko else
           f"same {n_all:,} items, greedy · init = RL 0% · "
           "overlapping point CIs do not imply no difference — see paired panels")
    if args.note:
        cap += " · " + args.note
    fig.text(0.052, 0.897, cap, ha="left", color=INK2, fontsize=9.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190)
    print(f"[paired] saved: {args.out}")
    print(f"[paired] 공통 문항 {n_all:,}  소스별 {src_n}")
    for t in tags:
        line = f"  step {step_of(t):>4}  {t:<9} 전체 {acc(t, common):6.2f}%"
        for s in src_n:
            line += f"  {s} {acc(t, by_src[s]):6.2f}%"
        print(line)
    for base in ("init", args.gate_base):
        if base not in runs:
            continue
        print(f"\n  ── {base} 대비 짝지음 Δ ──")
        for t in tags:
            if t == base:
                continue
            dd, ee, pp = mcnemar(pairs(base, t, common))
            print(f"    {t:<9} 전체 Δ={dd:+.2f}pp ±{ee:.2f}  p={pp:.4f}"
                  f"{'  유의' if pp < 0.05 else '  미검출'}")


if __name__ == "__main__":
    main()
