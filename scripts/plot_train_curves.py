#!/usr/bin/env python3
"""plot_train_curves.py — GRPO 학습 로그 → step별 지표 CSV + 6패널 학습 곡선.

사용:
    ./bin/python scripts/plot_train_curves.py logs/grpo_adv_73924.log
    ./bin/python scripts/plot_train_curves.py logs/grpo_adv_739*.log \
        -o docs/assets/stage2_expanded_73924_curves.png --csv logs/train_metrics.csv

  * 로그를 여러 개 주면 step 기준으로 병합한다(체인 잡 73924→73925→… 이어보기).
    같은 step 이 중복되면 나중 파일 값이 이긴다(resume 구간).
  * matplotlib 은 loader python 에만 있다 → 반드시 `./bin/python` 으로 실행할 것
    (호스트 python3 에는 없음). `--no-plot` 이면 CSV·요약만 내고 matplotlib 불필요.
  * 한글 폰트가 없는 환경(계산노드 등)에서는 레이블이 자동으로 영문으로 떨어진다.
    폰트가 있으면(개발자 로컬 등) 한글로 그린다.

출력:
  1) 6패널 PNG — 보상 구성 / KL / 엔트로피 / completion 길이 / 클리핑 / step 시간
  2) step별 전체 지표 CSV (--csv)
  3) stdout 요약 — 초반 100 step 대 최근 100 step 구간 대조표
"""
import argparse
import ast
import csv
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"\{'loss'.*?\}")

# 팔레트: dataviz 검증 통과(light, surface #fcfcfb) — 대비 미달 2색은 직접 레이블로 해소
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

# 구간 대조에 쓸 지표 (보고서 §2 표와 동일 순서)
SUMMARY_KEYS = [
    "reward", "rewards/AccuracyMix/mean", "rewards/FormatThink/mean",
    "rewards/SoftOverlong/mean", "kl", "entropy/mean",
    "completions/mean_length", "completions/clipped_ratio",
    "reward_std", "frac_reward_zero_std", "step_time",
]

KO = {
    "title": "Stage-2 확장셋 GDPO 학습 곡선", "rewards": "보상 구성",
    "rewards_sub": "reward 총합과 3개 구성요소 · 가중치 1.0 / 0.2 / 0.2",
    "kl": "KL 발산", "kl_sub": "참조 정책 대비",
    "ent": "정책 엔트로피", "ent_sub": "굵은 선 = mean · 음영 = 배치 내 min–max",
    "len": "Completion 길이", "len_sub": "음영 = 배치 내 min–max · 파선 = max_completion_length",
    "clip": "Overlong 클리핑 비율", "clip_sub": "최대 길이에서 잘린 completion 비중",
    "time": "step 소요시간", "time_sub": "체크포인트 저장 구간에서 스파이크",
    "total": "reward (총합)", "x": "global step", "tokens": "tokens", "sec": "초",
    "note": "옅은 선 = 원본 · 굵은 선 = {w}-step 이동평균 · 우측 수치 = 이동평균 종점",
    "mean_of": "구간 평균 {v}",
}
EN = {
    "title": "Stage-2 expanded GDPO training curves", "rewards": "Reward components",
    "rewards_sub": "total reward and 3 components · weights 1.0 / 0.2 / 0.2",
    "kl": "KL divergence", "kl_sub": "vs reference policy",
    "ent": "Policy entropy", "ent_sub": "bold = mean · band = per-batch min-max",
    "len": "Completion length", "len_sub": "band = per-batch min-max · dashed = max_completion_length",
    "clip": "Overlong clip ratio", "clip_sub": "share of completions truncated at max length",
    "time": "Step time", "time_sub": "spikes at checkpoint saves",
    "total": "reward (total)", "x": "global step", "tokens": "tokens", "sec": "sec",
    "note": "light = raw · bold = {w}-step moving average · right label = MA endpoint",
    "mean_of": "window mean {v}",
}


def parse_logs(paths):
    """로그들에서 step 별 지표를 뽑아 step 오름차순 리스트로 반환."""
    by_step = {}
    for p in paths:
        raw = Path(p).read_text(errors="replace")
        for m in BLOCK.finditer(raw):
            try:
                d = ast.literal_eval(m.group(0))
            except Exception:
                continue
            gs = str(d.get("global_step/max_steps", ""))
            try:
                step = int(gs.split("/")[0])
            except ValueError:
                continue
            row = {"step": step}
            for k, v in d.items():
                if k == "global_step/max_steps":
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            by_step[step] = row          # 나중 파일이 이김(resume 구간)
    return [by_step[s] for s in sorted(by_step)]


def write_csv(rows, path):
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, n=100):
    """초반 n step 대 최근 n step 구간 대조 — 보고서 §2 표를 재생성한다."""
    def col(key, seq):
        vals = [r[key] for r in seq if isinstance(r.get(key), float)]
        return sum(vals) / len(vals) if vals else float("nan")

    early, late = rows[:n], rows[-n:]
    print(f"\n구간 대조 — 1~{len(early)} step 평균 vs 최근 {len(late)} step 평균\n")
    print(f"{'지표':<32}{'초반':>12}{'최근':>12}{'변화':>12}")
    print("-" * 68)
    out = {}
    for k in SUMMARY_KEYS:
        a, b = col(k, early), col(k, late)
        pct = f"{(b - a) / abs(a) * 100:+.1f}%" if a and a == a and abs(a) > 1e-9 else "—"
        print(f"{k:<32}{a:>12.4f}{b:>12.4f}{pct:>12}")
        out[k] = (a, b)
    return out


def plot(rows, out_path, window, max_steps):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.ticker import FuncFormatter

    # 한글 폰트 자동 감지 — 없으면 영문 레이블(계산노드에는 CJK 폰트가 없다)
    have = {f.name for f in fm.fontManager.ttflist}
    font = next((c for c in ("Pretendard", "NanumGothic", "Nanum Gothic",
                             "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic")
                 if c in have), None)
    L = KO if font else EN
    if not font:
        print("[plot] 한글 폰트 없음 → 영문 레이블로 렌더", file=sys.stderr)

    plt.rcParams.update({
        **({"font.family": font} if font else {}),
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlesize": 12,
    })

    step = np.array([r["step"] for r in rows], dtype=float)

    def arr(k):
        return np.array([r.get(k, np.nan) if isinstance(r.get(k), float) else np.nan
                         for r in rows], dtype=float)

    def roll(y):
        out = np.full_like(y, np.nan)
        for i in range(len(y)):
            seg = y[max(0, i - window + 1):i + 1]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                out[i] = seg.mean()
        return out

    def frame(ax, title, sub=None, ylab=None):
        ax.set_title(title, color=INK, fontweight="600", pad=14 if sub else 8, loc="left")
        if sub:
            ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED,
                    fontsize=8.5, va="bottom", ha="left")
        ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ylab:
            ax.set_ylabel(ylab, color=MUTED, fontsize=9)
        ax.set_xlim(step.min(), step.max() * 1.14)     # 직접 레이블 여백
        ax.set_xlabel(L["x"], color=MUTED, fontsize=9)

    def series(ax, key, color, label, fmt="{:.3f}"):
        y = arr(key)
        ax.plot(step, y, color=color, lw=0.8, alpha=0.16, zorder=2)
        sm = roll(y)
        ax.plot(step, sm, color=color, lw=2.0, label=label, zorder=3, solid_capstyle="round")
        ax.text(step[-1] * 1.015, sm[-1], fmt.format(sm[-1]), color=color,
                fontsize=9, fontweight="600", va="center", zorder=4)
        return sm

    fig, ax = plt.subplots(3, 2, figsize=(13.2, 11.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.20, top=0.885, bottom=0.062,
                        left=0.062, right=0.985)

    a = ax[0][0]; frame(a, L["rewards"], L["rewards_sub"])
    series(a, "reward", BLUE, L["total"])
    series(a, "rewards/FormatThink/mean", AQUA, "FormatThink")
    series(a, "rewards/AccuracyMix/mean", ORANGE, "AccuracyMix")
    series(a, "rewards/SoftOverlong/mean", YELLOW, "SoftOverlong")
    a.axhline(0, color=BASE, lw=0.8, zorder=1)
    leg = a.legend(loc="lower left", fontsize=8.5, ncol=2, labelcolor=INK2,
                   handlelength=1.6, columnspacing=1.4, frameon=True,
                   facecolor=SURFACE, edgecolor="none", framealpha=0.92)
    leg.set_zorder(5)

    a = ax[0][1]; frame(a, L["kl"], L["kl_sub"]); series(a, "kl", BLUE, "kl", "{:.4f}")

    a = ax[1][0]; frame(a, L["ent"], L["ent_sub"])
    a.fill_between(step, roll(arr("entropy/min")), roll(arr("entropy/max")),
                   color=BLUE, alpha=0.11, lw=0, zorder=1)
    series(a, "entropy/mean", BLUE, "entropy")

    a = ax[1][1]; frame(a, L["len"], L["len_sub"], L["tokens"])
    a.fill_between(step, roll(arr("completions/min_length")), roll(arr("completions/max_length")),
                   color=BLUE, alpha=0.11, lw=0, zorder=1)
    cap = float(np.nanmax(arr("completions/max_length")))
    a.axhline(cap, color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a.text(step[-1] * 0.995, cap, f"max {cap:,.0f}", color=ORANGE, fontsize=9,
           fontweight="600", va="bottom", ha="right")
    series(a, "completions/mean_length", BLUE, "mean_length", "{:,.0f}")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    a = ax[2][0]; frame(a, L["clip"], L["clip_sub"])
    series(a, "completions/clipped_ratio", ORANGE, "clipped_ratio")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    a.text(0.5, 0.93, L["mean_of"].format(v=f"{np.nanmean(arr('completions/clipped_ratio')) * 100:.1f}%"),
           transform=a.transAxes, color=INK2, fontsize=9, ha="center")

    a = ax[2][1]; frame(a, L["time"], L["time_sub"], L["sec"])
    series(a, "step_time", BLUE, "step_time", "{:,.0f}s")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    last = int(step[-1])
    pct = f" ({last / max_steps * 100:.1f}%)" if max_steps else ""
    fig.suptitle(L["title"], x=0.062, y=0.965, ha="left", color=INK,
                 fontsize=17, fontweight="700")
    fig.text(0.062, 0.928,
             f"step {last:,} / {max_steps:,}{pct}" if max_steps else f"step {last:,}",
             ha="left", color=INK2, fontsize=10.5)
    fig.text(0.985, 0.928, L["note"].format(w=window), ha="right", color=MUTED, fontsize=9)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    print(f"[plot] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="학습 로그 (여러 개면 step 기준 병합)")
    ap.add_argument("-o", "--out", default="docs/assets/train_curves.png")
    ap.add_argument("--csv", help="step별 전체 지표를 CSV 로도 저장")
    ap.add_argument("-w", "--window", type=int, default=25, help="이동평균 창 (기본 25)")
    ap.add_argument("--max-steps", type=int, default=0, help="제목의 진행률 분모 (0=생략)")
    ap.add_argument("--no-plot", action="store_true", help="CSV·요약만 (matplotlib 불필요)")
    args = ap.parse_args()

    rows = parse_logs(args.logs)
    if not rows:
        sys.exit("지표 블록을 하나도 파싱하지 못했다 — 로그 경로를 확인할 것")
    steps = [r["step"] for r in rows]
    gaps = [s for s in range(steps[0], steps[-1] + 1) if s not in set(steps)]
    print(f"[parse] {len(rows)} rows · step {steps[0]}..{steps[-1]}"
          + (f" · 결측 {len(gaps)}" if gaps else " · 결측 없음"))

    if args.csv:
        write_csv(rows, args.csv)
        print(f"[csv] saved: {args.csv}")
    summarize(rows)
    if not args.no_plot:
        plot(rows, args.out, args.window, args.max_steps)


if __name__ == "__main__":
    main()
