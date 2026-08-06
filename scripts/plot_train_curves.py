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
    "kl": "KL 발산 · grad_norm", "kl_sub": "둘 다 형식 붕괴보다 ~50 step 먼저 상승 (붕괴 선행지표)",
    "ent": "정책 엔트로피", "ent_sub": "굵은 선 = mean · 음영 = 배치 내 min–max",
    "len": "Completion 길이", "len_sub": "음영 = 배치 내 min–max · 파선 = max_completion_length",
    "clip": "Overlong 클리핑 비율", "clip_sub": "최대 길이에서 잘린 completion 비중",
    "time": "step 소요시간", "time_sub": "체크포인트 저장 구간에서 스파이크",
    "total": "reward (총합)", "x": "global step", "tokens": "tokens", "sec": "초",
    "note": "옅은 선 = 원본 · 굵은 선 = {w}-step 이동평균 · 우측 수치 = 이동평균 종점",
    "mean_of": "구간 평균 {v}",
    "ev_best": "850 최고 ckpt", "ev_break": "899–904 형식 붕괴", "ev_len": "905 길이 폭주",
}
EN = {
    "title": "Stage-2 expanded GDPO training curves", "rewards": "Reward components",
    "rewards_sub": "total reward and 3 components · weights 1.0 / 0.2 / 0.2",
    "kl": "KL divergence · grad_norm", "kl_sub": "both rise ~50 steps before format collapse (leading indicators)",
    "ent": "Policy entropy", "ent_sub": "bold = mean · band = per-batch min-max",
    "len": "Completion length", "len_sub": "band = per-batch min-max · dashed = max_completion_length",
    "clip": "Overlong clip ratio", "clip_sub": "share of completions truncated at max length",
    "time": "Step time", "time_sub": "spikes at checkpoint saves",
    "total": "reward (total)", "x": "global step", "tokens": "tokens", "sec": "sec",
    "note": "light = raw · bold = {w}-step moving average · right label = MA endpoint",
    "mean_of": "window mean {v}",
    "ev_best": "850 best ckpt", "ev_break": "899-904 format collapse", "ev_len": "905 length runaway",
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


def plot(rows, out_path, window, max_steps, status="", draw_events=True):
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
        return ax.text(step[-1] * 1.015, sm[-1], fmt.format(sm[-1]), color=color,
                       fontsize=9, fontweight="600", va="center", zorder=4)

    # 붕괴 사후분석(docs/stage2_run73924_postmortem.md)에서 확정된 사건 위치.
    #  형식 붕괴(899~904)와 길이 폭주(905~)는 별개 사건이고 순서가 있다 — 같은 선으로 묶지 말 것.
    #  899 와 905 는 6 step 차라 라벨이 겹친다 → 높이를 계단식으로 어긋내고 전부 왼쪽으로 뽑는다.
    EVENTS = [(850, L["ev_best"], AQUA, 0.97),
              (899, L["ev_break"], ORANGE, 0.88),
              (905, L["ev_len"], YELLOW, 0.79)]

    def events(ax, label=False):
        """사건 위치에 세로 마커. label=True 인 패널에만 글자를 얹는다(6패널 반복 방지)."""
        if not draw_events:
            return
        for x, txt, c, ly in EVENTS:
            if not (step.min() <= x <= step.max()):
                continue
            ax.axvline(x, color=c, lw=1.0, ls=(0, (3, 3)), alpha=0.85, zorder=1.5)
            if label:
                ax.text(x - (step.max() - step.min()) * 0.012, ly, txt,
                        transform=ax.get_xaxis_transform(), color=c, fontsize=8.5,
                        fontweight="600", ha="right", va="center", zorder=6)

    def spread(ax, texts, pad_px=13):
        """끝점 라벨이 겹치면 세로로 밀어낸다.

        보상 패널은 계열 4개의 종점이 붙을 수 있다(붕괴 후 0.31/0.29/0.26 처럼).
        데이터 단위로 밀면 패널마다 스케일이 달라 튜닝이 안 되므로 화면 픽셀로 민다.
        """
        if len(texts) < 2:
            return
        ax.figure.canvas.draw()                       # 좌표 확정
        T, Ti = ax.transData.transform, ax.transData.inverted().transform
        items = sorted(texts, key=lambda t: t.get_position()[1])
        ys = [T((0, t.get_position()[1]))[1] for t in items]
        want = list(ys)
        for i in range(1, len(want)):
            want[i] = max(want[i], want[i - 1] + pad_px)
        shift = (sum(ys) - sum(want)) / len(ys)       # 밀어낸 뒤 원래 중심으로 되돌린다
        for t, y in zip(items, want):
            t.set_position((t.get_position()[0], Ti((0, y + shift))[1]))

    fig, ax = plt.subplots(3, 2, figsize=(13.2, 11.6))
    #  right 는 0.94 까지만 — KL 패널의 twinx(grad_norm) 눈금·축라벨이 그 바깥에 그려진다.
    fig.subplots_adjust(hspace=0.52, wspace=0.20, top=0.885, bottom=0.062,
                        left=0.062, right=0.940)

    a = ax[0][0]; frame(a, L["rewards"], L["rewards_sub"])
    spread(a, [series(a, "reward", BLUE, L["total"]),
               series(a, "rewards/FormatThink/mean", AQUA, "FormatThink"),
               series(a, "rewards/AccuracyMix/mean", ORANGE, "AccuracyMix"),
               series(a, "rewards/SoftOverlong/mean", YELLOW, "SoftOverlong")])
    a.axhline(0, color=BASE, lw=0.8, zorder=1)
    leg = a.legend(loc="lower left", fontsize=8.5, ncol=2, labelcolor=INK2,
                   handlelength=1.6, columnspacing=1.4, frameon=True,
                   facecolor=SURFACE, edgecolor="none", framealpha=0.92)
    leg.set_zorder(5)

    events(a)

    #  KL 과 grad_norm 은 단위가 달라 축을 나눈다. 둘의 "동시 상승"이 이 패널의 요점이므로
    #  같은 패널에 겹쳐 그리는 편이 두 패널로 나누는 것보다 읽기 쉽다.
    #  두 축 모두 로그 — kl 은 0.0004~1.1, grad_norm 은 0.009~10.0 으로 3 decade 를 넘는다.
    #  선형축이면 말기 스파이크(grad_norm 10.02 @ step 938)에 눌려 정작 중요한
    #  step 850~900 구간의 2~3배 상승이 0 근처에 붙어 보이지 않는다.
    a = ax[0][1]; frame(a, L["kl"], L["kl_sub"])
    series(a, "kl", BLUE, "kl", "{:.4f}")
    a.set_yscale("log")
    a2 = a.twinx()
    a2.set_yscale("log")
    a2.set_xlim(a.get_xlim())
    for s in ("top", "left"):
        a2.spines[s].set_visible(False)
    a2.spines["right"].set_color(BASE)
    a2.tick_params(colors=MUTED, labelsize=9)
    yg = arr("grad_norm")
    a2.plot(step, yg, color=ORANGE, lw=0.8, alpha=0.16, zorder=2)
    a2.plot(step, roll(yg), color=ORANGE, lw=2.0, zorder=3, solid_capstyle="round")
    a2.set_ylabel("grad_norm", color=ORANGE, fontsize=9)
    a.plot([], [], color=ORANGE, lw=2.0, label="grad_norm")     # 범례용 프록시
    a.legend(loc="upper left", fontsize=8.5, labelcolor=INK2, handlelength=1.6,
             frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92)
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    #  사건 라벨은 이 패널에만 얹는다 — 붕괴 시점(x≈0.75) 위쪽이 6패널 중 유일하게 비어 있다.
    events(a, label=True)

    a = ax[1][0]; frame(a, L["ent"], L["ent_sub"])
    a.fill_between(step, roll(arr("entropy/min")), roll(arr("entropy/max")),
                   color=BLUE, alpha=0.11, lw=0, zorder=1)
    series(a, "entropy/mean", BLUE, "entropy")

    events(a)

    a = ax[1][1]; frame(a, L["len"], L["len_sub"], L["tokens"])
    a.fill_between(step, roll(arr("completions/min_length")), roll(arr("completions/max_length")),
                   color=BLUE, alpha=0.11, lw=0, zorder=1)
    cap = float(np.nanmax(arr("completions/max_length")))
    a.axhline(cap, color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a.text(step[-1] * 0.995, cap, f"max {cap:,.0f}", color=ORANGE, fontsize=9,
           fontweight="600", va="bottom", ha="right")
    series(a, "completions/mean_length", BLUE, "mean_length", "{:,.0f}")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    events(a)

    a = ax[2][0]; frame(a, L["clip"], L["clip_sub"])
    series(a, "completions/clipped_ratio", ORANGE, "clipped_ratio")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    a.text(0.5, 0.93, L["mean_of"].format(v=f"{np.nanmean(arr('completions/clipped_ratio')) * 100:.1f}%"),
           transform=a.transAxes, color=INK2, fontsize=9, ha="center")

    events(a)

    a = ax[2][1]; frame(a, L["time"], L["time_sub"], L["sec"])
    series(a, "step_time", BLUE, "step_time", "{:,.0f}s")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    last = int(step[-1])
    pct = f" ({last / max_steps * 100:.1f}%)" if max_steps else ""
    head = f"step {last:,} / {max_steps:,}{pct}" if max_steps else f"step {last:,}"
    if status:
        head += f" · {status}"
    fig.suptitle(L["title"], x=0.062, y=0.965, ha="left", color=INK,
                 fontsize=17, fontweight="700")
    fig.text(0.062, 0.928, head, ha="left", color=INK2, fontsize=10.5)
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
    ap.add_argument("--status", default="", help="부제에 덧붙일 상태 문구 (예: 취소·붕괴)")
    ap.add_argument("--no-plot", action="store_true", help="CSV·요약만 (matplotlib 불필요)")
    ap.add_argument("--no-events", action="store_true",
                    help="붕괴 사건 마커(850·899·905) 숨김 — 73924/73925 이외 런에 쓸 것")
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
        plot(rows, args.out, args.window, args.max_steps, args.status,
             draw_events=not args.no_events)


if __name__ == "__main__":
    main()
