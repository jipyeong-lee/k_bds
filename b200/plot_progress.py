#!/usr/bin/env python3
"""parse_log.py 가 뽑은 CSV 를 학습 곡선 PNG 로 그린다.

    python3 b200/plot_progress.py b200/metrics_deepvision.csv b200/progress_deepvision.png

CSV 는 전 step 이 들어 있다 — step 별 값은 배치 난이도로 크게 튀므로 원본은 흐리게 깔고
이동평균으로 추세를 본다. 설정이 바뀐 지점은 세로 점선으로 표시한다.
곡선만 보면 "왜 여기서 꺾였나"를 알 수 없어서다.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 한글 라벨이 □ 로 깨지는 걸 막는다. 없는 환경에서는 조용히 기본 폰트로 떨어진다.
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _cand in ("NanumGothic", "Malgun Gothic", "Noto Sans CJK KR", "AppleGothic"):
    if _cand in _have:
        matplotlib.rcParams["font.family"] = _cand
        break
matplotlib.rcParams["axes.unicode_minus"] = False

src = sys.argv[1] if len(sys.argv) > 1 else "b200/metrics_deepvision.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "b200/progress_deepvision.png"

df = pd.read_csv(src).apply(pd.to_numeric, errors="coerce").dropna(subset=["step"])
df = df.sort_values("step")

# 설정을 바꾼 지점. 실측 근거는 b200/README.md 에 있다.
# 실행마다 이력이 달라서 CSV 이름으로 고른다 — 1 차의 표시가 2 차 그림에 찍히면 거짓말이 된다.
MARKS_BY_RUN = {
    # PDTBS 는 job #5(step 112 재개)부터 2×8 이다 — memory 곡선이 143→85 로 꺾이는 자리와 일치한다.
    "metrics_deepvision.csv": [
        (112, "PDTBS 4→2"),
        (284, "beta 0.04→0\ntemp 0.9→1.0"),
    ],
    # 2 차는 처음부터 엔트로피 마스크로 돌아서 도중 변경이 없다.
    "metrics_deepvision_entmask.csv": [],
}
MARKS = MARKS_BY_RUN.get(os.path.basename(src), [])

# 엔트로피 마스크 실행에만 있는 열이다. 1 차 CSV 로 그리면 빈 칸이라 곡선을 건너뛴다.
ENTMASK = "ent" in df.columns and df["ent"].notna().any()

# step 별 값은 배치 난이도 때문에 크게 튄다 — 원본은 흐리게 깔고 이동평균으로 추세를 본다.
# 창이 너무 넓으면 붕괴 같은 짧은 사건이 뭉개지므로 40 에서 끊는다.
WIN = min(40, max(3, len(df) // 12))


def trend(a, x, y, color, label, raw_alpha=0.15):
    a.plot(x, y, lw=0.6, color=color, alpha=raw_alpha)
    a.plot(x, y.rolling(WIN, min_periods=1, center=True).mean(),
           lw=1.8, color=color, label=label)

fig, ax = plt.subplots(2, 3, figsize=(19, 8))
fig.suptitle(
    "Stage-2 RLVR · deepvision (dr_grpo + gdpo + TIS, async rollout)"
    + (" + entropy mask top 0.2" if ENTMASK else ""),
    fontsize=13, fontweight="bold",
)


def marks(a):
    for x, label in MARKS:
        if df["step"].min() <= x <= df["step"].max():
            a.axvline(x, color="0.55", ls=":", lw=1)
            # y 를 축 비율로 잡아야 어떤 스케일에서도 라벨이 잘리지 않는다.
            a.text(x, 0.97, f" {label}", transform=a.get_xaxis_transform(),
                   fontsize=7, color="0.35", va="top", ha="left")


# 1) 보상 — 학습이 실제로 나아지고 있는지
a = ax[0][0]
trend(a, df["step"], df["reward"], "#1f77b4", "reward")
trend(a, df["step"], df["acc"], "#2ca02c", "AccuracyMix")
trend(a, df["step"], df["fmt"], "#ff7f0e", "FormatThink")
a.set_title(f"보상 (굵은 선 = {WIN}점 이동평균)", fontsize=10)
a.set_xlabel("step"); a.set_ylim(0, 1.05)
a.legend(fontsize=8, loc="lower left"); a.grid(alpha=0.3); marks(a)

# 2) 학습·추론 확률 불일치 — TIS 가 실제로 붙들고 있는지
a = ax[0][1]
trend(a, df["step"], df["ppl_abs_diff"], "#d62728", "log_ppl_abs_diff (진단)")
a.axhline(0.05, color="#d62728", ls="--", lw=0.9, alpha=0.6)
a.text(df["step"].max(), 0.051, "IcePop 임계 5%", fontsize=7,
       color="#d62728", ha="right", va="bottom")
a.set_ylabel("불일치", fontsize=8); a.set_xlabel("step")
a2 = a.twinx()
a2.plot(df["step"], df["ess"], lw=1.4, color="#9467bd", label="ESS (보정 후)")
a2.set_ylabel("ESS", fontsize=8); a2.set_ylim(0.9, 1.005)
a.set_title("off-policy: 진단값 vs 보정 후 유효표본", fontsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
a.grid(alpha=0.3); marks(a)

# 3) 학습 모델 vs 롤아웃 모델의 perplexity — 둘이 벌어지는 것이 붕괴의 실체다.
#    ro_ppl 이 내려가면 롤아웃 분포가 뾰족해진 것(엔트로피 붕괴), tr_ppl 이 올라가면
#    학습 모델이 그 생성을 덜 그럴듯하게 보는 것이다. 비율이 벌어지면 둘이 다른 모델이 된다.
#    tr_ppl 은 가끔 수십까지 튀므로 추세가 안 보이게 되는 걸 막으려고 5 에서 자른다.
a = ax[0][2]
trend(a, df["step"], df["tr_ppl"].clip(upper=5), "#d62728", "학습 ppl")
trend(a, df["step"], df["ro_ppl"].clip(upper=5), "#1f77b4", "롤아웃 ppl")
a.set_xlabel("step"); a.set_ylabel("perplexity", fontsize=8)
a2 = a.twinx()
a2.plot(df["step"], (df["tr_ppl"] / df["ro_ppl"]).clip(upper=3)
        .rolling(WIN, min_periods=1, center=True).mean(),
        lw=1.4, color="0.35", ls="--", label="비율 tr/ro")
a2.set_ylabel("비율", fontsize=8); a2.set_ylim(0.9, 2.2)
a.set_title("학습 ppl vs 롤아웃 ppl (괴리)", fontsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
a.grid(alpha=0.3); marks(a)

# 4) 생성 길이 — 붕괴는 길이에서 먼저 보인다
a = ax[1][0]
trend(a, df["step"], df["len"], "#8c564b", "completion 평균 길이")
a.set_xlabel("step"); a.set_ylabel("tokens", fontsize=8)
a2 = a.twinx()
a2.plot(df["step"], df["clipped"], lw=1.2, color="#e377c2", label="clipped_ratio")
a2.set_ylabel("clipped", fontsize=8)
a.set_title("생성 길이 / 잘림 비율", fontsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8); a.grid(alpha=0.3); marks(a)

# 5) 학습 신호가 남아 있는지 — zero_std 가 오르면 그룹 내 보상이 전부 같아 advantage 가 0 이 된다.
a = ax[1][1]
trend(a, df["step"], df["zero_std"], "#d62728", "advantage 0 비율")
trend(a, df["step"], df["r_std"], "#2ca02c", "그룹 보상 std")
if ENTMASK:
    # 1 차 붕괴의 실체가 엔트로피 소실이었다 — 마스크가 이걸 붙들고 있는지가 이 실행의 전부다.
    trend(a, df["step"], df["ent"], "#17becf", "정책 엔트로피")
a.set_xlabel("step"); a.set_ylim(0, 0.85 if ENTMASK else 0.6)
a2 = a.twinx()
a2.plot(df["step"], df["grad"].rolling(WIN, min_periods=1, center=True).mean(),
        lw=1.4, color="#bcbd22", label="grad_norm")
a2.set_ylabel("grad_norm", fontsize=8)
a.set_title("학습 신호 (advantage 소실 / 엔트로피 / 그래디언트)"
            if ENTMASK else "학습 신호 (advantage 소실 / 그래디언트)", fontsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
a.grid(alpha=0.3); marks(a)

# 6) 자원 — 배치를 되돌린 효과가 여기서 보인다
a = ax[1][2]
a.plot(df["step"], df["mem"], lw=1.4, color="#17becf", label="memory(GiB)")
a.axhline(180, color="0.5", ls="--", lw=0.9)
a.text(df["step"].max(), 178, "B200 180 GiB", fontsize=7, color="0.4",
       ha="right", va="top")
a.set_ylim(0, 195); a.set_xlabel("step"); a.set_ylabel("GiB", fontsize=8)
a2 = a.twinx()
a2.plot(df["step"], df["step_time"], lw=1.0, color="#bcbd22", alpha=0.75,
        label="step_time(s)")
a2.set_ylabel("sec", fontsize=8)
a.set_title("GPU 메모리 / step 시간", fontsize=10)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
a.grid(alpha=0.3); marks(a)

last = df.iloc[-1]
fig.text(
    0.5, 0.005,
    f"step {int(last['step'])}/5715 · reward {last['reward']:.3f} · "
    f"AccuracyMix {last['acc']:.3f} · FormatThink {last['fmt']:.3f} · "
    f"ESS {last['ess']:.4f} · 불일치 {last['ppl_abs_diff']:.3f} · mem {last['mem']:.1f} GiB"
    + (f" · 엔트로피 {last['ent']:.3f}" if ENTMASK else ""),
    ha="center", fontsize=8, color="0.3",
)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(dst, dpi=130)
print(f"saved {dst}  (step {int(df['step'].min())}~{int(df['step'].max())}, {len(df)} points)")
